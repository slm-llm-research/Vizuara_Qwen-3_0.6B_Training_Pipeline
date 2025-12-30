#!/usr/bin/env python3
"""
Qwen3-0.6B Training Pipeline (Refactored with Class-Based Structure)

A modular, class-based training pipeline for fine-tuning Qwen3-0.6B on AI/Human text detection.

Features:
- Clean class-based architecture
- CLI argument support
- Modular components
- Easy to test and extend
- All functionality from standalone script

Usage:
    # Basic usage
    python train_qwen.py --dataset 10k
    
    # With custom hyperparameters
    python train_qwen.py --dataset 1M --epochs 5 --batch-size 16 --learning-rate 2e-4
    
    # Disable WandB
    python train_qwen.py --dataset 100k --no-wandb
    
    # Resume from checkpoint
    python train_qwen.py --dataset 10k --resume ./output/qwen3-0.6b-10k/checkpoint-1000
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# HuggingFace
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import wandb

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

WANDB_API_KEY = "939a604fbdaf6ec2e540b7d12232eb267ec5bab2"

DATASET_OPTIONS = {
    "10k": {
        "name": "codefactory4791/raid_aligned_10k",
        "size": "~10K samples",
        "balance": "Balanced (50/50)",
        "class_weights": [1.0, 1.0]
    },
    "100k": {
        "name": "codefactory4791/raid_aligned_100k",
        "size": "~100K samples",
        "balance": "Balanced (50/50)",
        "class_weights": [1.0, 1.0]
    },
    "1M": {
        "name": "codefactory4791/raid_aligned_1000k",
        "size": "~1M samples",
        "balance": "Slightly Imbalanced (AI: 52.17%, Human: 47.83%)",
        "class_weights": [0.9167, 1.0909]
    },
    "2M": {
        "name": "codefactory4791/raid_aligned_2000k",
        "size": "~2M samples",
        "balance": "Imbalanced (AI: 54.57%, Human: 45.43%)",
        "class_weights": [0.8324, 1.2009]
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


def create_label_mappings(labels: List[str]) -> Tuple[Dict, Dict]:
    """Create label to ID and ID to label mappings"""
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    return label2id, id2label


def compute_class_weights(dataset, label_column: str, label2id: Dict) -> torch.Tensor:
    """Compute class weights using inverse frequency"""
    label_counts = {}
    for sample in dataset:
        label = sample[label_column]
        label_id = label2id[label]
        label_counts[label_id] = label_counts.get(label_id, 0) + 1
    
    total_samples = sum(label_counts.values())
    num_classes = len(label2id)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def save_metrics_to_file(metrics: Dict, filepath: str):
    """Save metrics to a text file"""
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to: {filepath}")


def get_cache_key(config: dict) -> str:
    """Generate a cache key based on tokenization settings"""
    key_data = {
        'model': config['model']['name'],
        'max_length': config['tokenization']['max_length'],
        'padding': config['tokenization']['padding'],
        'dataset': config['dataset']['dataset_name'],
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


# ============================================================================
# CUSTOM TRAINER
# ============================================================================

class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.detach().clone().float()
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = self.class_weights.to(self.args.device)
        else:
            self.class_weights = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted cross-entropy loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# MAIN TRAINING PIPELINE CLASS
# ============================================================================

class QwenTrainingPipeline:
    """
    Complete training pipeline for Qwen3-0.6B fine-tuning.
    
    This class encapsulates all stages of the training process:
    - Configuration and setup
    - Dataset loading and preprocessing
    - Model initialization
    - Training execution
    - Evaluation and saving
    """
    
    def __init__(
        self,
        dataset: str,
        config_path: str = "config.yaml",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        lora_rank: Optional[int] = None,
        enable_wandb: bool = True,
        resume_from: Optional[str] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            dataset: Dataset size to use ("10k", "100k", "1M", "2M")
            config_path: Path to config.yaml file
            epochs: Number of training epochs (overrides config)
            batch_size: Batch size per device (overrides config)
            learning_rate: Learning rate (overrides config)
            lora_rank: LoRA rank (overrides config)
            enable_wandb: Whether to enable WandB logging
            resume_from: Path to checkpoint to resume from
        """
        self.dataset_key = dataset
        self.config_path = config_path
        self.enable_wandb = enable_wandb
        self.resume_from = resume_from
        
        # Override parameters
        self.override_epochs = epochs
        self.override_batch_size = batch_size
        self.override_learning_rate = learning_rate
        self.override_lora_rank = lora_rank
        
        # Will be initialized during pipeline execution
        self.config = None
        self.dataset_info = None
        self.dataset_name = None
        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_tokenized = None
        self.val_tokenized = None
        self.test_tokenized = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label2id = None
        self.id2label = None
        self.labels = None
        self.pre_metrics = None
        self.post_metrics = None
    
    def load_config(self):
        """Load and configure settings"""
        print_section("Loading Configuration")
        
        # Validate dataset selection
        if self.dataset_key not in DATASET_OPTIONS:
            raise ValueError(
                f"Invalid dataset: {self.dataset_key}. "
                f"Choose from: {list(DATASET_OPTIONS.keys())}"
            )
        
        self.dataset_info = DATASET_OPTIONS[self.dataset_key]
        self.dataset_name = self.dataset_info["name"]
        self.class_weights = self.dataset_info["class_weights"]
        
        print(f"Selected Dataset: {self.dataset_key}")
        print(f"HuggingFace Dataset: {self.dataset_name}")
        print(f"Size: {self.dataset_info['size']}")
        print(f"Label Balance: {self.dataset_info['balance']}")
        print(f"Class Weights: {self.class_weights}")
        
        # Load configuration file
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update config with dataset selection
        self.config['dataset']['dataset_name'] = self.dataset_name
        self.config['class_weights']['manual_weights'] = self.class_weights
        
        # Update paths
        self.config['training']['output_dir'] = f"./output/qwen3-0.6b-{self.dataset_key}"
        self.config['training']['logging_dir'] = f"./logs/qwen3-0.6b-{self.dataset_key}"
        self.config['misc']['tokenized_cache_dir'] = f"./tokenized_cache/{self.dataset_key}"
        
        # Update run name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.config['training']['run_name'] = f"qwen3-0.6b-{self.dataset_key}-{timestamp}"
        
        # Update wandb settings
        self.config['wandb']['enabled'] = self.enable_wandb
        self.config['wandb']['tags'].append(f"dataset-{self.dataset_key}")
        
        # Apply CLI overrides
        if self.override_epochs is not None:
            self.config['training']['num_train_epochs'] = self.override_epochs
            print(f"Override: epochs = {self.override_epochs}")
        
        if self.override_batch_size is not None:
            self.config['training']['per_device_train_batch_size'] = self.override_batch_size
            self.config['training']['per_device_eval_batch_size'] = self.override_batch_size
            print(f"Override: batch_size = {self.override_batch_size}")
        
        if self.override_learning_rate is not None:
            self.config['training']['learning_rate'] = self.override_learning_rate
            print(f"Override: learning_rate = {self.override_learning_rate}")
        
        if self.override_lora_rank is not None:
            self.config['peft']['lora_r'] = self.override_lora_rank
            self.config['peft']['lora_alpha'] = self.override_lora_rank * 2
            print(f"Override: lora_rank = {self.override_lora_rank}")
        
        if self.resume_from is not None:
            self.config['misc']['resume_from_checkpoint'] = self.resume_from
            print(f"Will resume from: {self.resume_from}")
        
        print(f"\nModel: {self.config['model']['name']}")
        print(f"Output directory: {self.config['training']['output_dir']}")
        print(f"Run name: {self.config['training']['run_name']}")
    
    def initialize_wandb(self):
        """Initialize Weights & Biases logging"""
        if not self.config['wandb']['enabled']:
            print("WandB disabled")
            return
        
        print_section("Initializing Weights & Biases")
        
        wandb.login(key=WANDB_API_KEY)
        
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=self.config['training']['run_name'],
            tags=self.config['wandb']['tags'],
            notes=self.config['wandb']['notes'],
            config={
                'dataset': self.dataset_name,
                'dataset_size': self.dataset_key,
                'model': self.config['model']['name'],
                'batch_size': self.config['training']['per_device_train_batch_size'],
                'gradient_accumulation_steps': self.config['training']['gradient_accumulation_steps'],
                'effective_batch_size': self.config['training']['per_device_train_batch_size'] * 
                                       self.config['training']['gradient_accumulation_steps'],
                'learning_rate': self.config['training']['learning_rate'],
                'num_epochs': self.config['training']['num_train_epochs'],
                'lora_r': self.config['peft']['lora_r'],
                'lora_alpha': self.config['peft']['lora_alpha'],
                'max_length': self.config['tokenization']['max_length'],
                'class_weights': self.class_weights,
            }
        )
        
        print("WandB initialized!")
        print(f"Dashboard: {wandb.run.get_url()}")
    
    def load_datasets(self):
        """Load train/val/test datasets from HuggingFace"""
        print_section("Loading Datasets")
        
        print(f"Loading dataset: {self.dataset_name}...")
        print("This may take a few minutes...\n")
        
        dataset = load_dataset(self.dataset_name)
        
        print("Dataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Extract splits
        self.train_dataset = dataset[self.config['dataset']['train_split']]
        self.val_dataset = dataset[self.config['dataset']['validation_split']]
        self.test_dataset = dataset[self.config['dataset']['test_split']]
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(self.train_dataset):,} samples")
        print(f"  Validation: {len(self.val_dataset):,} samples")
        print(f"  Test: {len(self.test_dataset):,} samples")
        
        # Show label distribution
        label_column = self.config['model']['label_column']
        print(f"\nLabel distribution in training set:")
        train_labels = pd.Series([sample[label_column] for sample in self.train_dataset])
        print(train_labels.value_counts())
        print(f"\nNormalized:")
        print(train_labels.value_counts(normalize=True))
    
    def create_label_mappings(self):
        """Create label mappings and compute class weights"""
        print_section("Creating Label Mappings")
        
        self.labels = self.config['model']['labels']
        self.label2id, self.id2label = create_label_mappings(self.labels)
        
        print(f"Label to ID: {self.label2id}")
        print(f"ID to Label: {self.id2label}")
        
        # Get class weights
        if self.config['class_weights']['enabled']:
            if self.config['class_weights']['manual_weights']:
                class_weights_tensor = torch.tensor(
                    self.config['class_weights']['manual_weights'],
                    dtype=torch.float32
                )
                print(f"\nUsing pre-computed class weights: {class_weights_tensor.tolist()}")
            else:
                label_column = self.config['model']['label_column']
                class_weights_tensor = compute_class_weights(
                    self.train_dataset, label_column, self.label2id
                )
                print(f"\nComputed class weights: {class_weights_tensor.tolist()}")
            
            self.class_weights = class_weights_tensor
            
            print(f"\nClass weight interpretation:")
            for i, label in enumerate(self.labels):
                print(f"  {label}: {self.class_weights[i]:.4f}")
        else:
            self.class_weights = None
            print("\nClass weights disabled")
    
    def setup_tokenizer(self):
        """Load and configure tokenizer"""
        print_section("Loading Tokenizer")
        
        print(f"Loading tokenizer: {self.config['model']['name']}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            add_prefix_space=self.config['tokenization']['add_prefix_space'],
            use_fast=self.config['misc']['use_fast_tokenizer'],
            trust_remote_code=self.config['misc']['trust_remote_code'],
            cache_dir=self.config['misc']['cache_dir'],
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
        
        print(f"\nTokenizer loaded!")
        print(f"Vocab size: {len(self.tokenizer)}")
        print(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
    
    def tokenize_datasets(self):
        """Tokenize datasets with smart caching"""
        print_section("Tokenizing Datasets")
        
        text_column = self.config['model']['text_column']
        label_column = self.config['model']['label_column']
        
        def preprocess_function(examples):
            """Tokenize text and map labels to IDs"""
            texts = []
            for text in examples[text_column]:
                if text is None or (isinstance(text, float) and pd.isna(text)):
                    texts.append("")
                else:
                    texts.append(str(text))
            
            tokenized = self.tokenizer(
                texts,
                padding=self.config['tokenization']['padding'],
                truncation=self.config['tokenization']['truncation'],
                max_length=self.config['tokenization']['max_length'],
            )
            
            tokenized['labels'] = [self.label2id[label] for label in examples[label_column]]
            
            return tokenized
        
        # Check for cached tokenized datasets
        cache_dir = self.config['misc']['tokenized_cache_dir']
        use_cache = self.config['misc']['save_tokenized_datasets']
        force_retokenize = self.config['misc']['force_retokenize']
        
        cache_key = get_cache_key(self.config)
        cache_path_train = os.path.join(cache_dir, f'train_{cache_key}')
        cache_path_val = os.path.join(cache_dir, f'val_{cache_key}')
        cache_path_test = os.path.join(cache_dir, f'test_{cache_key}')
        
        cache_exists = (
            os.path.exists(cache_path_train) and 
            os.path.exists(cache_path_val) and 
            os.path.exists(cache_path_test)
        )
        
        if use_cache and cache_exists and not force_retokenize:
            print("Loading cached tokenized datasets...")
            print(f"Cache key: {cache_key}\n")
            
            self.train_tokenized = load_from_disk(cache_path_train)
            self.val_tokenized = load_from_disk(cache_path_val)
            self.test_tokenized = load_from_disk(cache_path_test)
            
            print(f"Loaded cached datasets!")
            print(f"  Train: {len(self.train_tokenized):,} samples")
            print(f"  Validation: {len(self.val_tokenized):,} samples")
            print(f"  Test: {len(self.test_tokenized):,} samples")
        else:
            print("Tokenizing datasets...")
            
            # Filter invalid entries
            def is_valid(example):
                text = example[text_column]
                label = example[label_column]
                text_valid = text is not None and not (isinstance(text, float) and pd.isna(text)) and str(text).strip() != ""
                label_valid = label is not None and label in self.label2id
                return text_valid and label_valid
            
            self.train_dataset = self.train_dataset.filter(is_valid, desc="Filtering train")
            self.val_dataset = self.val_dataset.filter(is_valid, desc="Filtering val")
            self.test_dataset = self.test_dataset.filter(is_valid, desc="Filtering test")
            
            print(f"\nAfter filtering:")
            print(f"  Train: {len(self.train_dataset):,}")
            print(f"  Val: {len(self.val_dataset):,}")
            print(f"  Test: {len(self.test_dataset):,}")
            
            # Tokenize
            self.train_tokenized = self.train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.train_dataset.column_names,
                desc="Tokenizing train",
            )
            
            self.val_tokenized = self.val_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.val_dataset.column_names,
                desc="Tokenizing val",
            )
            
            self.test_tokenized = self.test_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.test_dataset.column_names,
                desc="Tokenizing test",
            )
            
            print(f"\nTokenization complete!")
            
            # Save to cache
            if use_cache:
                print(f"\nSaving to cache...")
                os.makedirs(cache_dir, exist_ok=True)
                
                self.train_tokenized.save_to_disk(cache_path_train)
                self.val_tokenized.save_to_disk(cache_path_val)
                self.test_tokenized.save_to_disk(cache_path_test)
                
                print(f"Cached at: {cache_dir}")
    
    def setup_model(self):
        """Load model with quantization and LoRA"""
        print_section("Loading Model")
        
        # Configure quantization
        quantization_config = None
        if self.config['quantization']['enabled']:
            print("Setting up quantization...")
            
            compute_dtype = getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype'])
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config['quantization']['load_in_4bit'],
                load_in_8bit=self.config['quantization']['load_in_8bit'],
                bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
                bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_compute_dtype=compute_dtype,
            )
            
            print(f"Quantization: 4-bit" if self.config['quantization']['load_in_4bit'] else "8-bit")
        
        print(f"\nLoading model: {self.config['model']['name']}...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['name'],
            num_labels=self.config['model']['num_labels'],
            id2label=self.id2label,
            label2id=self.label2id,
            quantization_config=quantization_config,
            device_map=self.config['hardware']['device_map'],
            max_memory=self.config['hardware']['max_memory'],
            trust_remote_code=self.config['misc']['trust_remote_code'],
            cache_dir=self.config['misc']['cache_dir'],
        )
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        
        print(f"\nModel loaded!")
        print(f"Parameters: {self.model.num_parameters():,}")
        print_gpu_memory()
    
    def apply_lora(self):
        """Apply LoRA/PEFT to model"""
        print_section("Applying LoRA")
        
        if not self.config['peft']['enabled']:
            print("PEFT disabled - using full fine-tuning")
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
            return
        
        print("Applying LoRA...")
        
        # Prepare for k-bit training
        if self.config['quantization']['enabled']:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=self.config['peft']['lora_r'],
            lora_alpha=self.config['peft']['lora_alpha'],
            target_modules=self.config['peft']['target_modules'],
            lora_dropout=self.config['peft']['lora_dropout'],
            bias=self.config['peft']['bias'],
            task_type=TaskType.SEQ_CLS,
            modules_to_save=self.config['peft']['modules_to_save'],
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        print(f"\nLoRA applied!")
        print(f"Rank: {self.config['peft']['lora_r']}")
        print(f"Alpha: {self.config['peft']['lora_alpha']}")
        
        self.model.print_trainable_parameters()
        print_gpu_memory()
    
    def create_trainer(self):
        """Initialize trainer with callbacks"""
        print_section("Initializing Trainer")
        
        # Data collator
        if self.config['data_collator']['type'] == 'DataCollatorWithPadding':
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=self.config['data_collator']['padding'],
                pad_to_multiple_of=self.config['data_collator']['pad_to_multiple_of'],
            )
        else:
            data_collator = None
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            balanced_acc = balanced_accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        
        # Training arguments
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate eval steps
        steps_per_epoch = len(self.train_tokenized) // (
            self.config['training']['per_device_train_batch_size'] * 
            self.config['training']['gradient_accumulation_steps']
        )
        eval_steps = max(1, steps_per_epoch // 10)
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Eval steps: {eval_steps}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=eval_steps,
            save_strategy=self.config['training']['save_strategy'],
            save_steps=eval_steps,
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            logging_steps=self.config['training']['logging_steps'],
            logging_dir=self.config['training']['logging_dir'],
            report_to=self.config['training']['report_to'],
            run_name=self.config['training']['run_name'],
            fp16=self.config['training']['fp16'],
            bf16=self.config['training']['bf16'],
            optim=self.config['training']['optim'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            dataloader_persistent_workers=self.config['training']['dataloader_persistent_workers'],
            group_by_length=self.config['training']['group_by_length'],
            seed=self.config['training']['seed'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            push_to_hub=self.config['training']['push_to_hub'],
        )
        
        # Callbacks
        callbacks = []
        if self.config['early_stopping']['enabled']:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config['early_stopping']['patience'],
                early_stopping_threshold=self.config['early_stopping']['threshold'],
            )
            callbacks.append(early_stopping)
            print(f"Early stopping: patience={self.config['early_stopping']['patience']}")
        
        # Initialize trainer
        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tokenized,
            eval_dataset=self.val_tokenized,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=self.class_weights,
            callbacks=callbacks,
        )
        
        print("\nTrainer initialized!")
    
    def run_pre_evaluation(self):
        """Run pre-training evaluation (baseline)"""
        if not self.config['evaluation']['run_pre_training_eval']:
            print("Pre-training evaluation skipped")
            return
        
        print_section("Pre-Training Evaluation (Baseline)")
        
        pre_eval_samples = self.config['evaluation'].get('pre_training_eval_samples')
        if pre_eval_samples and pre_eval_samples < len(self.test_tokenized):
            print(f"Using subset: {pre_eval_samples:,} samples")
            import random
            random.seed(42)
            indices = random.sample(range(len(self.test_tokenized)), pre_eval_samples)
            pre_eval_dataset = self.test_tokenized.select(indices)
        else:
            pre_eval_dataset = self.test_tokenized
        
        # Run prediction
        pre_results = self.trainer.predict(pre_eval_dataset)
        pre_predictions = np.argmax(pre_results.predictions, axis=1)
        pre_labels = pre_results.label_ids
        
        # Compute metrics
        self.pre_metrics = {
            'accuracy': accuracy_score(pre_labels, pre_predictions),
            'balanced_accuracy': balanced_accuracy_score(pre_labels, pre_predictions),
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            pre_labels, pre_predictions, average='weighted', zero_division=0
        )
        self.pre_metrics['precision'] = precision
        self.pre_metrics['recall'] = recall
        self.pre_metrics['f1'] = f1
        
        # Print metrics
        print(f"\nPre-training Metrics:")
        for metric, value in self.pre_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(pre_labels, pre_predictions))
        
        # Save metrics
        output_dir = self.config['training']['output_dir']
        save_metrics_to_file(self.pre_metrics, os.path.join(output_dir, 'pre_training_metrics.txt'))
        
        # Log to WandB
        if self.config['wandb']['enabled']:
            wandb.log({
                "pre_eval_accuracy": self.pre_metrics['accuracy'],
                "pre_eval_balanced_accuracy": self.pre_metrics['balanced_accuracy'],
                "pre_eval_f1": self.pre_metrics['f1'],
            })
    
    def train(self):
        """Execute training"""
        print_section("Starting Training")
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Training samples: {len(self.train_tokenized):,}")
        print(f"Epochs: {self.config['training']['num_train_epochs']}")
        print(f"Effective batch size: {self.config['training']['per_device_train_batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        print("\nThis may take several hours...\n")
        
        # Train
        train_result = self.trainer.train(
            resume_from_checkpoint=self.config['misc']['resume_from_checkpoint']
        )
        
        print_section("Training Completed")
        
        print("Training Metrics:")
        for key, value in train_result.metrics.items():
            print(f"  {key}: {value}")
        
        print_gpu_memory()
    
    def run_post_evaluation(self):
        """Run post-training evaluation"""
        if not self.config['evaluation']['run_post_training_eval']:
            print("Post-training evaluation skipped")
            return
        
        print_section("Post-Training Evaluation")
        
        # Run prediction
        post_results = self.trainer.predict(self.test_tokenized)
        post_predictions = np.argmax(post_results.predictions, axis=1)
        post_labels = post_results.label_ids
        
        # Compute metrics
        self.post_metrics = {
            'accuracy': accuracy_score(post_labels, post_predictions),
            'balanced_accuracy': balanced_accuracy_score(post_labels, post_predictions),
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            post_labels, post_predictions, average='weighted', zero_division=0
        )
        self.post_metrics['precision'] = precision
        self.post_metrics['recall'] = recall
        self.post_metrics['f1'] = f1
        
        # Print metrics
        print(f"\nPost-training Metrics:")
        for metric, value in self.post_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(post_labels, post_predictions)
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(
            post_labels, post_predictions,
            target_names=[self.id2label[i] for i in range(len(self.labels))],
            digits=4
        ))
        
        # Save metrics
        output_dir = self.config['training']['output_dir']
        save_metrics_to_file(self.post_metrics, os.path.join(output_dir, 'post_training_metrics.txt'))
        
        # Save predictions
        if self.config['evaluation']['save_predictions']:
            predictions_df = pd.DataFrame({
                'true_label': [self.id2label[label] for label in post_labels],
                'predicted_label': [self.id2label[pred] for pred in post_predictions],
                'correct': post_labels == post_predictions,
            })
            predictions_file = os.path.join(output_dir, self.config['evaluation']['predictions_file'])
            predictions_df.to_csv(predictions_file, index=False)
            print(f"\nPredictions saved: {predictions_file}")
        
        # Compare with pre-training
        if self.pre_metrics:
            print("\n" + "=" * 80)
            print("IMPROVEMENT SUMMARY")
            print("=" * 80)
            for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
                if metric in self.pre_metrics and metric in self.post_metrics:
                    improvement = self.post_metrics[metric] - self.pre_metrics[metric]
                    print(f"{metric:20s}: {self.pre_metrics[metric]:.4f} -> {self.post_metrics[metric]:.4f} (Delta {improvement:+.4f})")
        
        # Log to WandB
        if self.config['wandb']['enabled']:
            wandb.log({
                "post_eval_accuracy": self.post_metrics['accuracy'],
                "post_eval_balanced_accuracy": self.post_metrics['balanced_accuracy'],
                "post_eval_f1": self.post_metrics['f1'],
            })
            
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=post_labels,
                preds=post_predictions,
                class_names=[self.id2label[i] for i in range(len(self.labels))]
            )})
    
    def save_artifacts(self):
        """Save model, tokenizer, and configuration"""
        print_section("Saving Artifacts")
        
        output_dir = self.config['training']['output_dir']
        
        print(f"Saving to: {output_dir}")
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_file = os.path.join(output_dir, 'training_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"\nModel saved!")
        print(f"Output directory: {output_dir}")
    
    def print_summary(self):
        """Print training summary"""
        print_section("Training Summary")
        
        print(f"Dataset: {self.dataset_name} ({self.dataset_key})")
        print(f"Model: {self.config['model']['name']}")
        print(f"Training samples: {len(self.train_tokenized):,}")
        print(f"Validation samples: {len(self.val_tokenized):,}")
        print(f"Test samples: {len(self.test_tokenized):,}")
        
        print(f"\nConfiguration:")
        print(f"  Method: {'LoRA' if self.config['peft']['enabled'] else 'Full fine-tuning'}")
        if self.config['peft']['enabled']:
            print(f"  LoRA rank: {self.config['peft']['lora_r']}")
            print(f"  LoRA alpha: {self.config['peft']['lora_alpha']}")
        print(f"  Quantization: {'4-bit' if self.config['quantization']['load_in_4bit'] else 'None'}")
        print(f"  Epochs: {self.config['training']['num_train_epochs']}")
        print(f"  Effective batch: {self.config['training']['per_device_train_batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Class weights: {self.class_weights}")
        
        if self.post_metrics:
            print(f"\nFinal Test Metrics:")
            for metric, value in self.post_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nOutput: {self.config['training']['output_dir']}")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        if self.config['wandb']['enabled']:
            print(f"\nWandB Dashboard: {wandb.run.get_url()}")
    
    def cleanup(self):
        """Cleanup and finalize"""
        if self.config and self.config['wandb']['enabled']:
            print("\nFinishing WandB run...")
            wandb.finish()
            print("WandB run finished!")
    
    def run(self):
        """
        Execute the complete training pipeline.
        
        This method orchestrates all stages of the training process.
        """
        try:
            # Print system information
            print_section("System Information")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Execute pipeline stages
            self.load_config()
            self.initialize_wandb()
            self.load_datasets()
            self.create_label_mappings()
            self.setup_tokenizer()
            self.tokenize_datasets()
            self.setup_model()
            self.apply_lora()
            self.create_trainer()
            self.run_pre_evaluation()
            self.train()
            self.run_post_evaluation()
            self.save_artifacts()
            self.print_summary()
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Training failed with exception:")
            print(f"{'='*80}")
            print(f"{type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            self.cleanup()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Qwen3-0.6B Training Pipeline for AI/Human Text Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python train_qwen.py --dataset 10k
  
  With custom hyperparameters:
    python train_qwen.py --dataset 1M --epochs 5 --batch-size 16 --learning-rate 2e-4
  
  Disable WandB:
    python train_qwen.py --dataset 100k --no-wandb
  
  Resume from checkpoint:
    python train_qwen.py --dataset 10k --resume ./output/qwen3-0.6b-10k/checkpoint-1000
  
  Full customization:
    python train_qwen.py --dataset 2M --epochs 3 --batch-size 32 --learning-rate 1e-4 --lora-rank 32
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['10k', '100k', '1M', '2M'],
        help='Dataset size to use (required)'
    )
    
    # Optional training parameters
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size per device (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=None,
        help='LoRA rank (overrides config)'
    )
    
    # WandB control
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create and run pipeline
    pipeline = QwenTrainingPipeline(
        dataset=args.dataset,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        enable_wandb=not args.no_wandb,
        resume_from=args.resume
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()

