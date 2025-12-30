#!/usr/bin/env python3
"""
Qwen3-0.6B Training Script for AI/Human Text Detection

Generic training pipeline that works across multiple dataset sizes with automatic class weighting.

Usage:
    python train_qwen_standalone.py

Configuration:
    - Modify SELECTED_DATASET below to choose dataset size
    - Edit config.yaml for hyperparameter changes
    - WandB API key is pre-configured
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings
import hashlib
import json

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
# CONFIGURATION
# ============================================================================

# SELECT YOUR DATASET HERE
# Options: "10k", "100k", "1M", "2M"
SELECTED_DATASET = "10k"  # <<< CHANGE THIS TO SELECT YOUR DATASET

# WandB API Key
WANDB_API_KEY = "939a604fbdaf6ec2e540b7d12232eb267ec5bab2"

# Dataset mapping with pre-computed class weights
DATASET_OPTIONS = {
    "10k": {
        "name": "codefactory4791/raid_aligned_10k",
        "size": "~10K samples",
        "balance": "Balanced (50/50)",
        "class_weights": [1.0, 1.0]  # Equal weights for balanced dataset
    },
    "100k": {
        "name": "codefactory4791/raid_aligned_100k",
        "size": "~100K samples",
        "balance": "Balanced (50/50)",
        "class_weights": [1.0, 1.0]  # Equal weights for balanced dataset
    },
    "1M": {
        "name": "codefactory4791/raid_aligned_1000k",
        "size": "~1M samples",
        "balance": "Slightly Imbalanced (AI: 52.17%, Human: 47.83%)",
        "class_weights": [0.9167, 1.0909]  # Inverse frequency weights
    },
    "2M": {
        "name": "codefactory4791/raid_aligned_2000k",
        "size": "~2M samples",
        "balance": "Imbalanced (AI: 54.57%, Human: 45.43%)",
        "class_weights": [0.8324, 1.2009]  # Inverse frequency weights
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
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
    # Count labels
    label_counts = {}
    for sample in dataset:
        label = sample[label_column]
        label_id = label2id[label]
        label_counts[label_id] = label_counts.get(label_id, 0) + 1
    
    # Calculate weights (inverse frequency)
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


def get_cache_key(config):
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
        
        # Compute weighted loss
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.class_weights
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print_section("Qwen3-0.6B Training Pipeline - AI/Human Text Detection")
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Validate dataset selection
    if SELECTED_DATASET not in DATASET_OPTIONS:
        raise ValueError(f"Invalid dataset selection. Choose from: {list(DATASET_OPTIONS.keys())}")
    
    dataset_info = DATASET_OPTIONS[SELECTED_DATASET]
    DATASET_NAME = dataset_info["name"]
    CLASS_WEIGHTS = dataset_info["class_weights"]
    
    print(f"\nSelected Dataset: {SELECTED_DATASET}")
    print(f"HuggingFace Dataset: {DATASET_NAME}")
    print(f"Size: {dataset_info['size']}")
    print(f"Label Balance: {dataset_info['balance']}")
    print(f"Class Weights (AI_Generated, Human_Written): {CLASS_WEIGHTS}")
    
    # Load configuration
    print_section("Loading Configuration")
    
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with selected dataset
    config['dataset']['dataset_name'] = DATASET_NAME
    config['class_weights']['manual_weights'] = CLASS_WEIGHTS
    
    # Update output directory based on dataset
    config['training']['output_dir'] = f"./output/qwen3-0.6b-{SELECTED_DATASET}"
    config['training']['logging_dir'] = f"./logs/qwen3-0.6b-{SELECTED_DATASET}"
    config['misc']['tokenized_cache_dir'] = f"./tokenized_cache/{SELECTED_DATASET}"
    config['misc']['sampled_cache_dir'] = f"./sampled_cache/{SELECTED_DATASET}"
    
    # Update run name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    config['training']['run_name'] = f"qwen3-0.6b-{SELECTED_DATASET}-{timestamp}"
    
    # Update wandb tags
    config['wandb']['tags'].append(f"dataset-{SELECTED_DATASET}")
    
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['dataset_name']}")
    print(f"Output directory: {config['training']['output_dir']}")
    print(f"Run name: {config['training']['run_name']}")
    
    # Initialize WandB
    print_section("Initializing Weights & Biases")
    
    if config['wandb']['enabled']:
        wandb.login(key=WANDB_API_KEY)
        
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['training']['run_name'],
            tags=config['wandb']['tags'],
            notes=config['wandb']['notes'],
            config={
                'dataset': DATASET_NAME,
                'dataset_size': SELECTED_DATASET,
                'model': config['model']['name'],
                'batch_size': config['training']['per_device_train_batch_size'],
                'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
                'effective_batch_size': config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'],
                'learning_rate': config['training']['learning_rate'],
                'num_epochs': config['training']['num_train_epochs'],
                'lora_r': config['peft']['lora_r'],
                'lora_alpha': config['peft']['lora_alpha'],
                'max_length': config['tokenization']['max_length'],
                'class_weights': CLASS_WEIGHTS,
            }
        )
        
        print("WandB initialized!")
        print(f"Dashboard: {wandb.run.get_url()}")
    
    # Load dataset
    print_section("Loading Dataset")
    
    print(f"Loading dataset: {DATASET_NAME}...")
    print("This may take a few minutes depending on dataset size.\n")
    
    dataset = load_dataset(DATASET_NAME)
    
    print("Dataset loaded successfully!")
    print(f"Dataset structure: {dataset}")
    
    # Extract splits
    train_dataset = dataset[config['dataset']['train_split']]
    val_dataset = dataset[config['dataset']['validation_split']]
    test_dataset = dataset[config['dataset']['test_split']]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    # Check label distribution
    label_column = config['model']['label_column']
    text_column = config['model']['text_column']
    
    print(f"\nLabel distribution in training set:")
    train_labels = pd.Series([sample[label_column] for sample in train_dataset])
    print(train_labels.value_counts())
    print(f"\nLabel distribution (normalized):")
    print(train_labels.value_counts(normalize=True))
    
    # Create label mappings
    print_section("Creating Label Mappings")
    
    labels = config['model']['labels']
    label2id, id2label = create_label_mappings(labels)
    
    print(f"Label to ID mapping: {label2id}")
    print(f"ID to Label mapping: {id2label}")
    
    # Get class weights
    if config['class_weights']['enabled']:
        if config['class_weights']['manual_weights']:
            class_weights = torch.tensor(config['class_weights']['manual_weights'], dtype=torch.float32)
            print(f"\nUsing pre-computed class weights: {class_weights.tolist()}")
        else:
            print(f"\nComputing class weights from training data...")
            class_weights = compute_class_weights(train_dataset, label_column, label2id)
            print(f"Computed class weights: {class_weights.tolist()}")
    else:
        class_weights = None
        print("\nClass weights disabled.")
    
    if class_weights is not None:
        print(f"\nClass weight interpretation:")
        for i, label in enumerate(labels):
            print(f"  {label}: {class_weights[i]:.4f}")
    
    # Load tokenizer
    print_section("Loading Tokenizer")
    
    print(f"Loading tokenizer: {config['model']['name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        add_prefix_space=config['tokenization']['add_prefix_space'],
        use_fast=config['misc']['use_fast_tokenizer'],
        trust_remote_code=config['misc']['trust_remote_code'],
        cache_dir=config['misc']['cache_dir'],
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    print(f"\nTokenizer loaded successfully!")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Tokenize datasets
    print_section("Tokenizing Datasets")
    
    def preprocess_function(examples):
        """Tokenize text and map labels to IDs"""
        # Convert text to strings and handle None/NaN values
        texts = []
        for text in examples[text_column]:
            if text is None or (isinstance(text, float) and pd.isna(text)):
                texts.append("")
            else:
                texts.append(str(text))
        
        # Tokenize text
        tokenized = tokenizer(
            texts,
            padding=config['tokenization']['padding'],
            truncation=config['tokenization']['truncation'],
            max_length=config['tokenization']['max_length'],
        )
        
        # Map labels to IDs
        tokenized['labels'] = [label2id[label] for label in examples[label_column]]
        
        return tokenized
    
    # Check for cached tokenized datasets
    cache_dir = config['misc']['tokenized_cache_dir']
    use_cache = config['misc']['save_tokenized_datasets']
    force_retokenize = config['misc']['force_retokenize']
    
    cache_key = get_cache_key(config)
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
        print(f"Cache directory: {cache_dir}")
        print(f"Cache key: {cache_key}\n")
        
        train_tokenized = load_from_disk(cache_path_train)
        val_tokenized = load_from_disk(cache_path_val)
        test_tokenized = load_from_disk(cache_path_test)
        
        print(f"Loaded cached tokenized datasets!")
        print(f"  Train: {len(train_tokenized):,} samples")
        print(f"  Validation: {len(val_tokenized):,} samples")
        print(f"  Test: {len(test_tokenized):,} samples")
    else:
        print("Tokenizing datasets...")
        print("This may take a while for large datasets...\n")
        
        # Filter out invalid entries
        def is_valid(example):
            """Check if example has valid text and label"""
            text = example[text_column]
            label = example[label_column]
            
            text_valid = text is not None and not (isinstance(text, float) and pd.isna(text)) and str(text).strip() != ""
            label_valid = label is not None and label in label2id
            
            return text_valid and label_valid
        
        train_dataset = train_dataset.filter(is_valid, desc="Filtering train dataset")
        val_dataset = val_dataset.filter(is_valid, desc="Filtering validation dataset")
        test_dataset = test_dataset.filter(is_valid, desc="Filtering test dataset")
        
        print(f"\nAfter filtering:")
        print(f"  Train: {len(train_dataset):,} samples")
        print(f"  Validation: {len(val_dataset):,} samples")
        print(f"  Test: {len(test_dataset):,} samples")
        
        # Tokenize datasets
        print("\nTokenizing datasets...")
        
        train_tokenized = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        
        val_tokenized = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset",
        )
        
        test_tokenized = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset",
        )
        
        print("\nTokenization complete!")
        print(f"  Train: {len(train_tokenized):,} samples")
        print(f"  Validation: {len(val_tokenized):,} samples")
        print(f"  Test: {len(test_tokenized):,} samples")
        
        # Save tokenized datasets
        if use_cache:
            print(f"\nSaving tokenized datasets to cache...")
            os.makedirs(cache_dir, exist_ok=True)
            
            train_tokenized.save_to_disk(cache_path_train)
            val_tokenized.save_to_disk(cache_path_val)
            test_tokenized.save_to_disk(cache_path_test)
            
            print(f"Tokenized datasets saved to: {cache_dir}")
    
    # Load model
    print_section("Loading Model")
    
    # Configure quantization
    quantization_config = None
    if config['quantization']['enabled']:
        print("Setting up quantization...")
        
        compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization']['load_in_4bit'],
            load_in_8bit=config['quantization']['load_in_8bit'],
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        
        print(f"Quantization: 4-bit" if config['quantization']['load_in_4bit'] else "8-bit")
    
    print(f"\nLoading model: {config['model']['name']}...")
    print("This may take a few minutes...\n")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        id2label=id2label,
        label2id=label2id,
        quantization_config=quantization_config,
        device_map=config['hardware']['device_map'],
        max_memory=config['hardware']['max_memory'],
        trust_remote_code=config['misc']['trust_remote_code'],
        cache_dir=config['misc']['cache_dir'],
    )
    
    # Configure model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    print(f"Model loaded successfully!")
    print(f"Number of parameters: {model.num_parameters():,}")
    print_gpu_memory()
    
    # Apply PEFT/LoRA
    print_section("Applying PEFT/LoRA")
    
    if config['peft']['enabled']:
        print("Applying PEFT/LoRA...")
        
        # Prepare model for k-bit training
        if config['quantization']['enabled']:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=config['peft']['lora_r'],
            lora_alpha=config['peft']['lora_alpha'],
            target_modules=config['peft']['target_modules'],
            lora_dropout=config['peft']['lora_dropout'],
            bias=config['peft']['bias'],
            task_type=TaskType.SEQ_CLS,
            modules_to_save=config['peft']['modules_to_save'],
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        print(f"\nLoRA applied successfully!")
        print(f"LoRA rank: {config['peft']['lora_r']}")
        print(f"LoRA alpha: {config['peft']['lora_alpha']}")
        
        # Print trainable parameters
        model.print_trainable_parameters()
    else:
        print("PEFT disabled. Using full fine-tuning.")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    print_gpu_memory()
    
    # Data collator
    print_section("Initializing Data Collator")
    
    if config['data_collator']['type'] == 'DataCollatorWithPadding':
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=config['data_collator']['padding'],
            pad_to_multiple_of=config['data_collator']['pad_to_multiple_of'],
        )
        print(f"Using DataCollatorWithPadding")
    else:
        data_collator = None
        print(f"Using default data collator")
    
    # Metrics function
    def compute_metrics(eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
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
    print_section("Configuring Training Arguments")
    
    # Create output directory
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate eval and save steps
    steps_per_epoch = len(train_tokenized) // (config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'])
    eval_steps = max(1, steps_per_epoch // 10)  # Evaluate 10 times per epoch
    
    config['training']['eval_steps'] = eval_steps
    config['training']['save_steps'] = eval_steps
    
    print(f"Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Eval steps: {eval_steps}")
    print(f"Total optimization steps: ~{steps_per_epoch * config['training']['num_train_epochs']}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        
        # Evaluation and saving
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=eval_steps,
        save_strategy=config['training']['save_strategy'],
        save_steps=eval_steps,
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        
        # Logging
        logging_steps=config['training']['logging_steps'],
        logging_dir=config['training']['logging_dir'],
        report_to=config['training']['report_to'],
        run_name=config['training']['run_name'],
        
        # Mixed precision
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        
        # Optimization
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        
        # Data loading
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        dataloader_persistent_workers=config['training']['dataloader_persistent_workers'],
        group_by_length=config['training']['group_by_length'],
        
        # Misc
        seed=config['training']['seed'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        push_to_hub=config['training']['push_to_hub'],
        hub_model_id=config['training']['hub_model_id'],
        hub_token=config['training']['hub_token'],
    )
    
    # Initialize trainer
    print_section("Initializing Trainer")
    
    # Early stopping callback
    callbacks = []
    if config['early_stopping']['enabled']:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_threshold=config['early_stopping']['threshold'],
        )
        callbacks.append(early_stopping)
        print(f"Early stopping enabled with patience: {config['early_stopping']['patience']}")
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )
    
    print("\nTrainer initialized successfully!")
    
    # Pre-training evaluation
    pre_metrics = None
    if config['evaluation']['run_pre_training_eval'] and test_tokenized:
        print_section("Pre-Training Evaluation (Baseline)")
        
        pre_eval_samples = config['evaluation'].get('pre_training_eval_samples')
        if pre_eval_samples and pre_eval_samples < len(test_tokenized):
            print(f"Using subset of {pre_eval_samples:,} samples for fast baseline")
            import random
            random.seed(42)
            indices = random.sample(range(len(test_tokenized)), pre_eval_samples)
            pre_eval_dataset = test_tokenized.select(indices)
        else:
            pre_eval_dataset = test_tokenized
        
        # Run prediction
        pre_results = trainer.predict(pre_eval_dataset)
        
        # Extract predictions and labels
        pre_predictions = np.argmax(pre_results.predictions, axis=1)
        pre_labels = pre_results.label_ids
        
        # Compute metrics
        pre_metrics = {
            'accuracy': accuracy_score(pre_labels, pre_predictions),
            'balanced_accuracy': balanced_accuracy_score(pre_labels, pre_predictions),
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            pre_labels, pre_predictions, average='weighted', zero_division=0
        )
        pre_metrics['precision'] = precision
        pre_metrics['recall'] = recall
        pre_metrics['f1'] = f1
        
        # Print metrics
        print(f"\nPre-training Metrics:")
        for metric, value in pre_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(pre_labels, pre_predictions)
        print(cm)
        
        # Save metrics
        save_metrics_to_file(pre_metrics, os.path.join(output_dir, 'pre_training_metrics.txt'))
        
        # Log to wandb
        if config['wandb']['enabled']:
            wandb.log({
                "pre_eval_accuracy": pre_metrics['accuracy'],
                "pre_eval_balanced_accuracy": pre_metrics['balanced_accuracy'],
                "pre_eval_f1": pre_metrics['f1'],
            })
    
    # Training
    print_section("Starting Training")
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"Training samples: {len(train_tokenized):,}")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"\nThis may take several hours...\n")
    
    # Train
    resume_from_checkpoint = config['misc']['resume_from_checkpoint']
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print_section("Training Completed")
    
    # Print training metrics
    print(f"Training Metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")
    
    print_gpu_memory()
    
    # Post-training evaluation
    if config['evaluation']['run_post_training_eval'] and test_tokenized:
        print_section("Post-Training Evaluation")
        
        # Run prediction on test set
        post_results = trainer.predict(test_tokenized)
        
        # Extract predictions and labels
        post_predictions = np.argmax(post_results.predictions, axis=1)
        post_labels = post_results.label_ids
        
        # Compute metrics
        post_metrics = {
            'accuracy': accuracy_score(post_labels, post_predictions),
            'balanced_accuracy': balanced_accuracy_score(post_labels, post_predictions),
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            post_labels, post_predictions, average='weighted', zero_division=0
        )
        post_metrics['precision'] = precision
        post_metrics['recall'] = recall
        post_metrics['f1'] = f1
        
        # Print metrics
        print(f"\nPost-training Metrics:")
        for metric, value in post_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(post_labels, post_predictions)
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(
            post_labels, post_predictions,
            target_names=[id2label[i] for i in range(len(labels))],
            digits=4
        ))
        
        # Save metrics
        save_metrics_to_file(post_metrics, os.path.join(output_dir, 'post_training_metrics.txt'))
        
        # Save predictions
        if config['evaluation']['save_predictions']:
            predictions_df = pd.DataFrame({
                'true_label': [id2label[label] for label in post_labels],
                'predicted_label': [id2label[pred] for pred in post_predictions],
                'correct': post_labels == post_predictions,
            })
            predictions_file = os.path.join(output_dir, config['evaluation']['predictions_file'])
            predictions_df.to_csv(predictions_file, index=False)
            print(f"\nPredictions saved to: {predictions_file}")
        
        # Compare with pre-training metrics
        if pre_metrics:
            print("\n" + "=" * 80)
            print("IMPROVEMENT SUMMARY")
            print("=" * 80)
            for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
                if metric in pre_metrics and metric in post_metrics:
                    improvement = post_metrics[metric] - pre_metrics[metric]
                    print(f"{metric:20s}: {pre_metrics[metric]:.4f} -> {post_metrics[metric]:.4f} (Delta {improvement:+.4f})")
        
        # Log to wandb
        if config['wandb']['enabled']:
            wandb.log({
                "post_eval_accuracy": post_metrics['accuracy'],
                "post_eval_balanced_accuracy": post_metrics['balanced_accuracy'],
                "post_eval_f1": post_metrics['f1'],
            })
            
            # Log confusion matrix
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=post_labels,
                preds=post_predictions,
                class_names=[id2label[i] for i in range(len(labels))]
            )})
    
    # Save model
    print_section("Saving Model")
    
    print(f"Saving model to: {output_dir}")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    training_config_file = os.path.join(output_dir, 'training_config.yaml')
    with open(training_config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nModel saved successfully!")
    print(f"Output directory: {output_dir}")
    
    # Training summary
    print_section("Training Summary")
    
    print(f"Dataset: {DATASET_NAME} ({SELECTED_DATASET})")
    print(f"Model: {config['model']['name']}")
    print(f"Training samples: {len(train_tokenized):,}")
    print(f"Validation samples: {len(val_tokenized):,}")
    print(f"Test samples: {len(test_tokenized):,}")
    
    print(f"\nTraining Configuration:")
    print(f"  Method: {'PEFT/LoRA' if config['peft']['enabled'] else 'Full fine-tuning'}")
    if config['peft']['enabled']:
        print(f"  LoRA rank: {config['peft']['lora_r']}")
        print(f"  LoRA alpha: {config['peft']['lora_alpha']}")
    print(f"  Quantization: {'4-bit' if config['quantization']['load_in_4bit'] else '8-bit' if config['quantization']['load_in_8bit'] else 'None'}")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Class weights: {CLASS_WEIGHTS}")
    
    if 'post_metrics' in locals() and post_metrics:
        print(f"\nFinal Test Metrics:")
        for metric, value in post_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nOutput directory: {output_dir}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    if config['wandb']['enabled']:
        print(f"\nWandB Dashboard: {wandb.run.get_url()}")
    
    print(f"\nNext steps:")
    print(f"  1. Review training logs and metrics")
    print(f"  2. Test the model on new data")
    print(f"  3. Deploy the model for inference")
    
    # Finish wandb run
    if config['wandb']['enabled']:
        print("\nFinishing WandB run...")
        wandb.finish()
        print("WandB run finished successfully!")


if __name__ == "__main__":
    main()

