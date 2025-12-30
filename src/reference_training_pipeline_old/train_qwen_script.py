#!/usr/bin/env python3
"""
Qwen Fine-tuning Script for AI/Human Text Detection

Professional, modular training script for fine-tuning Qwen models on AI/Human text classification.

Model: Configurable (Qwen3-0.6B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-8B-Base, etc.)
Dataset: https://huggingface.co/datasets/codefactory4791/ai-human-text-detection-balanced_sampled_200k

Usage:
    python train_qwen_script.py --config config.yaml [--skip_pre_eval] [--wandb_token TOKEN]
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
from torch.utils.data import WeightedRandomSampler
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

# Weights & Biases
import wandb

# HuggingFace Hub
from huggingface_hub import login as hf_login


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen models for AI/Human text detection"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--wandb_token", type=str, default=None,
        help="Weights & Biases API token"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--skip_pre_eval", action="store_true",
        help="Skip pre-training evaluation"
    )
    return parser.parse_args()


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_environment():
    """Configure environment variables and print system info."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 80)


def load_config(config_path: str, skip_pre_eval: bool = False) -> Dict:
    """Load and validate configuration from YAML file."""
    print(f"\nLoading configuration: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if skip_pre_eval:
        config['evaluation']['run_pre_training_eval'] = False
    
    print("Configuration loaded:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  PEFT enabled: {config['peft']['enabled']}")
    print(f"  Output dir: {config['training']['output_dir']}")
    print(f"  Pre-training eval: {config['evaluation']['run_pre_training_eval']}")
    
    return config


def setup_wandb(config: Dict, wandb_token: Optional[str] = None):
    """Initialize Weights & Biases."""
    if not config['wandb']['enabled']:
        print("\nWeights & Biases disabled")
        return
    
    print("\n" + "=" * 80)
    print("WEIGHTS & BIASES SETUP")
    print("=" * 80)
    
    if wandb_token:
        wandb.login(key=wandb_token)
    else:
        print("\nPlease enter your W&B API token (https://wandb.ai/authorize):")
        wandb.login()
    
    run_name = config['training'].get('run_name') or f"{config['model']['name'].split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=run_name,
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config={
            'model': config['model']['name'],
            'batch_size': config['training']['per_device_train_batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_train_epochs'],
            'lora_r': config['peft']['lora_r'] if config['peft']['enabled'] else None,
            'max_length': config['tokenization']['max_length'],
        }
    )
    
    print(f"W&B initialized: {wandb.run.get_url()}")
    print("=" * 80)


def setup_huggingface(config: Dict, hf_token: Optional[str] = None):
    """Login to HuggingFace Hub if needed."""
    if not config['training']['push_to_hub']:
        return
    
    print("\n" + "=" * 80)
    print("HUGGINGFACE HUB SETUP")
    print("=" * 80)
    
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("\nPlease enter your HF API token (https://huggingface.co/settings/tokens):")
        hf_login()
    
    print(f"Will push to: {config['training']['hub_model_id']}")
    print("=" * 80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label to ID and ID to label mappings."""
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_class_weights(dataset: Dataset, label_column: str, label2id: Dict[str, int]) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency."""
    labels_df = pd.DataFrame(dataset)[label_column]
    labels_df = labels_df.map(label2id)
    
    value_counts = labels_df.value_counts(normalize=True).sort_index()
    weights = 1.0 / value_counts
    weights = torch.tensor(weights.values, dtype=torch.float32)
    return weights / weights.sum()


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB allocated, "
              f"{torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")


def save_metrics_to_file(metrics: Dict, filepath: str):
    """Save metrics to a file."""
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def get_cache_key(key_data: Dict) -> str:
    """Generate MD5 cache key from dictionary."""
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


# ============================================================================
# DATASET FUNCTIONS
# ============================================================================

def load_datasets(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load datasets from local path or HuggingFace Hub."""
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    # Check if using local pre-sampled dataset
    if config['dataset'].get('use_local_dataset', False):
        dataset_path = config['dataset']['dataset_path']
        print(f"Source: Local pre-sampled dataset")
        print(f"Path: {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        print(f"✓ Dataset loaded successfully")
    else:
        # Load from HuggingFace Hub
        print(f"Source: HuggingFace Hub")
        print(f"Dataset: {config['dataset']['name']}")
        
        # Load non-streaming dataset (pre-sampled dataset is already split)
        dataset = load_dataset(config['dataset']['name'], streaming=False)
        print(f"✓ Dataset loaded successfully")
    
    print(f"\nDataset structure: {dataset}")
    
    # Get splits
    train_split = config['dataset']['train_split']
    val_split = config['dataset']['validation_split']
    test_split = config['dataset']['test_split']
    
    train_dataset = dataset[train_split]
    val_dataset = dataset[val_split] if val_split in dataset else None
    test_dataset = dataset[test_split] if test_split in dataset else None
    
    print(f"\nLoaded splits:")
    print(f"  Train: {len(train_dataset):,} samples")
    if val_dataset:
        print(f"  Validation: {len(val_dataset):,} samples")
    if test_dataset:
        print(f"  Test: {len(test_dataset):,} samples")
    
    # Check if sample_weight column exists in train
    if 'sample_weight' in train_dataset.column_names:
        print(f"  ✓ sample_weight column found in training data")
    
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# NOTE: Sampling functions removed - dataset is pre-sampled in notebook
# See: src/experiments/balanced_dataset_sampling.ipynb
# ============================================================================


def check_data_leakage(train_dataset: Dataset, val_dataset: Dataset, 
                       test_dataset: Dataset, text_column: str = 'text',
                       sample_size: int = 10000) -> Dict:
    """
    Check for data leakage between train/val/test splits.
    
    Samples a subset for efficiency, checks for exact text matches.
    """
    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)
    
    # Sample for efficiency (checking millions of texts is slow)
    import random
    random.seed(42)
    
    def get_text_sample(dataset, size):
        if len(dataset) <= size:
            return set([str(item[text_column]) for item in dataset if text_column in item])
        indices = random.sample(range(len(dataset)), min(size, len(dataset)))
        return set([str(dataset[i][text_column]) for i in indices if text_column in dataset.column_names])
    
    print(f"Sampling {sample_size:,} texts from each split for leakage check...")
    train_texts = get_text_sample(train_dataset, sample_size)
    val_texts = get_text_sample(val_dataset, min(sample_size, len(val_dataset)))
    test_texts = get_text_sample(test_dataset, min(sample_size, len(test_dataset)))
    
    print(f"  Train sample: {len(train_texts):,} unique texts")
    print(f"  Val sample: {len(val_texts):,} unique texts")
    print(f"  Test sample: {len(test_texts):,} unique texts")
    
    # Check overlaps
    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts
    
    results = {
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'total_overlap': len(train_val_overlap | train_test_overlap | val_test_overlap)
    }
    
    print(f"\nOverlap Detection Results:")
    print(f"  Train ∩ Val:  {results['train_val_overlap']:,} duplicates")
    print(f"  Train ∩ Test: {results['train_test_overlap']:,} duplicates")
    print(f"  Val ∩ Test:   {results['val_test_overlap']:,} duplicates")
    
    if results['total_overlap'] > 0:
        print(f"\n⚠️  WARNING: Found {results['total_overlap']:,} overlapping texts!")
        print(f"   This indicates potential data leakage.")
        print(f"   Percentage of sample: {results['total_overlap']/sample_size*100:.2f}%")
    else:
        print(f"\n✅ No data leakage detected in sample!")
    
    print("=" * 80)
    
    return results


# Removed - dataset is pre-sampled in balanced_dataset_sampling.ipynb


def create_label_mappings_and_weights(config: Dict, train_dataset: Dataset) -> Tuple[Dict, Dict, Optional[torch.Tensor]]:
    """Create label mappings and compute class weights."""
    print("\n" + "=" * 80)
    print("LABEL MAPPINGS & CLASS WEIGHTS")
    print("=" * 80)
    
    labels = config['model']['labels']
    label2id, id2label = create_label_mappings(labels)
    
    print(f"Label mappings: {label2id}")
    
    # Compute class weights
    if config['class_weights']['enabled']:
        if config['class_weights']['method'] == 'inverse_frequency':
            class_weights = compute_class_weights(
                train_dataset, 
                config['model']['label_column'], 
                label2id
            )
            print(f"Class weights: {class_weights}")
        elif config['class_weights']['manual_weights']:
            class_weights = torch.tensor(config['class_weights']['manual_weights'], dtype=torch.float32)
            print(f"Manual class weights: {class_weights}")
    else:
        class_weights = None
        print("Class weights disabled")
    
    print("=" * 80)
    
    return label2id, id2label, class_weights


# ============================================================================
# TOKENIZATION FUNCTIONS
# ============================================================================

def load_tokenizer(config: Dict):
    """Load and configure tokenizer."""
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)
    print(f"Tokenizer: {config['model']['name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        add_prefix_space=config['tokenization']['add_prefix_space'],
        use_fast=config['misc']['use_fast_tokenizer'],
        trust_remote_code=config['misc']['trust_remote_code'],
        cache_dir=config['misc']['cache_dir'],
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print("=" * 80)
    
    return tokenizer


def tokenize_datasets(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset,
                     tokenizer, config: Dict, label2id: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Tokenize datasets with caching support."""
    
    text_column = config['model']['text_column']
    label_column = config['model']['label_column']
    
    def preprocess_function(examples):
        """Tokenize text and map labels."""
        texts = []
        for text in examples[text_column]:
            if text is None or (isinstance(text, float) and pd.isna(text)):
                texts.append("")
            else:
                texts.append(str(text))
        
        tokenized = tokenizer(
            texts,
            padding=config['tokenization']['padding'],
            truncation=config['tokenization']['truncation'],
            max_length=config['tokenization']['max_length'],
        )
        
        tokenized['labels'] = [label2id[label] for label in examples[label_column]]
        return tokenized
    
    # Check tokenized cache
    cache_dir = config['misc'].get('tokenized_cache_dir', './tokenized_cache')
    use_cache = config['misc'].get('save_tokenized_datasets', True)
    force_retokenize = config['misc'].get('force_retokenize', False)
    
    cache_key_data = {
        'model': config['model']['name'],
        'max_length': config['tokenization']['max_length'],
        'text_column': text_column,
        'label_column': label_column,
    }
    cache_key = get_cache_key(cache_key_data)
    
    cache_path_train = os.path.join(cache_dir, f'train_{cache_key}')
    cache_path_val = os.path.join(cache_dir, f'val_{cache_key}')
    cache_path_test = os.path.join(cache_dir, f'test_{cache_key}')
    
    cache_exists = all([
        os.path.exists(cache_path_train),
        os.path.exists(cache_path_val),
        os.path.exists(cache_path_test)
    ])
    
    # Try loading from cache
    if use_cache and cache_exists and not force_retokenize:
        print("\n" + "=" * 80)
        print("LOADING TOKENIZED DATASETS FROM CACHE")
        print("=" * 80)
        print(f"Cache key: {cache_key}\n")
        
        train_tokenized = load_from_disk(cache_path_train)
        val_tokenized = load_from_disk(cache_path_val)
        test_tokenized = load_from_disk(cache_path_test)
        
        print(f"Loaded in seconds:")
        print(f"  Train: {len(train_tokenized):,} samples")
        print(f"  Validation: {len(val_tokenized):,} samples")
        print(f"  Test: {len(test_tokenized):,} samples")
        print("=" * 80)
        
        return train_tokenized, val_tokenized, test_tokenized
    
    # Perform tokenization
    print("\n" + "=" * 80)
    print("TOKENIZING DATASETS")
    print("=" * 80)
    
    if force_retokenize:
        print("Force re-tokenization enabled")
    else:
        print("No tokenized cache found")
    
    print("Filtering invalid entries...")
    
    def is_valid(example):
        text = example[text_column]
        label = example[label_column]
        text_valid = text is not None and not (isinstance(text, float) and pd.isna(text)) and str(text).strip() != ""
        label_valid = label is not None and label in label2id
        return text_valid and label_valid
    
    train_dataset = train_dataset.filter(is_valid, desc="Filtering train")
    val_dataset = val_dataset.filter(is_valid, desc="Filtering validation")
    test_dataset = test_dataset.filter(is_valid, desc="Filtering test")
    
    print(f"\nTokenizing datasets...")
    
    # Check if sample_weight column exists (needed for WeightedRandomSampler)
    sample_weight_column = config.get('sample_weighting', {}).get('weight_column', 'sample_weight')
    preserve_sample_weight = sample_weight_column in train_dataset.column_names
    
    # Columns to remove: all except sample_weight (if it exists)
    train_columns_to_remove = [col for col in train_dataset.column_names 
                               if col != sample_weight_column]
    val_columns_to_remove = [col for col in val_dataset.column_names 
                            if col != sample_weight_column]
    test_columns_to_remove = [col for col in test_dataset.column_names 
                             if col != sample_weight_column]
    
    if preserve_sample_weight:
        print(f"  ✓ Preserving '{sample_weight_column}' column for WeightedRandomSampler")
    
    train_tokenized = train_dataset.map(preprocess_function, batched=True, 
                                       remove_columns=train_columns_to_remove,
                                       desc="Tokenizing train")
    val_tokenized = val_dataset.map(preprocess_function, batched=True,
                                    remove_columns=val_columns_to_remove,
                                    desc="Tokenizing validation")
    test_tokenized = test_dataset.map(preprocess_function, batched=True,
                                     remove_columns=test_columns_to_remove,
                                     desc="Tokenizing test")
    
    print(f"\nTokenization complete:")
    print(f"  Train: {len(train_tokenized):,} samples")
    print(f"  Validation: {len(val_tokenized):,} samples")
    print(f"  Test: {len(test_tokenized):,} samples")
    
    # Save to cache
    if use_cache:
        print(f"\nSaving to cache...")
        os.makedirs(cache_dir, exist_ok=True)
        
        train_tokenized.save_to_disk(cache_path_train)
        val_tokenized.save_to_disk(cache_path_val)
        test_tokenized.save_to_disk(cache_path_test)
        
        print(f"Saved to: {cache_dir}")
    
    print("=" * 80)
    
    return train_tokenized, val_tokenized, test_tokenized


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def load_model(config: Dict, label2id: Dict, id2label: Dict):
    """Load and configure model with optional quantization."""
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    # Configure quantization
    quantization_config = None
    if config['quantization']['enabled']:
        compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization']['load_in_4bit'],
            load_in_8bit=config['quantization']['load_in_8bit'],
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=compute_dtype,
        )
        print(f"Quantization: {config['quantization']['bnb_4bit_quant_type']} 4-bit")
    
    print(f"Loading model: {config['model']['name']}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        id2label=id2label,
        label2id=label2id,
        quantization_config=quantization_config,
        device_map=config['hardware']['device_map'],
        trust_remote_code=config['misc']['trust_remote_code'],
        cache_dir=config['misc']['cache_dir'],
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    # Explicitly disable gradient checkpointing if not wanted
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("Gradient checkpointing explicitly disabled")
    
    print(f"Model loaded: {model.num_parameters():,} parameters")
    print_gpu_memory()
    print("=" * 80)
    
    return model


def apply_peft(model, config: Dict):
    """Apply PEFT/LoRA if enabled."""
    if not config['peft']['enabled']:
        print("\nPEFT disabled - using full fine-tuning")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return model
    
    print("\n" + "=" * 80)
    print("APPLYING PEFT/LORA")
    print("=" * 80)
    
    if config['quantization']['enabled']:
        model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=config['peft']['lora_r'],
        lora_alpha=config['peft']['lora_alpha'],
        target_modules=config['peft']['target_modules'],
        lora_dropout=config['peft']['lora_dropout'],
        bias=config['peft']['bias'],
        task_type=TaskType.SEQ_CLS,
        modules_to_save=config['peft']['modules_to_save'],
    )
    
    model = get_peft_model(model, peft_config)
    
    print(f"LoRA applied:")
    print(f"  Rank: {config['peft']['lora_r']}")
    print(f"  Alpha: {config['peft']['lora_alpha']}")
    model.print_trainable_parameters()
    print_gpu_memory()
    print("=" * 80)
    
    return model


# ============================================================================
# TRAINER FUNCTIONS
# ============================================================================

class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss and optional WeightedRandomSampler."""
    
    def __init__(self, *args, class_weights=None, use_weighted_sampler=False, 
                 sample_weight_column='sample_weight', **kwargs):
        super().__init__(*args, **kwargs)
        
        # Class weights for loss
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.detach().clone().float()
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = self.class_weights.to(self.args.device)
        else:
            self.class_weights = None
        
        # WeightedRandomSampler settings
        self.use_weighted_sampler = use_weighted_sampler
        self.sample_weight_column = sample_weight_column
    
    def _get_train_sampler(self, dataset=None):
        """Create WeightedRandomSampler if enabled.
        
        Args:
            dataset: Optional dataset argument (for compatibility with newer transformers versions).
                    If None, uses self.train_dataset.
        """
        # Use provided dataset or fall back to self.train_dataset
        train_dataset = dataset if dataset is not None else self.train_dataset
        
        if self.use_weighted_sampler and train_dataset is not None:
            # Check if sample_weight column exists
            if self.sample_weight_column in train_dataset.column_names:
                print(f"  ✓ Using WeightedRandomSampler with '{self.sample_weight_column}' column")
                
                # Extract sample weights
                sample_weights = torch.tensor(
                    train_dataset[self.sample_weight_column],
                    dtype=torch.float32
                )
                
                # Create weighted sampler
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True  # Allow replacement for balanced sampling
                )
                return sampler
            else:
                print(f"  ⚠️  sample_weight column '{self.sample_weight_column}' not found, using default sampler")
        
        # Fall back to default sampler (parent method doesn't accept dataset arg)
        return super()._get_train_sampler()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted cross-entropy loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
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


def setup_trainer(model, tokenizer, train_tokenized, val_tokenized, 
                 class_weights, config: Dict):
    """Initialize trainer with all configurations."""
    print("\n" + "=" * 80)
    print("SETTING UP TRAINER")
    print("=" * 80)
    
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Explicitly disable gradient checkpointing on model if config says false
    if not config['training']['gradient_checkpointing']:
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            print("Gradient checkpointing forcibly disabled on model")
    
    # Calculate eval_steps dynamically (10 times per epoch)
    eval_steps = config['training']['eval_steps']
    save_steps = config['training']['save_steps']
    
    if eval_steps is None and config['training']['eval_strategy'] == 'steps':
        # Steps per epoch = total_samples / effective_batch_size
        effective_batch_size = (config['training']['per_device_train_batch_size'] * 
                               config['training']['gradient_accumulation_steps'])
        steps_per_epoch = len(train_tokenized) // effective_batch_size
        eval_steps = max(steps_per_epoch // 10, 1)  # Evaluate 10 times per epoch
        save_steps = eval_steps  # Save at same frequency
        
        print(f"Auto-calculated evaluation steps:")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Eval frequency: 10 times per epoch")
        print(f"  Eval steps: {eval_steps:,}")
    
    # Training arguments
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
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=eval_steps,
        save_strategy=config['training']['save_strategy'],
        save_steps=save_steps,
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        logging_steps=config['training']['logging_steps'],
        logging_dir=config['training']['logging_dir'],
        report_to=config['training']['report_to'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        optim=config['training']['optim'],
        max_grad_norm=config['training']['max_grad_norm'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        dataloader_persistent_workers=config['training'].get('dataloader_persistent_workers', True) if config['training']['dataloader_num_workers'] > 0 else False,
        group_by_length=config['training']['group_by_length'],
        seed=config['training']['seed'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        push_to_hub=config['training']['push_to_hub'],
        hub_model_id=config['training']['hub_model_id'],
        hub_token=config['training']['hub_token'],
    )
    
    # Data collator
    if config['data_collator']['type'] == 'DataCollatorWithPadding':
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=config['data_collator']['padding'],
            pad_to_multiple_of=config['data_collator']['pad_to_multiple_of'],
        )
    else:
        data_collator = None
    
    # Early stopping
    callbacks = []
    if config['early_stopping']['enabled']:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_threshold=config['early_stopping']['threshold'],
        ))
        print(f"  ✓ Early stopping enabled (patience={config['early_stopping']['patience']}, threshold={config['early_stopping']['threshold']})")
    
    # WeightedRandomSampler settings
    use_weighted_sampler = config.get('sample_weighting', {}).get('enabled', False)
    sample_weight_column = config.get('sample_weighting', {}).get('weight_column', 'sample_weight')
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_weighted_sampler=use_weighted_sampler,
        sample_weight_column=sample_weight_column,
        callbacks=callbacks,
    )
    
    print(f"Trainer initialized")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    if use_weighted_sampler:
        print(f"  ✓ WeightedRandomSampler enabled (domain + label balanced)")
    print("=" * 80)
    
    return trainer


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def run_pre_training_evaluation(trainer, test_tokenized, config: Dict, 
                               id2label: Dict, labels: List[str], output_dir: str):
    """Run pre-training baseline evaluation."""
    if not config['evaluation']['run_pre_training_eval'] or not test_tokenized:
        print("\nPre-training evaluation skipped")
        return None
    
    print("\n" + "=" * 80)
    print("PRE-TRAINING EVALUATION (Baseline)")
    print("=" * 80)
    
    # Use subset for speed
    pre_eval_samples = config['evaluation'].get('pre_training_eval_samples')
    if pre_eval_samples and pre_eval_samples < len(test_tokenized):
        print(f"Using {pre_eval_samples:,} sample subset for baseline")
        import random
        random.seed(42)
        indices = random.sample(range(len(test_tokenized)), pre_eval_samples)
        pre_eval_dataset = test_tokenized.select(indices)
    else:
        pre_eval_dataset = test_tokenized
    
    pre_results = trainer.predict(pre_eval_dataset)
    pre_predictions = np.argmax(pre_results.predictions, axis=1)
    pre_labels = pre_results.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        pre_labels, pre_predictions, average='weighted', zero_division=0
    )
    
    pre_metrics = {
        'accuracy': accuracy_score(pre_labels, pre_predictions),
        'balanced_accuracy': balanced_accuracy_score(pre_labels, pre_predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    print(f"\nPre-training Metrics ({len(pre_eval_dataset):,} samples):")
    for metric, value in pre_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(pre_labels, pre_predictions))
    
    save_metrics_to_file(pre_metrics, os.path.join(output_dir, 'pre_training_metrics.txt'))
    
    if config['wandb']['enabled']:
        wandb.log({"pre_eval": pre_metrics})
    
    print("=" * 80)
    
    return pre_metrics


def run_post_training_evaluation(trainer, test_tokenized, config: Dict, 
                                id2label: Dict, labels: List[str], output_dir: str, pre_metrics: Optional[Dict]):
    """Run post-training evaluation."""
    if not config['evaluation']['run_post_training_eval'] or not test_tokenized:
        print("\nPost-training evaluation skipped")
        return None
    
    print("\n" + "=" * 80)
    print("POST-TRAINING EVALUATION")
    print("=" * 80)
    
    post_results = trainer.predict(test_tokenized)
    post_predictions = np.argmax(post_results.predictions, axis=1)
    post_labels = post_results.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        post_labels, post_predictions, average='weighted', zero_division=0
    )
    
    post_metrics = {
        'accuracy': accuracy_score(post_labels, post_predictions),
        'balanced_accuracy': balanced_accuracy_score(post_labels, post_predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    print(f"\nPost-training Metrics ({len(test_tokenized):,} samples):")
    for metric, value in post_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(post_labels, post_predictions))
    
    print(f"\nClassification Report:")
    print(classification_report(post_labels, post_predictions,
                               target_names=[id2label[i] for i in range(len(labels))],
                               digits=4))
    
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
        print(f"\nPredictions saved: {predictions_file}")
    
    if config['wandb']['enabled']:
        wandb.log({"post_eval": post_metrics})
    
    # Compare with baseline
    if pre_metrics:
        print("\n" + "=" * 80)
        print("IMPROVEMENT SUMMARY")
        print("=" * 80)
        for metric in ['accuracy', 'balanced_accuracy', 'f1']:
            if metric in pre_metrics and metric in post_metrics:
                improvement = post_metrics[metric] - pre_metrics[metric]
                print(f"{metric:20s}: {pre_metrics[metric]:.4f} -> {post_metrics[metric]:.4f} (delta {improvement:+.4f})")
    
    print("=" * 80)
    
    return post_metrics


def save_model(trainer, tokenizer, config: Dict, output_dir: str):
    """Save model and push to Hub if configured."""
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    print(f"Output: {output_dir}")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Model saved")
    
    # Push to Hub
    if config['training']['push_to_hub']:
        print("\nPushing to HuggingFace Hub...")
        try:
            trainer.push_to_hub(commit_message="Training completed")
            tokenizer.push_to_hub(config['training']['hub_model_id'])
            print(f"Pushed to: https://huggingface.co/{config['training']['hub_model_id']}")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
    
    print("=" * 80)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main training orchestration function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = load_config(args.config, args.skip_pre_eval)
    
    # Setup integrations
    setup_wandb(config, args.wandb_token)
    setup_huggingface(config, args.hf_token)
    
    # Load datasets (pre-sampled from balanced_dataset_sampling.ipynb)
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    
    # Dataset is already balanced and sampled in the notebook
    # No sampling needed here - just verify no leakage
    
    # Check for data leakage between splits (verification only)
    leakage_results = check_data_leakage(
        train_dataset, val_dataset, test_dataset, 
        text_column=config['model']['text_column'],
        sample_size=min(10000, len(val_dataset))  # Check up to 10k samples
    )
    
    # Create label mappings and class weights
    label2id, id2label, class_weights = create_label_mappings_and_weights(config, train_dataset)
    
    # Load tokenizer
    global tokenizer
    tokenizer = load_tokenizer(config)
    
    # Tokenize datasets (with caching)
    train_tokenized, val_tokenized, test_tokenized = tokenize_datasets(
        train_dataset, val_dataset, test_dataset, tokenizer, config, label2id
    )
    
    # Load model
    model = load_model(config, label2id, id2label)
    
    # Apply PEFT if enabled
    model = apply_peft(model, config)
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_tokenized, val_tokenized, 
                          class_weights, config)
    
    # Pre-training evaluation
    output_dir = config['training']['output_dir']
    labels = config['model']['labels']
    pre_metrics = run_pre_training_evaluation(trainer, test_tokenized, config, 
                                              id2label, labels, output_dir)
    
    # Training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Training samples: {len(train_tokenized):,}")
    print("=" * 80)
    
    train_result = trainer.train(resume_from_checkpoint=config['misc']['resume_from_checkpoint'])
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")
    print_gpu_memory()
    
    # Post-training evaluation
    post_metrics = run_post_training_evaluation(trainer, test_tokenized, config, 
                                                id2label, labels, output_dir, pre_metrics)
    
    # Save model
    save_model(trainer, tokenizer, config, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: {config['model']['name']}")
    print(f"Training samples: {len(train_tokenized):,}")
    print(f"Method: {'PEFT/LoRA' if config['peft']['enabled'] else 'Full fine-tuning'}")
    
    if post_metrics:
        print(f"\nFinal metrics:")
        for metric, value in post_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nOutput: {output_dir}")
    if config['wandb']['enabled']:
        print(f"W&B Dashboard: {wandb.run.get_url()}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Cleanup
    if config['wandb']['enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()
