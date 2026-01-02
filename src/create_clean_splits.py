#!/usr/bin/env python3
"""
Create Clean Dataset Splits with Zero Overlap

This script creates train/val/test splits using the validation-first approach
with strict deduplication to ensure zero overlap between splits.

Strategy:
1. Load full dataset
2. Deduplicate by text content (keep only unique texts across all domains)
3. Validation-first approach: Reserve validation and test samples first
4. Stratify by domain and label for balanced representation
5. Ensure zero overlap between splits
6. Validate after creation

Usage:
    python create_clean_splits.py --dataset 10k
    python create_clean_splits.py --dataset 100k
    python create_clean_splits.py --dataset 1M
    python create_clean_splits.py --dataset 2M
    python create_clean_splits.py --all  # Create all splits
"""

import os
import sys
import pandas as pd
import numpy as np
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Tuple, Set
from collections import defaultdict
from datetime import datetime


class CleanSplitCreator:
    """Create clean, overlap-free dataset splits"""
    
    def __init__(
        self,
        source_path: str,
        output_path: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize split creator.
        
        Args:
            source_path: Path to source CSV file
            output_path: Path to output directory
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            random_seed: Random seed for reproducibility
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio = 1.0 - val_ratio - test_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
    def hash_text(self, text: str) -> str:
        """Create hash of text for deduplication"""
        if pd.isna(text) or text is None:
            return None
        return hashlib.md5(str(text).encode('utf-8')).hexdigest()
    
    def deduplicate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate texts, keeping only first occurrence.
        
        Args:
            df: Input dataframe
            
        Returns:
            Deduplicated dataframe
        """
        print(f"  Original samples: {len(df):,}")
        
        # Create text hashes
        df['text_hash'] = df['text'].apply(self.hash_text)
        
        # Remove rows with None hash (null texts)
        df = df[df['text_hash'].notna()].copy()
        print(f"  After removing null texts: {len(df):,}")
        
        # Remove duplicates (keep first occurrence)
        df_dedup = df.drop_duplicates(subset='text_hash', keep='first')
        
        duplicates_removed = len(df) - len(df_dedup)
        print(f"  Duplicates removed: {duplicates_removed:,}")
        print(f"  Unique samples: {len(df_dedup):,}")
        
        return df_dedup
    
    def stratified_split(
        self,
        df: pd.DataFrame,
        val_size: int,
        test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified splits by domain and label using validation-first approach.
        
        Args:
            df: Deduplicated dataframe
            val_size: Target validation set size
            test_size: Target test set size
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\nCreating stratified splits...")
        print(f"  Target validation size: {val_size:,}")
        print(f"  Target test size: {test_size:,}")
        
        # Check if domain column exists
        if 'domain_name' not in df.columns:
            print("  WARNING: 'domain_name' column not found, using random split")
            return self._random_split(df, val_size, test_size)
        
        # Get unique domains and labels
        domains = df['domain_name'].unique()
        labels = df['label'].unique()
        
        print(f"  Domains: {len(domains)} ({', '.join(sorted(domains))})")
        print(f"  Labels: {len(labels)} ({', '.join(sorted(labels))})")
        
        # Validation-first approach: Select validation and test first
        val_indices = []
        test_indices = []
        
        # Calculate target samples per domain for val and test
        domain_counts = df['domain_name'].value_counts()
        
        print(f"\n  Sampling validation set...")
        for domain in sorted(domains):
            domain_mask = df['domain_name'] == domain
            domain_df = df[domain_mask]
            domain_proportion = len(domain_df) / len(df)
            
            # Target samples for this domain in validation
            domain_val_target = int(val_size * domain_proportion)
            
            # Further stratify by label within domain
            for label in labels:
                label_mask = domain_df['label'] == label
                domain_label_df = domain_df[label_mask]
                
                if len(domain_label_df) == 0:
                    continue
                
                label_proportion = len(domain_label_df) / len(domain_df)
                domain_label_val_target = max(1, int(domain_val_target * label_proportion))
                
                # Sample for validation
                sample_size = min(domain_label_val_target, len(domain_label_df))
                sampled = domain_label_df.sample(n=sample_size, random_state=self.random_seed)
                val_indices.extend(sampled.index.tolist())
                
                if len(domain_label_df) > 0:
                    print(f"    {domain}/{label}: {sample_size} samples")
        
        print(f"  Total validation samples: {len(val_indices):,}")
        
        # Remove validation indices from available pool
        remaining_df = df[~df.index.isin(val_indices)]
        
        print(f"\n  Sampling test set...")
        for domain in sorted(domains):
            domain_mask = remaining_df['domain_name'] == domain
            domain_df = remaining_df[domain_mask]
            domain_proportion = len(domain_df) / len(remaining_df)
            
            # Target samples for this domain in test
            domain_test_target = int(test_size * domain_proportion)
            
            # Further stratify by label within domain
            for label in labels:
                label_mask = domain_df['label'] == label
                domain_label_df = domain_df[label_mask]
                
                if len(domain_label_df) == 0:
                    continue
                
                label_proportion = len(domain_label_df) / len(domain_df)
                domain_label_test_target = max(1, int(domain_test_target * label_proportion))
                
                # Sample for test
                sample_size = min(domain_label_test_target, len(domain_label_df))
                sampled = domain_label_df.sample(n=sample_size, random_state=self.random_seed + 1)
                test_indices.extend(sampled.index.tolist())
                
                if len(domain_label_df) > 0:
                    print(f"    {domain}/{label}: {sample_size} samples")
        
        print(f"  Total test samples: {len(test_indices):,}")
        
        # Remaining samples go to training
        val_test_indices = set(val_indices + test_indices)
        train_indices = [idx for idx in df.index if idx not in val_test_indices]
        
        print(f"\n  Training samples: {len(train_indices):,}")
        
        # Create split dataframes
        val_df = df.loc[val_indices].copy()
        test_df = df.loc[test_indices].copy()
        train_df = df.loc[train_indices].copy()
        
        # Verify no overlap (safety check)
        train_hashes = set(train_df['text_hash'])
        val_hashes = set(val_df['text_hash'])
        test_hashes = set(test_df['text_hash'])
        
        assert len(train_hashes.intersection(val_hashes)) == 0, "Train/Val overlap detected!"
        assert len(train_hashes.intersection(test_hashes)) == 0, "Train/Test overlap detected!"
        assert len(val_hashes.intersection(test_hashes)) == 0, "Val/Test overlap detected!"
        
        print(f"\n  Verification: PASS - Zero overlaps confirmed")
        
        # Remove hash column before returning
        train_df = train_df.drop(columns=['text_hash'])
        val_df = val_df.drop(columns=['text_hash'])
        test_df = test_df.drop(columns=['text_hash'])
        
        return train_df, val_df, test_df
    
    def _random_split(
        self,
        df: pd.DataFrame,
        val_size: int,
        test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fallback random split when domain stratification not possible"""
        print(f"  Using random split (no domain stratification)")
        
        # Shuffle
        df = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        
        # Split
        val_df = df.iloc[:val_size].copy()
        test_df = df.iloc[val_size:val_size+test_size].copy()
        train_df = df.iloc[val_size+test_size:].copy()
        
        # Remove hash column
        train_df = train_df.drop(columns=['text_hash'])
        val_df = val_df.drop(columns=['text_hash'])
        test_df = test_df.drop(columns=['text_hash'])
        
        return train_df, val_df, test_df
    
    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main method to create clean splits.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n{'='*80}")
        print(f"Loading source dataset: {self.source_path.name}")
        print(f"{'='*80}")
        
        # Load dataset
        df = pd.read_csv(self.source_path)
        print(f"Loaded: {len(df):,} samples")
        
        # Show column info
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Deduplicate
        print(f"\nDeduplicating dataset...")
        df_dedup = self.deduplicate_dataset(df)
        
        # Calculate split sizes
        total_unique = len(df_dedup)
        val_size = int(total_unique * self.val_ratio)
        test_size = int(total_unique * self.test_ratio)
        train_size = total_unique - val_size - test_size
        
        print(f"\nSplit targets:")
        print(f"  Train: {train_size:,} ({self.train_ratio*100:.1f}%)")
        print(f"  Validation: {val_size:,} ({self.val_ratio*100:.1f}%)")
        print(f"  Test: {test_size:,} ({self.test_ratio*100:.1f}%)")
        
        # Create stratified splits
        train_df, val_df, test_df = self.stratified_split(df_dedup, val_size, test_size)
        
        # Print final counts
        print(f"\nFinal split sizes:")
        print(f"  Train: {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test: {len(test_df):,}")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,}")
        
        # Show label distribution in each split
        print(f"\nLabel distribution:")
        print(f"  Train: {dict(train_df['label'].value_counts())}")
        print(f"  Validation: {dict(val_df['label'].value_counts())}")
        print(f"  Test: {dict(test_df['label'].value_counts())}")
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """
        Save splits to CSV files.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
        """
        print(f"\n{'='*80}")
        print(f"Saving splits to: {self.output_path}")
        print(f"{'='*80}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSVs
        train_path = self.output_path / 'train.csv'
        val_path = self.output_path / 'validation.csv'
        test_path = self.output_path / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        print(f"  Saved: train.csv ({len(train_df):,} samples)")
        
        val_df.to_csv(val_path, index=False)
        print(f"  Saved: validation.csv ({len(val_df):,} samples)")
        
        test_df.to_csv(test_path, index=False)
        print(f"  Saved: test.csv ({len(test_df):,} samples)")
        
        # Create metadata file
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            else:
                return obj
        
        metadata = {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': str(self.source_path),
            'deduplication': 'enabled',
            'stratification': 'domain + label',
            'approach': 'validation-first',
            'random_seed': self.random_seed,
            'split_ratios': {
                'train': float(self.train_ratio),
                'validation': float(self.val_ratio),
                'test': float(self.test_ratio)
            },
            'split_sizes': {
                'train': int(len(train_df)),
                'validation': int(len(val_df)),
                'test': int(len(test_df)),
                'total': int(len(train_df) + len(val_df) + len(test_df))
            },
            'label_distribution': {
                'train': convert_to_native(dict(train_df['label'].value_counts())),
                'validation': convert_to_native(dict(val_df['label'].value_counts())),
                'test': convert_to_native(dict(test_df['label'].value_counts()))
            }
        }
        
        import json
        metadata_path = self.output_path / 'splits_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved: splits_metadata.json")
        print(f"\nSplits saved successfully!")
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate class weights from training data.
        
        Args:
            train_df: Training dataframe
            
        Returns:
            Dictionary with class weights
        """
        label_counts = train_df['label'].value_counts()
        total = len(train_df)
        num_classes = len(label_counts)
        
        weights = {}
        for label, count in label_counts.items():
            weight = total / (num_classes * count)
            weights[label] = weight
        
        return weights


def create_dataset_splits(
    dataset_size: str,
    source_base: str,
    output_base: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> bool:
    """
    Create splits for a specific dataset size.
    
    Args:
        dataset_size: Dataset size ("10k", "100k", "1M", "2M")
        source_base: Base path to source datasets
        output_base: Base path to output directory
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        
    Returns:
        True if successful, False otherwise
    """
    # Map dataset size to source file
    source_files = {
        "10k": "datasets_ai_human_text_10000.csv",
        "100k": "datasets_ai_human_text_100000.csv",
        "1M": "datasets_ai_human_text_1000000.csv",
        "2M": "datasets_ai_human_text_2000000.csv"
    }
    
    if dataset_size not in source_files:
        print(f"ERROR: Invalid dataset size: {dataset_size}")
        print(f"Options: {list(source_files.keys())}")
        return False
    
    print(f"\n{'#'*80}")
    print(f"Creating Clean Splits for {dataset_size} Dataset".center(80))
    print(f"{'#'*80}")
    
    source_path = Path(source_base) / source_files[dataset_size]
    output_path = Path(output_base) / dataset_size
    
    if not source_path.exists():
        print(f"ERROR: Source file not found: {source_path}")
        return False
    
    # Create splits
    creator = CleanSplitCreator(
        source_path=source_path,
        output_path=output_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    try:
        # Create splits
        train_df, val_df, test_df = creator.create_splits()
        
        # Save splits
        creator.save_splits(train_df, val_df, test_df)
        
        # Calculate and print class weights
        print(f"\n{'='*80}")
        print(f"Class Weights for {dataset_size} Dataset")
        print(f"{'='*80}")
        
        weights = creator.calculate_class_weights(train_df)
        print(f"\nInverse frequency weights:")
        for label, weight in sorted(weights.items()):
            print(f"  {label}: {weight:.4f}")
        
        # Format for Python code
        ai_weight = weights.get('AI_Generated', 1.0)
        human_weight = weights.get('Human_Written', 1.0)
        print(f"\nFor train_qwen.py:")
        print(f'  "class_weights": [{ai_weight:.4f}, {human_weight:.4f}]  # AI_Generated, Human_Written')
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to create splits for {dataset_size}")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def upload_to_huggingface(
    dataset_path: Path,
    dataset_size: str,
    hf_user: str = 'codefactory4791',
    hf_token: str = None
) -> bool:
    """
    Upload dataset splits to HuggingFace Hub.
    
    Args:
        dataset_path: Path to dataset folder containing train/val/test CSVs
        dataset_size: Dataset size identifier ("10k", "100k", "1M", "2M")
        hf_user: HuggingFace username
        hf_token: HuggingFace API token
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'#'*80}")
    print(f"Uploading {dataset_size} to HuggingFace Hub".center(80))
    print(f"{'#'*80}")
    
    try:
        from datasets import Dataset, DatasetDict
        import pandas as pd
        
        # Map size to repo name
        size_map = {'10k': '10k', '100k': '100k', '1M': '1000k', '2M': '2000k'}
        repo_name = f"{hf_user}/raid_aligned_{size_map[dataset_size]}"
        
        print(f"\nRepository: {repo_name}")
        print(f"Loading splits from: {dataset_path}")
        
        # Load splits
        train_df = pd.read_csv(dataset_path / 'train.csv')
        val_df = pd.read_csv(dataset_path / 'validation.csv')
        test_df = pd.read_csv(dataset_path / 'test.csv')
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Validation: {len(val_df):,} samples")
        print(f"  Test: {len(test_df):,} samples")
        
        # Create HuggingFace datasets
        dataset_dict = {
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'validation': Dataset.from_pandas(val_df, preserve_index=False),
            'test': Dataset.from_pandas(test_df, preserve_index=False)
        }
        
        # Create DatasetDict
        ds_dict = DatasetDict(dataset_dict)
        
        print(f"\nDataset structure:")
        print(ds_dict)
        
        # Push to hub
        print(f"\nPushing to HuggingFace Hub...")
        print(f"This may take a few minutes...")
        
        ds_dict.push_to_hub(
            repo_id=repo_name,
            token=hf_token,
            private=False
        )
        
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
        print(f"   The old splits have been replaced with clean V2 splits")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ HuggingFace libraries not available: {e}")
        print(f"   Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"\n❌ Upload failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def validate_created_splits(output_base: str, dataset_size: str) -> bool:
    """
    Validate created splits for overlaps using inline validation.
    
    Args:
        output_base: Base path to output directory
        dataset_size: Dataset size to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'#'*80}")
    print(f"Validating Created Splits for {dataset_size}".center(80))
    print(f"{'#'*80}")
    
    dataset_path = Path(output_base) / dataset_size
    
    def hash_text(text):
        if pd.isna(text) or text is None:
            return None
        return hashlib.md5(str(text).encode('utf-8')).hexdigest()
    
    # Load splits
    train_df = pd.read_csv(dataset_path / 'train.csv')
    val_df = pd.read_csv(dataset_path / 'validation.csv')
    test_df = pd.read_csv(dataset_path / 'test.csv')
    
    # Create hash sets
    train_hashes = set(train_df['text'].apply(hash_text).dropna())
    val_hashes = set(val_df['text'].apply(hash_text).dropna())
    test_hashes = set(test_df['text'].apply(hash_text).dropna())
    
    # Check overlaps
    train_val_overlap = len(train_hashes.intersection(val_hashes))
    train_test_overlap = len(train_hashes.intersection(test_hashes))
    val_test_overlap = len(val_hashes.intersection(test_hashes))
    
    total_overlaps = train_val_overlap + train_test_overlap + val_test_overlap
    
    print(f"\nOverlap check:")
    print(f"  Train vs Validation: {train_val_overlap}")
    print(f"  Train vs Test: {train_test_overlap}")
    print(f"  Validation vs Test: {val_test_overlap}")
    print(f"  Total overlaps: {total_overlaps}")
    
    if total_overlaps == 0:
        print(f"\n✅ VALIDATION PASSED: No overlaps found in {dataset_size} dataset")
        return True
    else:
        print(f"\n❌ VALIDATION FAILED: {total_overlaps} overlaps found in {dataset_size} dataset")
        print(f"   This should not happen - please review the split creation logic")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create clean dataset splits with zero overlap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create splits for specific dataset:
    python create_clean_splits.py --dataset 10k
    python create_clean_splits.py --dataset 100k
    python create_clean_splits.py --dataset 1M
    python create_clean_splits.py --dataset 2M
  
  Create splits for all datasets:
    python create_clean_splits.py --all
  
  Create and upload to HuggingFace:
    python create_clean_splits.py --all --upload --hf-token YOUR_TOKEN
    python create_clean_splits.py --dataset 10k --upload --hf-token YOUR_TOKEN
  
  Upload existing v2 splits to HuggingFace:
    python create_clean_splits.py --all --upload --hf-token YOUR_TOKEN --no-validate
  
  Custom split ratios:
    python create_clean_splits.py --dataset 10k --val-ratio 0.15 --test-ratio 0.15
  
  Custom paths:
    python create_clean_splits.py --dataset 10k --source /custom/path --output /custom/output
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['10k', '100k', '1M', '2M'],
        help='Dataset size to process'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='../dataset/prepared_datasets',
        help='Path to source datasets folder (default: ../dataset/prepared_datasets)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../dataset/prepared_splits_v2',
        help='Path to output folder (default: ../dataset/prepared_splits_v2)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after creating splits'
    )
    
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload splits to HuggingFace Hub after creation'
    )
    
    parser.add_argument(
        '--hf-user',
        type=str,
        default='codefactory4791',
        help='HuggingFace username (default: codefactory4791)'
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace API token (required for --upload)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")
    
    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("val_ratio + test_ratio must be < 1.0")
    
    # Check HuggingFace upload requirements
    if args.upload and not args.hf_token:
        parser.error("--upload requires --hf-token to be specified")
    
    # Resolve paths
    script_dir = Path(__file__).parent
    source_base = (script_dir / args.source).resolve()
    output_base = (script_dir / args.output).resolve()
    
    if not source_base.exists():
        print(f"ERROR: Source path does not exist: {source_base}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"Clean Split Creation".center(80))
    print(f"{'#'*80}")
    print(f"\nSource: {source_base}")
    print(f"Output: {output_base}")
    print(f"Split ratios: Train={1-args.val_ratio-args.test_ratio:.1%}, Val={args.val_ratio:.1%}, Test={args.test_ratio:.1%}")
    print(f"Upload to HuggingFace: {'Yes' if args.upload else 'No'}")
    if args.upload:
        print(f"HuggingFace user: {args.hf_user}")
    
    # Determine which datasets to process
    if args.all:
        datasets = ['10k', '100k', '1M', '2M']
    else:
        datasets = [args.dataset]
    
    print(f"Datasets to process: {', '.join(datasets)}")
    
    # Process each dataset
    results = {}
    upload_results = {}
    
    for dataset in datasets:
        success = create_dataset_splits(
            dataset_size=dataset,
            source_base=str(source_base),
            output_base=str(output_base),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        results[dataset] = success
        
        # Validate if requested
        if success and not args.no_validate:
            validation_passed = validate_created_splits(str(output_base), dataset)
            results[dataset] = validation_passed
        
        # Upload to HuggingFace if requested and validation passed
        if success and args.upload and results[dataset]:
            dataset_path = output_base / dataset
            upload_success = upload_to_huggingface(
                dataset_path=dataset_path,
                dataset_size=dataset,
                hf_user=args.hf_user,
                hf_token=args.hf_token
            )
            upload_results[dataset] = upload_success
            
            # If upload failed, mark overall result as failed
            if not upload_success:
                results[dataset] = False
    
    # Print summary
    print(f"\n\n{'#'*80}")
    print(f"Summary".center(80))
    print(f"{'#'*80}\n")
    
    if args.upload:
        print(f"{'Dataset':<15} {'Creation':<15} {'Upload':<15} {'Result'}")
        print(f"{'-'*80}")
        for dataset in datasets:
            creation_status = "✅ SUCCESS" if results.get(dataset, False) else "❌ FAILED"
            upload_status = "✅ SUCCESS" if upload_results.get(dataset, False) else "❌ FAILED" if dataset in upload_results else "⊘ SKIPPED"
            overall = "✅" if results.get(dataset, False) and (not args.upload or upload_results.get(dataset, False)) else "❌"
            print(f"{dataset:<15} {creation_status:<15} {upload_status:<15} {overall}")
    else:
        print(f"{'Dataset':<15} {'Status':<15} {'Result'}")
        print(f"{'-'*80}")
        for dataset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{dataset:<15} {status:<15}")
    
    # Overall result
    all_passed = all(results.values())
    if args.upload:
        all_uploaded = all(upload_results.get(d, False) for d in datasets if results.get(d, False))
        all_passed = all_passed and all_uploaded
    
    print(f"\n{'='*80}")
    if all_passed:
        print(f"ALL DATASETS PROCESSED SUCCESSFULLY")
        print(f"Output location: {output_base}")
        if args.upload:
            print(f"\n✅ All splits uploaded to HuggingFace Hub:")
            size_map = {'10k': '10k', '100k': '100k', '1M': '1000k', '2M': '2000k'}
            for dataset in datasets:
                if results.get(dataset, False):
                    repo_name = f"{args.hf_user}/raid_aligned_{size_map[dataset]}"
                    print(f"   - https://huggingface.co/datasets/{repo_name}")
        print(f"\nNext step: Use these clean splits for training with train_qwen.py")
    else:
        print(f"SOME DATASETS FAILED")
        failed = [d for d, s in results.items() if not s]
        print(f"Failed: {', '.join(failed)}")
        if args.upload and upload_results:
            upload_failed = [d for d in datasets if results.get(d, False) and not upload_results.get(d, False)]
            if upload_failed:
                print(f"Upload failed: {', '.join(upload_failed)}")
    print(f"{'='*80}\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

