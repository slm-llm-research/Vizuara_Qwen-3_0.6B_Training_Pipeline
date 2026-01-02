#!/usr/bin/env python3
"""Quick validation script for prepared_splits_v2"""

import pandas as pd
import hashlib
from pathlib import Path

def hash_text(text):
    if pd.isna(text) or text is None:
        return None
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

def validate_dataset(dataset_path):
    """Validate a single dataset for overlaps"""
    dataset_name = dataset_path.name
    print(f"\n{'='*80}")
    print(f"Validating: {dataset_name}")
    print(f"{'='*80}")
    
    # Load files
    train_df = pd.read_csv(dataset_path / 'train.csv')
    val_df = pd.read_csv(dataset_path / 'validation.csv')
    test_df = pd.read_csv(dataset_path / 'test.csv')
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # Create hash sets
    train_hashes = set(train_df['text'].apply(hash_text).dropna())
    val_hashes = set(val_df['text'].apply(hash_text).dropna())
    test_hashes = set(test_df['text'].apply(hash_text).dropna())
    
    # Check overlaps
    train_val = len(train_hashes.intersection(val_hashes))
    train_test = len(train_hashes.intersection(test_hashes))
    val_test = len(val_hashes.intersection(test_hashes))
    
    total = train_val + train_test + val_test
    
    print(f"\n  Train vs Validation: {train_val} overlaps")
    print(f"  Train vs Test: {train_test} overlaps")
    print(f"  Validation vs Test: {val_test} overlaps")
    print(f"  Total: {total} overlaps")
    
    # Label distribution
    train_labels = train_df['label'].value_counts()
    print(f"\n  Label distribution (Train):")
    for label, count in train_labels.items():
        print(f"    {label}: {count:,} ({count/len(train_df)*100:.2f}%)")
    
    # Calculate class weights
    total_samples = len(train_df)
    num_classes = len(train_labels)
    print(f"\n  Class weights (inverse frequency):")
    for label, count in sorted(train_labels.items()):
        weight = total_samples / (num_classes * count)
        print(f"    {label}: {weight:.4f}")
    
    if total == 0:
        print(f"\n  ✅ PASS - No overlaps")
        return True
    else:
        print(f"\n  ❌ FAIL - {total} overlaps found!")
        return False

# Main
base_path = Path("../dataset/prepared_splits_v2")

print(f"\n{'#'*80}")
print(f"Validating Clean Splits (v2)".center(80))
print(f"{'#'*80}")
print(f"\nBase path: {base_path.resolve()}")

datasets = sorted([d for d in base_path.iterdir() if d.is_dir()])

results = {}
for dataset_path in datasets:
    results[dataset_path.name] = validate_dataset(dataset_path)

# Summary
print(f"\n\n{'#'*80}")
print(f"Summary".center(80))
print(f"{'#'*80}\n")

print(f"{'Dataset':<15} {'Status':<10}")
print(f"{'-'*30}")
for dataset, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{dataset:<15} {status:<10}")

all_passed = all(results.values())
print(f"\n{'='*80}")
if all_passed:
    print("ALL DATASETS VALIDATED SUCCESSFULLY")
    print("Zero overlaps - safe to use for training")
else:
    print("SOME DATASETS HAVE OVERLAPS")
print(f"{'='*80}\n")

