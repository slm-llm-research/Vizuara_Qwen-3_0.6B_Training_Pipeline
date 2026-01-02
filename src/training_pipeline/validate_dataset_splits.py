#!/usr/bin/env python3
"""
Dataset Split Validation Script

Validates that train, validation, and test splits have no overlapping examples
across all datasets in the prepared_splits folder.

This script checks for data leakage by ensuring no text appears in multiple splits.
Even a single overlapping example is reported as a validation failure.

Usage:
    python validate_dataset_splits.py
    
    # With verbose output
    python validate_dataset_splits.py --verbose
    
    # Check specific dataset only
    python validate_dataset_splits.py --dataset 10k
    
    # Save detailed report
    python validate_dataset_splits.py --output validation_report.txt
"""

import os
import sys
import pandas as pd
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple
from datetime import datetime
from collections import defaultdict


class DatasetValidator:
    """Validate dataset splits for overlaps and data leakage"""
    
    def __init__(self, base_path: str, verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            base_path: Path to prepared_splits folder
            verbose: Whether to print detailed information
        """
        self.base_path = Path(base_path)
        self.verbose = verbose
        self.results = {}
        self.total_overlaps = 0
        
    def hash_text(self, text: str) -> str:
        """Create hash of text for efficient comparison"""
        if pd.isna(text) or text is None:
            return None
        # Use MD5 hash for fast comparison
        return hashlib.md5(str(text).encode('utf-8')).hexdigest()
    
    def load_split_hashes(self, csv_path: Path, text_column: str = 'text') -> Set[str]:
        """
        Load CSV and return set of text hashes.
        
        Args:
            csv_path: Path to CSV file
            text_column: Column containing text data
            
        Returns:
            Set of text hashes
        """
        if not csv_path.exists():
            return set()
        
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}")
        
        # Create hashes for all texts
        hashes = set()
        for text in df[text_column]:
            text_hash = self.hash_text(text)
            if text_hash:
                hashes.add(text_hash)
        
        return hashes
    
    def find_overlaps(self, set1: Set[str], set2: Set[str]) -> Set[str]:
        """Find overlapping hashes between two sets"""
        return set1.intersection(set2)
    
    def validate_dataset(self, dataset_name: str) -> Dict:
        """
        Validate a single dataset for split overlaps.
        
        Args:
            dataset_name: Name of dataset folder (e.g., "10k", "100k")
            
        Returns:
            Dictionary with validation results
        """
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            return {
                'status': 'ERROR',
                'message': f'Dataset folder not found: {dataset_path}',
                'overlaps': {}
            }
        
        print(f"\n{'='*80}")
        print(f"Validating Dataset: {dataset_name}")
        print(f"Path: {dataset_path}")
        print(f"{'='*80}")
        
        # Find all CSV files in the directory
        csv_files = list(dataset_path.glob('*.csv'))
        
        if not csv_files:
            return {
                'status': 'ERROR',
                'message': 'No CSV files found',
                'overlaps': {}
            }
        
        # Identify split files
        train_file = dataset_path / 'train.csv'
        val_files = list(dataset_path.glob('val*.csv'))
        test_files = list(dataset_path.glob('test*.csv'))
        
        # Also check for validation.csv
        validation_file = dataset_path / 'validation.csv'
        if validation_file.exists():
            val_files.append(validation_file)
        
        if not train_file.exists():
            return {
                'status': 'ERROR',
                'message': 'train.csv not found',
                'overlaps': {}
            }
        
        # Load train split
        print(f"\nLoading train.csv...")
        train_hashes = self.load_split_hashes(train_file)
        print(f"  Train samples: {len(train_hashes):,}")
        
        # Load validation splits
        val_hashes = set()
        for val_file in val_files:
            print(f"Loading {val_file.name}...")
            val_split_hashes = self.load_split_hashes(val_file)
            print(f"  Validation samples: {len(val_split_hashes):,}")
            val_hashes.update(val_split_hashes)
        
        if val_files:
            print(f"  Total unique validation samples: {len(val_hashes):,}")
        
        # Load test splits
        test_hashes = set()
        for test_file in test_files:
            print(f"Loading {test_file.name}...")
            test_split_hashes = self.load_split_hashes(test_file)
            print(f"  Test samples: {len(test_split_hashes):,}")
            test_hashes.update(test_split_hashes)
        
        if test_files:
            print(f"  Total unique test samples: {len(test_hashes):,}")
        
        # Check for overlaps
        print(f"\nChecking for overlaps...")
        
        overlaps = {}
        
        # Train vs Validation
        if val_hashes:
            train_val_overlap = self.find_overlaps(train_hashes, val_hashes)
            overlaps['train_val'] = len(train_val_overlap)
            if train_val_overlap:
                print(f"  WARNING: {len(train_val_overlap)} overlaps between train and validation!")
            else:
                print(f"  OK: No overlaps between train and validation")
        
        # Train vs Test
        if test_hashes:
            train_test_overlap = self.find_overlaps(train_hashes, test_hashes)
            overlaps['train_test'] = len(train_test_overlap)
            if train_test_overlap:
                print(f"  WARNING: {len(train_test_overlap)} overlaps between train and test!")
            else:
                print(f"  OK: No overlaps between train and test")
        
        # Validation vs Test
        if val_hashes and test_hashes:
            val_test_overlap = self.find_overlaps(val_hashes, test_hashes)
            overlaps['val_test'] = len(val_test_overlap)
            if val_test_overlap:
                print(f"  WARNING: {len(val_test_overlap)} overlaps between validation and test!")
            else:
                print(f"  OK: No overlaps between validation and test")
        
        # Determine status
        total_overlaps = sum(overlaps.values())
        status = 'PASS' if total_overlaps == 0 else 'FAIL'
        
        result = {
            'status': status,
            'dataset': dataset_name,
            'counts': {
                'train': len(train_hashes),
                'validation': len(val_hashes),
                'test': len(test_hashes)
            },
            'overlaps': overlaps,
            'total_overlaps': total_overlaps
        }
        
        # Print summary
        print(f"\n{'-'*80}")
        if status == 'PASS':
            print(f"RESULT: PASS - No overlaps found")
        else:
            print(f"RESULT: FAIL - {total_overlaps} total overlaps found")
        print(f"{'-'*80}")
        
        return result
    
    def validate_all_datasets(self, dataset_filter: str = None) -> Dict:
        """
        Validate all datasets in the prepared_splits folder.
        
        Args:
            dataset_filter: Optional filter to check only specific dataset
            
        Returns:
            Dictionary with results for all datasets
        """
        print(f"\n{'#'*80}")
        print(f"Dataset Split Validation".center(80))
        print(f"{'#'*80}")
        print(f"\nBase path: {self.base_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find all dataset folders
        if dataset_filter:
            dataset_folders = [dataset_filter]
        else:
            dataset_folders = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        
        dataset_folders = sorted(dataset_folders)
        
        print(f"\nDatasets to validate: {', '.join(dataset_folders)}")
        
        # Validate each dataset
        all_results = {}
        for dataset_name in dataset_folders:
            result = self.validate_dataset(dataset_name)
            all_results[dataset_name] = result
            self.total_overlaps += result.get('total_overlaps', 0)
        
        self.results = all_results
        return all_results
    
    def print_summary(self):
        """Print summary of validation results"""
        print(f"\n\n{'#'*80}")
        print(f"Validation Summary".center(80))
        print(f"{'#'*80}\n")
        
        # Summary table
        print(f"{'Dataset':<15} {'Status':<10} {'Train':>10} {'Val':>10} {'Test':>10} {'Overlaps':>10}")
        print(f"{'-'*80}")
        
        for dataset, result in self.results.items():
            if result['status'] == 'ERROR':
                print(f"{dataset:<15} {'ERROR':<10} {result['message']}")
                continue
            
            status = result['status']
            counts = result['counts']
            total_overlaps = result['total_overlaps']
            
            status_symbol = 'PASS' if status == 'PASS' else 'FAIL'
            
            print(f"{dataset:<15} {status_symbol:<10} {counts['train']:>10,} {counts['validation']:>10,} "
                  f"{counts['test']:>10,} {total_overlaps:>10}")
        
        print(f"{'-'*80}")
        
        # Overall status
        failed_datasets = [d for d, r in self.results.items() if r['status'] == 'FAIL']
        
        print(f"\n{'='*80}")
        if not failed_datasets:
            print(f"OVERALL RESULT: ALL DATASETS PASSED VALIDATION")
            print(f"No overlaps found in any dataset splits")
        else:
            print(f"OVERALL RESULT: {len(failed_datasets)} DATASET(S) FAILED VALIDATION")
            print(f"Failed datasets: {', '.join(failed_datasets)}")
            print(f"Total overlaps across all datasets: {self.total_overlaps}")
        print(f"{'='*80}\n")
    
    def print_detailed_report(self):
        """Print detailed overlap information for each dataset"""
        print(f"\n\n{'#'*80}")
        print(f"Detailed Overlap Report".center(80))
        print(f"{'#'*80}\n")
        
        for dataset, result in self.results.items():
            if result['status'] == 'ERROR' or result['total_overlaps'] == 0:
                continue
            
            print(f"\nDataset: {dataset}")
            print(f"{'-'*80}")
            
            overlaps = result['overlaps']
            
            if overlaps.get('train_val', 0) > 0:
                print(f"  Train vs Validation: {overlaps['train_val']} overlapping examples")
            
            if overlaps.get('train_test', 0) > 0:
                print(f"  Train vs Test: {overlaps['train_test']} overlapping examples")
            
            if overlaps.get('val_test', 0) > 0:
                print(f"  Validation vs Test: {overlaps['val_test']} overlapping examples")
            
            print(f"  Total: {result['total_overlaps']} overlapping examples")
    
    def save_report(self, output_file: str):
        """Save validation report to file"""
        with open(output_file, 'w') as f:
            f.write(f"Dataset Split Validation Report\n")
            f.write(f"{'='*80}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base path: {self.base_path}\n\n")
            
            # Summary table
            f.write(f"{'Dataset':<15} {'Status':<10} {'Train':>10} {'Val':>10} {'Test':>10} {'Overlaps':>10}\n")
            f.write(f"{'-'*80}\n")
            
            for dataset, result in self.results.items():
                if result['status'] == 'ERROR':
                    f.write(f"{dataset:<15} {'ERROR':<10} {result['message']}\n")
                    continue
                
                status = result['status']
                counts = result['counts']
                total_overlaps = result['total_overlaps']
                
                f.write(f"{dataset:<15} {status:<10} {counts['train']:>10,} {counts['validation']:>10,} "
                       f"{counts['test']:>10,} {total_overlaps:>10}\n")
            
            f.write(f"{'-'*80}\n\n")
            
            # Detailed overlaps
            f.write(f"\nDetailed Overlap Information\n")
            f.write(f"{'='*80}\n\n")
            
            for dataset, result in self.results.items():
                if result['status'] == 'ERROR' or result['total_overlaps'] == 0:
                    continue
                
                f.write(f"\nDataset: {dataset}\n")
                f.write(f"{'-'*80}\n")
                
                overlaps = result['overlaps']
                
                if overlaps.get('train_val', 0) > 0:
                    f.write(f"  Train vs Validation: {overlaps['train_val']} overlaps\n")
                
                if overlaps.get('train_test', 0) > 0:
                    f.write(f"  Train vs Test: {overlaps['train_test']} overlaps\n")
                
                if overlaps.get('val_test', 0) > 0:
                    f.write(f"  Validation vs Test: {overlaps['val_test']} overlaps\n")
                
                f.write(f"  Total: {result['total_overlaps']} overlaps\n")
            
            # Overall summary
            failed_datasets = [d for d, r in self.results.items() if r['status'] == 'FAIL']
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Overall Result\n")
            f.write(f"{'='*80}\n")
            
            if not failed_datasets:
                f.write(f"STATUS: ALL DATASETS PASSED\n")
                f.write(f"No overlaps found in any dataset splits\n")
            else:
                f.write(f"STATUS: {len(failed_datasets)} DATASET(S) FAILED\n")
                f.write(f"Failed datasets: {', '.join(failed_datasets)}\n")
                f.write(f"Total overlaps: {self.total_overlaps}\n")
        
        print(f"\nReport saved to: {output_file}")
    
    def get_overlap_details(self, dataset_name: str, text_column: str = 'text') -> Dict:
        """
        Get detailed information about overlapping examples.
        
        Args:
            dataset_name: Name of dataset to analyze
            text_column: Column containing text
            
        Returns:
            Dictionary with overlap details including actual texts
        """
        dataset_path = self.base_path / dataset_name
        
        # Load dataframes
        train_df = pd.read_csv(dataset_path / 'train.csv')
        
        # Load validation files
        val_files = list(dataset_path.glob('val*.csv'))
        if (dataset_path / 'validation.csv').exists():
            val_files.append(dataset_path / 'validation.csv')
        
        val_dfs = [pd.read_csv(f) for f in val_files]
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        
        # Load test files
        test_files = list(dataset_path.glob('test*.csv'))
        test_dfs = [pd.read_csv(f) for f in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        # Create text to hash mapping
        def create_text_map(df):
            text_map = {}
            for idx, row in df.iterrows():
                text = row[text_column]
                if pd.notna(text):
                    text_hash = self.hash_text(text)
                    if text_hash:
                        text_map[text_hash] = text
            return text_map
        
        train_map = create_text_map(train_df)
        val_map = create_text_map(val_df) if not val_df.empty else {}
        test_map = create_text_map(test_df) if not test_df.empty else {}
        
        # Find overlaps
        overlap_details = {}
        
        if val_map:
            train_val_overlaps = set(train_map.keys()).intersection(set(val_map.keys()))
            overlap_details['train_val'] = {
                'count': len(train_val_overlaps),
                'examples': [train_map[h][:100] + '...' for h in list(train_val_overlaps)[:5]]  # First 5
            }
        
        if test_map:
            train_test_overlaps = set(train_map.keys()).intersection(set(test_map.keys()))
            overlap_details['train_test'] = {
                'count': len(train_test_overlaps),
                'examples': [train_map[h][:100] + '...' for h in list(train_test_overlaps)[:5]]
            }
        
        if val_map and test_map:
            val_test_overlaps = set(val_map.keys()).intersection(set(test_map.keys()))
            overlap_details['val_test'] = {
                'count': len(val_test_overlaps),
                'examples': [val_map[h][:100] + '...' for h in list(val_test_overlaps)[:5]]
            }
        
        return overlap_details


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(
        description='Validate dataset splits for overlaps and data leakage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Validate all datasets:
    python validate_dataset_splits.py
  
  Validate specific dataset:
    python validate_dataset_splits.py --dataset 10k
  
  Verbose output with examples:
    python validate_dataset_splits.py --verbose
  
  Save report to file:
    python validate_dataset_splits.py --output validation_report.txt
  
  Show overlap examples:
    python validate_dataset_splits.py --show-examples --dataset 100k
        """
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='../dataset/prepared_splits',
        help='Path to prepared_splits folder (default: ../dataset/prepared_splits)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Validate specific dataset only (e.g., 10k, 100k, 1m, 2m)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during validation'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save validation report to file'
    )
    
    parser.add_argument(
        '--show-examples',
        action='store_true',
        help='Show example overlapping texts (first 100 chars of first 5 examples)'
    )
    
    args = parser.parse_args()
    
    # Resolve base path
    script_dir = Path(__file__).parent
    base_path = (script_dir / args.base_path).resolve()
    
    if not base_path.exists():
        print(f"ERROR: Base path does not exist: {base_path}")
        print(f"\nPlease provide the correct path using --base-path")
        sys.exit(1)
    
    # Create validator
    validator = DatasetValidator(base_path, verbose=args.verbose)
    
    # Run validation
    try:
        results = validator.validate_all_datasets(dataset_filter=args.dataset)
        
        # Print summary
        validator.print_summary()
        
        # Print detailed report if there are overlaps
        if validator.total_overlaps > 0:
            validator.print_detailed_report()
        
        # Show example overlaps if requested
        if args.show_examples and validator.total_overlaps > 0:
            print(f"\n\n{'#'*80}")
            print(f"Example Overlapping Texts".center(80))
            print(f"{'#'*80}\n")
            
            for dataset, result in results.items():
                if result['status'] == 'FAIL':
                    print(f"\nDataset: {dataset}")
                    print(f"{'-'*80}")
                    
                    overlap_details = validator.get_overlap_details(dataset)
                    
                    for overlap_type, details in overlap_details.items():
                        if details['count'] > 0:
                            print(f"\n  {overlap_type.replace('_', ' vs ').title()}:")
                            print(f"  Count: {details['count']}")
                            print(f"  Examples (first 100 chars):")
                            for i, example in enumerate(details['examples'], 1):
                                print(f"    {i}. {example}")
        
        # Save report if requested
        if args.output:
            validator.save_report(args.output)
        
        # Exit with appropriate code
        if validator.total_overlaps > 0:
            print(f"\nValidation FAILED: {validator.total_overlaps} total overlaps found")
            sys.exit(1)
        else:
            print(f"\nValidation PASSED: No overlaps found")
            sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: Validation failed with exception:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

