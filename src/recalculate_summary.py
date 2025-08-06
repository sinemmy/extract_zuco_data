#!/usr/bin/env python3
"""
Recalculate extraction summary with accurate counts of words with EEG data
Previous version only counted the first 100 words in each file, which led to inaccurate statistics.
"""

import h5py
import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

def check_file_stats(filepath):
    """Get accurate statistics for a single file"""
    with h5py.File(filepath, 'r') as f:
        total_words = len(f.keys())
        words_with_eeg = 0
        words_with_bands = 0
        words_with_fixations = 0
        
        for key in f.keys():
            word_group = f[key]
            
            # Check for raw EEG with actual data
            if 'raw_eeg' in word_group:
                raw_eeg = word_group['raw_eeg'][:]
                if raw_eeg.size > 0:  # Has actual data, not just empty array
                    words_with_eeg += 1
            
            # Check for frequency bands with actual data
            has_bands = False
            for band_name in ['FFD_t1', 'FFD_t2', 'FFD_a1', 'FFD_a2', 
                             'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2']:
                if band_name in word_group:
                    band_data = word_group[band_name][:]
                    if band_data.size > 0:
                        has_bands = True
                        break
            if has_bands:
                words_with_bands += 1
            
            # Check fixation attribute
            if word_group.attrs.get('has_fixation', False):
                words_with_fixations += 1
        
        return {
            'total': total_words,
            'with_eeg': words_with_eeg,
            'with_bands': words_with_bands,
            'with_fixations': words_with_fixations
        }

def recalculate_summary():
    """Recalculate extraction summary with accurate statistics"""
    
    extracted_dir = Path('extracted_data')
    
    # Initialize summary
    summary = {
        'total_words': 0,
        'total_files': 0,
        'versions': defaultdict(int),
        'subjects': defaultdict(int),
        'tasks': defaultdict(int),
        'words_with_fixations': 0,
        'words_with_raw_eeg': 0,
        'words_with_frequency_bands': 0,
        'file_statistics': {},  # Store per-file stats
        'extraction_completed': datetime.now().isoformat()
    }
    
    # Get all HDF5 files
    h5_files = sorted(extracted_dir.glob('*.h5'))
    
    print(f"Found {len(h5_files)} HDF5 files to analyze")
    
    # Process each file
    for h5_file in tqdm(h5_files, desc="Analyzing files"):
        summary['total_files'] += 1
        
        # Parse filename to get version, subject, task
        parts = h5_file.stem.split('_')
        if len(parts) >= 3:
            version = parts[0]
            subject_id = parts[1]
            task_name = parts[2]
            
            summary['versions'][version] += 1
            
            # Get accurate statistics for this file
            file_stats = check_file_stats(h5_file)
            
            # Update totals
            summary['total_words'] += file_stats['total']
            summary['subjects'][subject_id] += file_stats['total']
            summary['tasks'][task_name] += file_stats['total']
            summary['words_with_fixations'] += file_stats['with_fixations']
            summary['words_with_raw_eeg'] += file_stats['with_eeg']
            summary['words_with_frequency_bands'] += file_stats['with_bands']
            
            # Store per-file statistics
            summary['file_statistics'][h5_file.name] = {
                'total_words': file_stats['total'],
                'words_with_eeg': file_stats['with_eeg'],
                'words_with_bands': file_stats['with_bands'],
                'words_with_fixations': file_stats['with_fixations'],
                'percentage_with_eeg': round(100 * file_stats['with_eeg'] / file_stats['total'], 1) if file_stats['total'] > 0 else 0
            }
    
    # Convert defaultdicts to regular dicts
    summary['versions'] = dict(summary['versions'])
    summary['subjects'] = dict(summary['subjects'])
    summary['tasks'] = dict(summary['tasks'])
    
    # Add overall percentages
    if summary['total_words'] > 0:
        summary['overall_percentages'] = {
            'words_with_eeg': round(100 * summary['words_with_raw_eeg'] / summary['total_words'], 1),
            'words_with_frequency_bands': round(100 * summary['words_with_frequency_bands'] / summary['total_words'], 1),
            'words_with_fixations': round(100 * summary['words_with_fixations'] / summary['total_words'], 1)
        }
    
    return summary

def main():
    print("Recalculating extraction summary with accurate counts...")
    print("This will analyze all extracted HDF5 files\n")
    
    summary = recalculate_summary()
    
    # Save the corrected summary
    output_path = Path('extracted_data/extraction_summary_full.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Corrected Extraction Summary:")
    print(f"{'='*60}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total words: {summary['total_words']:,}")
    print(f"\nWords with data:")
    print(f"  With raw EEG: {summary['words_with_raw_eeg']:,} ({summary['overall_percentages']['words_with_eeg']:.1f}%)")
    print(f"  With frequency bands: {summary['words_with_frequency_bands']:,} ({summary['overall_percentages']['words_with_frequency_bands']:.1f}%)")
    print(f"  With fixations: {summary['words_with_fixations']:,} ({summary['overall_percentages']['words_with_fixations']:.1f}%)")
    print(f"\nSummary saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()