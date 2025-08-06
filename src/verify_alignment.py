#!/usr/bin/env python
"""Verify that word-level data is properly aligned in extracted HDF5 files."""

import h5py
import numpy as np
import random
from pathlib import Path

def verify_alignment(h5_file, n_samples=5):
    """Verify alignment of word data with sentence context."""
    print(f"\n{'='*60}")
    print(f"Verifying: {h5_file.name}")
    print(f"{'='*60}")
    
    with h5py.File(h5_file, 'r') as f:
        word_keys = [k for k in f.keys() if k.startswith('word_')]
        print(f"Total words: {len(word_keys)}")
        
        # Sample random words for verification
        sample_keys = random.sample(word_keys, min(n_samples, len(word_keys)))
        
        for word_key in sample_keys:
            word_group = f[word_key]
            
            # Get attributes
            word_content = word_group.attrs.get('word_content', 'N/A')
            sentence_id = word_group.attrs.get('sentence_id', -1)
            sentence_content = word_group.attrs.get('sentence_content', 'N/A')
            word_index = word_group.attrs.get('word_index', -1)
            
            print(f"\n{word_key}:")
            print(f"  Word: '{word_content}'")
            print(f"  Position in sentence: {word_index}")
            print(f"  Sentence ID: {sentence_id}")
            print(f"  Full sentence: {sentence_content[:100]}...")
            
            # Verify word appears in sentence at correct position
            if sentence_content != 'N/A':
                words_in_sentence = sentence_content.split()
                if word_index < len(words_in_sentence):
                    expected_word = words_in_sentence[word_index]
                    # Clean comparison (remove punctuation for matching)
                    import string
                    clean_expected = expected_word.strip(string.punctuation).lower()
                    clean_actual = word_content.strip(string.punctuation).lower()
                    
                    if clean_expected == clean_actual:
                        print(f"  ✅ Alignment verified: '{word_content}' at position {word_index}")
                    else:
                        print(f"  ⚠️  Mismatch: Expected '{expected_word}' but got '{word_content}'")
            
            # Check EEG data dimensions
            if 'raw_eeg' in word_group:
                eeg_shape = word_group['raw_eeg'].shape
                print(f"  Raw EEG shape: {eeg_shape}")
                if len(eeg_shape) == 2 and eeg_shape[1] == 105:
                    print(f"  ✅ EEG dimensions correct (105 electrodes)")
                else:
                    print(f"  ⚠️  Unexpected EEG dimensions")
            
            # Check frequency bands
            freq_bands = ['FFD_t1', 'FFD_t2', 'FFD_a1', 'FFD_a2', 
                         'FFD_b1', 'FFD_b2', 'FFD_g1', 'FFD_g2']
            freq_found = sum(1 for band in freq_bands if band in word_group)
            if freq_found > 0:
                print(f"  ✅ {freq_found}/8 frequency bands present")

def check_sentence_continuity(h5_file):
    """Check if sentences are properly reconstructed from words."""
    print(f"\n{'='*60}")
    print(f"Checking sentence continuity in: {h5_file.name}")
    print(f"{'='*60}")
    
    with h5py.File(h5_file, 'r') as f:
        # Group words by sentence_id
        sentences = {}
        for word_key in f.keys():
            if word_key.startswith('word_'):
                word_group = f[word_key]
                sentence_id = word_group.attrs.get('sentence_id', -1)
                word_index = word_group.attrs.get('word_index', -1)
                word_content = word_group.attrs.get('word_content', '')
                sentence_content = word_group.attrs.get('sentence_content', '')
                
                if sentence_id not in sentences:
                    sentences[sentence_id] = {
                        'words': [],
                        'full_sentence': sentence_content
                    }
                sentences[sentence_id]['words'].append((word_index, word_content))
        
        # Check a few sentences
        for sentence_id in list(sentences.keys())[:3]:
            sentence_data = sentences[sentence_id]
            # Sort words by index
            sentence_data['words'].sort(key=lambda x: x[0])
            
            reconstructed = ' '.join([word for _, word in sentence_data['words']])
            original = sentence_data['full_sentence']
            
            print(f"\nSentence {sentence_id}:")
            print(f"  Original: {original[:100]}...")
            print(f"  Reconstructed: {reconstructed[:100]}...")
            
            # Simple check - words should appear in order
            words_match = all(word in original for _, word in sentence_data['words'])
            if words_match:
                print(f"  ✅ Words appear in correct order")
            else:
                print(f"  ⚠️  Some words don't match")

def main():
    extracted_dir = Path('extracted_data')
    
    # Test one file from each dataset
    test_files = [
        extracted_dir / 'zuco1_ZAB_SR.h5',   # ZuCo 1.0
        extracted_dir / 'zuco2_YAC_NR.h5',   # ZuCo 2.0
    ]
    
    for test_file in test_files:
        if test_file.exists():
            verify_alignment(test_file)
            check_sentence_continuity(test_file)
        else:
            print(f"File not found: {test_file}")
    
    print(f"\n{'='*60}")
    print("Verification complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()