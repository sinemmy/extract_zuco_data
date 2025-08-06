#!/usr/bin/env python3
"""
Extract word data from ZuCo 2.0
Found 'word' and 'wordbounds' in sentenceData!
"""

import h5py
import numpy as np
from pathlib import Path


def decode_string(h5file, ref_or_array):
    """Decode a string from HDF5 reference or array"""
    try:
        if isinstance(ref_or_array, h5py.Reference):
            data = h5file[ref_or_array][()]
        else:
            data = ref_or_array
        
        if isinstance(data, np.ndarray):
            # Try to decode as UTF-16 (uint16) or UTF-8
            if data.dtype == np.uint16:
                # UTF-16 encoded
                text = ''.join(chr(c) for c in data.flatten() if c > 0)
            else:
                # Assume character codes
                text = ''.join(chr(c) for c in data.flatten() if c > 0 and c < 256)
            return text
        return str(data)
    except:
        return ""


def main():
    """Extract word data from ZuCo 2.0"""
    
    base_path = Path("zuco_data/zuco2.0")
    test_file = list((base_path / "task1 - NR" / "Matlab files").glob("*.mat"))[0]
    
    print(f"Extracting from: {test_file.name}")
    print("="*60)
    
    with h5py.File(test_file, 'r') as f:
        if 'sentenceData' in f:
            sent_data = f['sentenceData']
            
            # Check the 'word' data
            if 'word' in sent_data:
                word_data = sent_data['word']
                print(f"word dataset: shape={word_data.shape}, dtype={word_data.dtype}")
                
                # Check wordbounds
                if 'wordbounds' in sent_data:
                    bounds_data = sent_data['wordbounds']
                    print(f"wordbounds dataset: shape={bounds_data.shape}, dtype={bounds_data.dtype}")
                
                # Extract some words
                print("\nExtracting words...")
                print("-"*40)
                
                n_items = min(10, word_data.shape[0])
                
                for i in range(n_items):
                    # Get word reference
                    word_ref = word_data[i, 0] if len(word_data.shape) > 1 else word_data[i]
                    
                    if isinstance(word_ref, h5py.Reference):
                        word_obj = f[word_ref]
                        
                        # Check if it's a group or dataset
                        if isinstance(word_obj, h5py.Group):
                            print(f"\nWord {i}: Group with keys: {list(word_obj.keys())[:20]}")
                            
                            # Try to get word content
                            if 'content' in word_obj:
                                content = word_obj['content'][()]
                                word_text = decode_string(f, content)
                                print(f"  Content: '{word_text}'")
                            
                            # Check for EEG data
                            for key in ['rawEEG', 'raw_eeg', 'rawData']:
                                if key in word_obj:
                                    eeg = word_obj[key]
                                    if isinstance(eeg, h5py.Dataset):
                                        print(f"  {key}: shape={eeg.shape}")
                                    break
                            
                            # Check for frequency bands
                            freq_keys = [k for k in word_obj.keys() if any(k.startswith(p) for p in ['FFD', 'TRT', 'GD', 'GPT', 't1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2'])]
                            if freq_keys:
                                print(f"  Frequency bands: {len(freq_keys)} found")
                                print(f"    Sample: {freq_keys[:5]}")
                            
                            # Check for fixation data
                            for key in ['nFixations', 'n_fixations', 'fixations']:
                                if key in word_obj:
                                    print(f"  Has {key}")
                                    break
                        
                        elif isinstance(word_obj, h5py.Dataset):
                            print(f"\nWord {i}: Dataset shape={word_obj.shape}, dtype={word_obj.dtype}")
                            
                            # Try to decode if it's text
                            if word_obj.dtype.kind in ['U', 'S', 'O']:
                                try:
                                    word_text = decode_string(f, word_obj[()])
                                    print(f"  Content: '{word_text}'")
                                except:
                                    pass
                
                # Check content array to understand sentence structure
                print("\n" + "="*60)
                print("Checking sentence content...")
                content_data = sent_data['content']
                
                for i in range(min(3, content_data.shape[0])):
                    content_ref = content_data[i, 0] if len(content_data.shape) > 1 else content_data[i]
                    sentence_text = decode_string(f, content_ref)
                    print(f"\nSentence {i}: {sentence_text[:100]}...")
                    
                # Check if words align with sentences
                print("\n" + "="*60)
                print("Data structure summary:")
                print(f"- {word_data.shape[0]} word entries")
                print(f"- {content_data.shape[0]} sentence entries")
                print("- Appears to be word-level data after all!")


if __name__ == "__main__":
    main()