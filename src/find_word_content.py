#!/usr/bin/env python3
"""
Find where word content is stored in ZuCo 2.0
The word objects have no 'content' field, so it must be stored elsewhere
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
    """Find word content in ZuCo 2.0"""
    
    base_path = Path("zuco_data/zuco2.0")
    test_file = list((base_path / "task1 - NR" / "Matlab files").glob("*.mat"))[0]
    
    print(f"Searching for word content in: {test_file.name}")
    print("="*60)
    
    with h5py.File(test_file, 'r') as f:
        if 'sentenceData' in f:
            sent_data = f['sentenceData']
            
            # Get the sentence content for reference
            content_data = sent_data['content']
            first_sent_ref = content_data[0, 0] if len(content_data.shape) > 1 else content_data[0]
            first_sentence = decode_string(f, first_sent_ref)
            print(f"First sentence: {first_sentence}\n")
            
            # Split into words for comparison
            expected_words = first_sentence.split()
            print(f"Expected words: {expected_words[:10]}...\n")
            
            # Check the word objects
            word_data = sent_data['word']
            first_word_ref = word_data[0, 0] if len(word_data.shape) > 1 else word_data[0]
            first_word_obj = f[first_word_ref]
            
            print("Keys in first word object:")
            word_keys = list(first_word_obj.keys())
            for key in sorted(word_keys):
                if 'content' in key.lower() or 'text' in key.lower() or 'word' in key.lower():
                    print(f"  *** {key} ***")
                elif not any(x in key for x in ['FFD', 'TRT', 'GD', 'GPT', 'SFD', '_a', '_b', '_g', '_t', 'diff', 'pupil']):
                    print(f"  {key}")
            
            # Check if there's a 'content' field at all
            if 'content' in first_word_obj:
                print("\nFound 'content' field!")
                content_ref = first_word_obj['content'][()]
                word_text = decode_string(f, content_ref)
                print(f"Word content: '{word_text}'")
            
            # Check wordbounds - maybe it contains word positions
            print("\n" + "="*60)
            print("Checking wordbounds...")
            if 'wordbounds' in sent_data:
                bounds_data = sent_data['wordbounds']
                print(f"wordbounds shape: {bounds_data.shape}")
                
                # Check first few wordbounds
                for i in range(min(5, bounds_data.shape[0])):
                    bound_ref = bounds_data[i, 0] if len(bounds_data.shape) > 1 else bounds_data[i]
                    if isinstance(bound_ref, h5py.Reference):
                        bound_obj = f[bound_ref]
                        if isinstance(bound_obj, h5py.Dataset):
                            bound_val = bound_obj[()]
                            print(f"  Wordbound {i}: {bound_val.shape} - {bound_val[:10] if bound_val.size > 10 else bound_val}")
                        elif isinstance(bound_obj, h5py.Group):
                            print(f"  Wordbound {i} is a Group with keys: {list(bound_obj.keys())}")
            
            # Maybe the word content is derived from sentence + wordbounds?
            print("\n" + "="*60)
            print("Hypothesis: Words might be extracted using sentence + wordbounds")
            print("Let me check if the number of words matches expected...")
            
            # Count words in first sentence
            n_words_expected = len(expected_words)
            
            # Count word objects for first sentence
            # This is tricky - how do we know which words belong to which sentence?
            print(f"\nTotal word objects: {word_data.shape[0]}")
            print(f"Total sentences: {content_data.shape[0]}")
            print(f"Average words per sentence: {word_data.shape[0] / content_data.shape[0]:.1f}")
            
            # Check preprocessed folder for word content
            print("\n" + "="*60)
            print("Checking preprocessed folder for word data...")
            
            subject_id = test_file.stem.replace("results", "").replace("_NR", "")
            preprocessed_path = base_path / "preprocessed" / subject_id
            
            if preprocessed_path.exists():
                print(f"Found preprocessed folder for {subject_id}")
                for file in preprocessed_path.glob("*.mat"):
                    print(f"  - {file.name}")
                
                # Check wordbounds file specifically
                wordbounds_file = preprocessed_path / f"wordbounds_{subject_id}.mat"
                if wordbounds_file.exists():
                    print(f"\nFound wordbounds file! Checking structure...")
                    try:
                        # Try HDF5 first
                        with h5py.File(wordbounds_file, 'r') as wb:
                            print(f"  HDF5 keys: {list(wb.keys())}")
                    except:
                        # Try scipy
                        import scipy.io as sio
                        try:
                            wb_data = sio.loadmat(str(wordbounds_file))
                            print(f"  MATLAB v7 keys: {[k for k in wb_data.keys() if not k.startswith('__')]}")
                            for key in wb_data.keys():
                                if not key.startswith('__'):
                                    print(f"    {key}: {type(wb_data[key])}, shape={wb_data[key].shape if hasattr(wb_data[key], 'shape') else 'N/A'}")
                        except Exception as e:
                            print(f"  Could not load wordbounds: {e}")
            else:
                print(f"No preprocessed folder found at {preprocessed_path}")


if __name__ == "__main__":
    main()