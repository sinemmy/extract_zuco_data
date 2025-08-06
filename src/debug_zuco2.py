#!/usr/bin/env python3
"""
Debug script for ZuCo 2.0 HDF5 structure
Explores the actual structure to understand how to extract data
"""

import h5py
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explore_hdf5_structure(file_path, max_depth=3, current_depth=0, prefix=""):
    """Recursively explore HDF5 structure"""
    
    with h5py.File(file_path, 'r') as f:
        print(f"\n{prefix}File: {file_path.name}")
        print(f"{prefix}Root keys: {list(f.keys())[:10]}")
        
        if 'sentenceData' in f:
            sent_data = f['sentenceData']
            print(f"\n{prefix}sentenceData type: {type(sent_data)}")
            print(f"{prefix}sentenceData dtype: {sent_data.dtype if hasattr(sent_data, 'dtype') else 'N/A'}")
            print(f"{prefix}sentenceData shape: {sent_data.shape if hasattr(sent_data, 'shape') else 'N/A'}")
            
            # Try to access the data
            if isinstance(sent_data, h5py.Group):
                print(f"{prefix}sentenceData is a Group")
                print(f"{prefix}Group keys: {list(sent_data.keys())[:20]}")
                
                # Check what's in the group
                for key in list(sent_data.keys())[:5]:
                    item = sent_data[key]
                    print(f"{prefix}  {key}: type={type(item)}, dtype={item.dtype if hasattr(item, 'dtype') else 'N/A'}")
                
            elif isinstance(sent_data, h5py.Dataset):
                print(f"{prefix}sentenceData is a Dataset")
                
                # Check the actual data
                print(f"{prefix}Dataset shape: {sent_data.shape}")
                print(f"{prefix}Dataset dtype: {sent_data.dtype}")
                
                # Try to access first element
                if sent_data.shape[0] > 0:
                    print(f"\n{prefix}Trying to access first sentence...")
                    first_elem = sent_data[0]
                    
                    # Check if it's a reference
                    if isinstance(first_elem, np.ndarray):
                        print(f"{prefix}First element is array with shape: {first_elem.shape}")
                        # For 2D arrays, MATLAB stores column-major
                        if len(first_elem.shape) > 0:
                            first_ref = first_elem[0] if first_elem.shape[0] > 0 else None
                        else:
                            first_ref = first_elem
                    else:
                        first_ref = first_elem
                    
                    print(f"{prefix}First element type: {type(first_ref)}")
                    
                    if isinstance(first_ref, h5py.Reference):
                        print(f"{prefix}It's a reference! Dereferencing...")
                        sent_obj = f[first_ref]
                        print(f"{prefix}Dereferenced type: {type(sent_obj)}")
                        
                        if isinstance(sent_obj, h5py.Group):
                            print(f"{prefix}Sentence is a Group with keys: {list(sent_obj.keys())}")
                            
                            # Check content
                            if 'content' in sent_obj:
                                content_data = sent_obj['content']
                                print(f"{prefix}  content shape: {content_data.shape if hasattr(content_data, 'shape') else 'N/A'}")
                                
                                # Try to get actual content
                                try:
                                    content_val = content_data[()]
                                    if isinstance(content_val, h5py.Reference):
                                        actual_content = f[content_val][()]
                                        if isinstance(actual_content, np.ndarray):
                                            # Try to decode as string
                                            if actual_content.dtype.kind in ['U', 'S']:
                                                print(f"{prefix}  Sentence content: {actual_content[:100]}")
                                            else:
                                                # Try to convert from char codes
                                                try:
                                                    text = ''.join(chr(c) for c in actual_content.flatten() if c > 0)
                                                    print(f"{prefix}  Sentence content: {text[:100]}")
                                                except:
                                                    print(f"{prefix}  Could not decode content")
                                except Exception as e:
                                    print(f"{prefix}  Error getting content: {e}")
                            
                            # Check words
                            if 'word' in sent_obj:
                                word_data = sent_obj['word']
                                print(f"{prefix}  word type: {type(word_data)}")
                                print(f"{prefix}  word shape: {word_data.shape if hasattr(word_data, 'shape') else 'N/A'}")
                                
                                if hasattr(word_data, 'shape'):
                                    print(f"{prefix}  Number of words: {word_data.shape[0]}")
                                    
                                    # Check first word
                                    if word_data.shape[0] > 0:
                                        first_word_ref = word_data[0, 0] if len(word_data.shape) > 1 else word_data[0]
                                        
                                        if isinstance(first_word_ref, h5py.Reference):
                                            print(f"{prefix}  First word is a reference, dereferencing...")
                                            word_obj = f[first_word_ref]
                                            print(f"{prefix}  Word keys: {list(word_obj.keys())[:20]}")
                                            
                                            # Check word content
                                            if 'content' in word_obj:
                                                try:
                                                    word_content_ref = word_obj['content'][()]
                                                    if isinstance(word_content_ref, h5py.Reference):
                                                        word_text = f[word_content_ref][()]
                                                        if isinstance(word_text, np.ndarray):
                                                            decoded = ''.join(chr(c) for c in word_text.flatten() if c > 0)
                                                            print(f"{prefix}    First word content: '{decoded}'")
                                                except:
                                                    pass
                                            
                                            # Check for EEG data
                                            if 'rawEEG' in word_obj:
                                                raw_ref = word_obj['rawEEG'][()]
                                                print(f"{prefix}    Has rawEEG: {type(raw_ref)}")
                                                if isinstance(raw_ref, h5py.Reference):
                                                    raw_data = f[raw_ref][()]
                                                    print(f"{prefix}    Raw EEG shape: {raw_data.shape}")
                                            
                                            # Check for frequency bands
                                            freq_bands = [k for k in word_obj.keys() if any(k.startswith(p) for p in ['FFD_', 'TRT_', 'GD_', 'GPT_'])]
                                            print(f"{prefix}    Frequency bands found: {len(freq_bands)}")
                                            if freq_bands:
                                                print(f"{prefix}    Sample bands: {freq_bands[:5]}")


def main():
    """Debug ZuCo 2.0 extraction"""
    
    base_path = Path("zuco_data/zuco2.0")
    
    # Find a test file
    test_files = list((base_path / "task1 - NR" / "Matlab files").glob("*.mat"))
    
    if not test_files:
        print("No ZuCo 2.0 files found!")
        return
    
    # Test with first available file
    test_file = test_files[0]
    print(f"Testing with: {test_file.name}")
    print("="*60)
    
    explore_hdf5_structure(test_file)
    
    print("\n" + "="*60)
    print("Testing extraction with actual code...")
    print("="*60)
    
    # Now test our extraction function
    from zuco_extraction_pipeline import extract_file_zuco2
    
    subject_id = test_file.stem.replace("results", "").replace("_NR", "")
    word_data = extract_file_zuco2(test_file, subject_id, "NR", show_progress=False)
    
    print(f"\nExtracted {len(word_data)} words")
    
    if word_data:
        print(f"First word: '{word_data[0].word_content}'")
        print(f"Has raw EEG: {word_data[0].raw_eeg is not None}")
        if word_data[0].raw_eeg is not None:
            print(f"Raw EEG shape: {word_data[0].raw_eeg.shape}")


if __name__ == "__main__":
    main()