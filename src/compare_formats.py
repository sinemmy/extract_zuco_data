#!/usr/bin/env python
"""Compare HDF5 vs Parquet for EEG data storage."""

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
from pathlib import Path

def test_current_h5_structure():
    """Show current HDF5 structure and challenges for Parquet."""
    print("="*60)
    print("CURRENT HDF5 STRUCTURE")
    print("="*60)
    
    with h5py.File('extracted_data/zuco1_ZAB_SR.h5', 'r') as f:
        word = f['word_00000']
        
        print("\nAttributes (metadata):")
        for key, val in word.attrs.items():
            if isinstance(val, str) and len(val) > 50:
                print(f"  {key}: {type(val).__name__} - '{val[:50]}...'")
            else:
                print(f"  {key}: {type(val).__name__} - {val}")
        
        print("\nDatasets (arrays):")
        for key in word.keys():
            if isinstance(word[key], h5py.Dataset):
                shape = word[key].shape
                dtype = word[key].dtype
                print(f"  {key}: shape={shape}, dtype={dtype}")
                
                # Show the challenge for Parquet
                if len(shape) > 1:
                    print(f"    ⚠️  2D array - Parquet challenge!")

def create_parquet_version():
    """Create a Parquet version showing the challenges."""
    print("\n" + "="*60)
    print("PARQUET VERSION - CHALLENGES")
    print("="*60)
    
    # Read some data from HDF5
    with h5py.File('extracted_data/zuco1_ZAB_SR.h5', 'r') as f:
        data_rows = []
        
        # Just get first 10 words for demo
        for i in range(10):
            word_key = f'word_{i:05d}'
            if word_key not in f:
                continue
                
            word = f[word_key]
            
            # Problem 1: 2D arrays need flattening or separate storage
            if 'raw_eeg' in word:
                raw_eeg = word['raw_eeg'][:]  # Shape: (n_fixations, 105)
            else:
                # Use frequency band shape as proxy
                raw_eeg = np.zeros((1, 105))  # Placeholder
            
            # Problem 2: Variable-length arrays (different n_fixations per word)
            print(f"\n{word_key}: raw_eeg shape = {raw_eeg.shape}")
            print(f"  Problem: Variable fixations ({raw_eeg.shape[0]}) × 105 electrodes")
            
            # Option 1: Flatten to 1D (loses structure)
            row = {
                'word_id': word_key,
                'word_content': word.attrs['word_content'],
                'sentence_id': word.attrs['sentence_id'],
                'sentence_content': word.attrs['sentence_content'],
                'word_index': word.attrs['word_index'],
                'n_fixations': raw_eeg.shape[0],
                # Flatten EEG - but now we lose the fixation structure!
                'raw_eeg_flat': raw_eeg.flatten().tolist(),  # Variable length!
            }
            
            # Add frequency bands (these are already 1D, easier)
            for band in ['FFD_t1', 'FFD_t2', 'FFD_a1', 'FFD_a2']:
                if band in word:
                    # These are 105-length arrays, manageable
                    row[band] = word[band][:].tolist()
            
            data_rows.append(row)
    
    # Try to create DataFrame - this will show the problems
    print("\n" + "-"*60)
    print("ATTEMPTING PARQUET CONVERSION...")
    print("-"*60)
    
    try:
        df = pd.DataFrame(data_rows)
        print("\n✅ DataFrame created, but notice the issues:")
        print(df.info())
        
        # Problem: Lists in columns
        print("\n⚠️  Issue 1: 'raw_eeg_flat' is stored as Python list (inefficient)")
        print(f"  Type: {type(df['raw_eeg_flat'].iloc[0])}")
        
        # Save to Parquet
        df.to_parquet('test_output/test_format.parquet')
        print("\n✅ Parquet saved, but with compromises")
        
        # Read back
        df_read = pd.read_parquet('test_output/test_format.parquet')
        print("\n⚠️  Issue 2: Arrays converted to lists lose NumPy efficiency")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def show_better_alternatives():
    """Show why HDF5 is actually better for this use case."""
    print("\n" + "="*60)
    print("FORMAT COMPARISON FOR EEG DATA")
    print("="*60)
    
    comparison = """
    | Feature                | HDF5                  | Parquet              | Zarr                 |
    |------------------------|----------------------|---------------------|---------------------|
    | Variable-length arrays | ✅ Native support     | ❌ Must flatten      | ✅ Native support    |
    | 2D/3D arrays          | ✅ Native support     | ❌ Must flatten      | ✅ Native support    |
    | Hierarchical structure | ✅ Groups/datasets    | ❌ Flat table only   | ✅ Groups/arrays     |
    | Compression           | ✅ Multiple options   | ✅ Good compression  | ✅ Multiple options  |
    | Python 3.13 support   | ✅ Via h5py           | ✅ Via pyarrow       | ✅ Via zarr          |
    | Cross-language        | ✅ C/MATLAB/Python    | ✅ Many languages    | ⚠️  Mainly Python    |
    | Cloud-optimized       | ⚠️  Needs kerchunk    | ✅ Native support    | ✅ Native support    |
    | Partial reads         | ✅ Chunks             | ✅ Column selection  | ✅ Chunks            |
    | Scientific standard   | ✅ Widely used        | ❌ Business/analytics| ⚠️  Growing          |
    
    VERDICT FOR EEG DATA:
    -----------------------
    🏆 HDF5: Best for your current use case
       - Handles variable-length 2D arrays naturally
       - Preserves data structure without flattening
       - Standard in neuroscience/scientific computing
       - MATLAB compatible (important for ZuCo)
    
    🥈 Zarr: Good alternative if you want cloud-native
       - Similar to HDF5 but cloud-optimized
       - Would require restructuring code
    
    ❌ Parquet: Not ideal for this data
       - Would require flattening 2D arrays
       - Loses natural structure of EEG data
       - Better for tabular business data
    """
    print(comparison)

def show_h5py_compatibility():
    """Show h5py's Python version support."""
    print("\n" + "="*60)
    print("H5PY PYTHON COMPATIBILITY")
    print("="*60)
    
    info = """
    h5py Release    | Python Versions Supported
    ----------------|---------------------------
    h5py 3.11       | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 ✅
    h5py 3.10       | 3.8, 3.9, 3.10, 3.11, 3.12
    h5py 3.9        | 3.8, 3.9, 3.10, 3.11
    
    As of January 2025:
    - h5py actively maintains Python 3.13 support
    - HDF5 file format is stable across all versions
    - Files created in Python 3.8 will work in 3.13
    
    To ensure compatibility:
    1. pip install h5py (gets latest version)
    2. Use standard HDF5 features (which you are)
    3. Avoid deprecated NumPy features
    """
    print(info)

if __name__ == '__main__':
    # Create test output directory
    Path('test_output').mkdir(exist_ok=True)
    
    # Run comparisons
    test_current_h5_structure()
    create_parquet_version()
    show_better_alternatives()
    show_h5py_compatibility()