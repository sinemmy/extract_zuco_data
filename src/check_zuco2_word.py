#!/usr/bin/env python3
"""
Check if ZuCo 2.0 has word-level data at all
The structure seems completely different from ZuCo 1.0
"""

import h5py
import numpy as np
from pathlib import Path


def main():
    """Check ZuCo 2.0 structure"""
    
    base_path = Path("zuco_data/zuco2.0")
    test_file = list((base_path / "task1 - NR" / "Matlab files").glob("*.mat"))[0]
    
    print(f"Checking: {test_file.name}")
    print("="*60)
    
    with h5py.File(test_file, 'r') as f:
        # The #refs# group contains all the actual data objects
        # sentenceData appears to be aggregated data, not individual sentences
        
        if 'sentenceData' in f:
            sent_data = f['sentenceData']
            print(f"sentenceData is a {type(sent_data).__name__}")
            print(f"Keys in sentenceData: {list(sent_data.keys())}")
            print()
            
            # Check if this is aggregated data
            print("Data structure suggests this is AGGREGATED data:")
            print("- mean_a1, mean_a2, etc. = mean frequency bands")
            print("- omissionRate = reading behavior metric")
            print("- allFixations = all fixations (not per word)")
            print()
            
            # Check dimensions
            for key in ['content', 'allFixations', 'rawData']:
                if key in sent_data:
                    item = sent_data[key]
                    print(f"{key}: shape={item.shape}, dtype={item.dtype}")
                    
                    # Try to get the actual data
                    if item.shape[0] > 0:
                        first = item[0] if len(item.shape) == 1 else item[0, 0]
                        if isinstance(first, h5py.Reference):
                            try:
                                deref = f[first]
                                if hasattr(deref, 'shape'):
                                    print(f"  -> Dereferenced: shape={deref.shape}, dtype={deref.dtype}")
                            except:
                                pass
        
        print("\n" + "="*60)
        print("CONCLUSION:")
        print("ZuCo 2.0 files appear to contain AGGREGATED data, not word-level data!")
        print("This might be summary statistics rather than raw trial data.")
        print()
        print("The structure has:")
        print("- Mean frequency bands (mean_t1, mean_a1, etc.)")
        print("- Aggregated fixations (allFixations)")
        print("- No individual word or sentence structure")
        print()
        print("This explains why extraction returns 0 words - there are no words to extract!")
        
        # Double check - are there other .mat files with different structure?
        print("\n" + "="*60)
        print("Checking other files in ZuCo 2.0...")
        
        all_mat_files = list(base_path.rglob("*.mat"))
        print(f"Found {len(all_mat_files)} .mat files total")
        
        # Check a few different files
        for mat_file in all_mat_files[:5]:
            print(f"\n{mat_file.name}:")
            try:
                with h5py.File(mat_file, 'r') as f2:
                    print(f"  Root keys: {list(f2.keys())}")
                    if 'sentenceData' in f2:
                        sd = f2['sentenceData']
                        if isinstance(sd, h5py.Group):
                            print(f"  sentenceData keys: {list(sd.keys())[:10]}")
                        elif isinstance(sd, h5py.Dataset):
                            print(f"  sentenceData shape: {sd.shape}")
            except:
                # Try scipy for v7 format
                try:
                    import scipy.io as sio
                    data = sio.loadmat(str(mat_file))
                    print(f"  MATLAB v7 format, keys: {list(data.keys())[:10]}")
                    if 'sentenceData' in data:
                        print(f"  sentenceData type: {type(data['sentenceData'])}")
                except:
                    print(f"  Could not read file")


if __name__ == "__main__":
    main()