#!/usr/bin/env python3
"""
Deep exploration of ZuCo 2.0 structure
Look for word-level data in the HDF5 file
"""

import h5py
import numpy as np
from pathlib import Path


def explore_reference(h5file, ref, prefix="  "):
    """Explore an HDF5 reference"""
    try:
        obj = h5file[ref]
        if isinstance(obj, h5py.Group):
            return f"Group with keys: {list(obj.keys())[:10]}"
        elif isinstance(obj, h5py.Dataset):
            return f"Dataset with shape: {obj.shape}, dtype: {obj.dtype}"
        else:
            return f"Unknown type: {type(obj)}"
    except:
        return "Could not dereference"


def explore_dataset(h5file, dataset, name, max_items=3):
    """Explore a dataset in detail"""
    print(f"\n  Dataset: {name}")
    print(f"    Shape: {dataset.shape}")
    print(f"    Dtype: {dataset.dtype}")
    
    if dataset.shape[0] > 0:
        # Check first few items
        for i in range(min(max_items, dataset.shape[0])):
            try:
                if len(dataset.shape) == 1:
                    item = dataset[i]
                else:
                    item = dataset[i, 0]  # MATLAB column-major
                
                if isinstance(item, h5py.Reference):
                    print(f"    Item {i}: Reference -> {explore_reference(h5file, item)}")
                elif isinstance(item, np.ndarray):
                    print(f"    Item {i}: Array with shape {item.shape}")
                else:
                    print(f"    Item {i}: {type(item)} = {item}")
            except Exception as e:
                print(f"    Item {i}: Error - {e}")


def find_word_data(h5file, group, prefix="", max_depth=3, current_depth=0):
    """Recursively search for word-level data"""
    if current_depth >= max_depth:
        return
    
    for key in group.keys():
        item = group[key]
        print(f"{prefix}{key}: {type(item)}")
        
        # Look for anything with 'word' in the name
        if 'word' in key.lower():
            print(f"{prefix}  *** FOUND WORD-RELATED DATA ***")
            if isinstance(item, h5py.Dataset):
                explore_dataset(h5file, item, key)
            elif isinstance(item, h5py.Group):
                print(f"{prefix}  Word group with keys: {list(item.keys())[:20]}")
        
        # Recurse into groups
        if isinstance(item, h5py.Group) and current_depth < max_depth - 1:
            find_word_data(h5file, item, prefix + "  ", max_depth, current_depth + 1)


def main():
    """Explore ZuCo 2.0 structure deeply"""
    
    base_path = Path("zuco_data/zuco2.0")
    test_file = list((base_path / "task1 - NR" / "Matlab files").glob("*.mat"))[0]
    
    print(f"Exploring: {test_file.name}")
    print("="*60)
    
    with h5py.File(test_file, 'r') as f:
        print(f"Root keys: {list(f.keys())}")
        
        # Look everywhere for word data
        print("\nSearching for word-level data...")
        print("-"*40)
        find_word_data(f, f, max_depth=4)
        
        # Check if there's a different structure
        print("\n" + "="*60)
        print("Checking sentenceData structure in detail...")
        
        if 'sentenceData' in f:
            sent_group = f['sentenceData']
            
            # List all items in sentenceData
            print(f"\nAll keys in sentenceData ({len(sent_group.keys())} total):")
            for key in sorted(sent_group.keys()):
                item = sent_group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  {key}: Dataset {item.shape} {item.dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"  {key}: Group with {len(item.keys())} items")
            
            # Check specific datasets
            if 'content' in sent_group:
                print("\nExploring 'content' dataset...")
                explore_dataset(f, sent_group['content'], 'content', max_items=5)
            
            if 'rawData' in sent_group:
                print("\nExploring 'rawData' dataset...")
                explore_dataset(f, sent_group['rawData'], 'rawData', max_items=3)
            
            # Check for any nested groups
            for key in sent_group.keys():
                item = sent_group[key]
                if isinstance(item, h5py.Group):
                    print(f"\nFound nested group: {key}")
                    print(f"  Keys: {list(item.keys())[:20]}")
                    
                    # Check for word data in nested groups
                    if 'word' in item:
                        print(f"  *** HAS WORD DATA ***")
                        word_item = item['word']
                        print(f"  Word type: {type(word_item)}")
                        if isinstance(word_item, h5py.Dataset):
                            explore_dataset(f, word_item, 'word', max_items=5)


if __name__ == "__main__":
    main()