#!/usr/bin/env python
"""Minimal test script to verify HDF5 files work in Python 3.13."""

import sys
print(f"Python version: {sys.version}")

try:
    import h5py
    print(f"h5py version: {h5py.__version__}")
    
    # Test reading a file
    test_file = 'extracted_data/zuco1_ZAB_SR.h5'
    
    with h5py.File(test_file, 'r') as f:
        # Basic operations
        print(f"\n✅ File opened: {test_file}")
        word_count = len([k for k in f.keys() if k.startswith('word_')])
        print(f"✅ Word count: {word_count}")
        
        # Read attributes
        word = f['word_00000']
        print(f"✅ Word content: {word.attrs['word_content']}")
        
        # Read arrays
        if 'raw_eeg' in word:
            eeg = word['raw_eeg'][:]
            print(f"✅ EEG shape: {eeg.shape}")
        
        print("\n🎉 All operations successful!")
        
except ImportError:
    print("❌ h5py not installed - run: pip install h5py")
except Exception as e:
    print(f"❌ Error: {e}")