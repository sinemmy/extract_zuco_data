#!/usr/bin/env python3
"""
Test extraction script - processes just one subject from each dataset to verify pipeline
"""

from zuco_extraction_pipeline import ZuCoExtractor
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_extraction():
    """Test extraction with limited data"""
    
    print("="*60)
    print("TEST EXTRACTION - Processing first subject only")
    print("="*60)
    
    # Create test output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = ZuCoExtractor()
    
    # Test ZuCo 1.0 - just one file
    print("\nTesting ZuCo 1.0 extraction...")
    test_file = extractor.zuco1_path / "task1-SR" / "Matlab files" / "resultsZAB_SR.mat"
    
    if test_file.exists():
        import scipy.io as sio
        data = sio.loadmat(str(test_file), struct_as_record=False, squeeze_me=True)
        
        if 'sentenceData' in data:
            sentences = data['sentenceData']
            print(f"Found {len(sentences)} sentences")
            
            # Process just first 5 sentences
            for i in range(min(5, len(sentences))):
                sentence = sentences[i]
                if hasattr(sentence, 'word'):
                    words = sentence.word
                    if not hasattr(words, '__len__'):
                        words = [words]
                    print(f"  Sentence {i}: {len(words)} words")
                    
                    # Extract first word for testing
                    if len(words) > 0:
                        word = words[0]
                        word_data = extractor.extract_word_zuco1(
                            word, "ZAB", i, 0, "SR", 
                            getattr(sentence, 'content', '')
                        )
                        if word_data:
                            print(f"    Extracted word: '{word_data.word_content}'")
                            print(f"    Has fixation: {word_data.has_fixation}")
                            if word_data.raw_eeg is not None:
                                print(f"    Raw EEG shape: {word_data.raw_eeg.shape}")
                            extractor.extracted_data.append(word_data)
    
    # Test ZuCo 2.0 - just one file
    print("\nTesting ZuCo 2.0 extraction...")
    test_file = extractor.zuco2_path / "task1 - NR" / "Matlab files" / "resultsYAC_NR.mat"
    
    if test_file.exists():
        import h5py
        try:
            with h5py.File(test_file, 'r') as f:
                print(f"HDF5 file keys: {list(f.keys())}")
                
                if 'sentenceData' in f:
                    sent_data = f['sentenceData']
                    print(f"sentenceData type: {type(sent_data)}")
                    
                    if isinstance(sent_data, h5py.Group):
                        # Explore structure
                        keys = list(sent_data.keys())[:10]
                        print(f"First 10 keys in sentenceData: {keys}")
                        
                        # Try to access first item
                        if 'word' in sent_data:
                            word_group = sent_data['word']
                            print(f"  word group found with {len(word_group)} items")
                        
                        # Alternative: sentences might be indexed
                        for key in keys[:3]:
                            if not key.startswith('#'):
                                item = sent_data[key]
                                print(f"  Item '{key}': {type(item)}")
                                if isinstance(item, h5py.Group):
                                    print(f"    Sub-keys: {list(item.keys())[:5]}")
                    
        except Exception as e:
            print(f"Error reading HDF5: {e}")
    
    # Save test results
    if extractor.extracted_data:
        print(f"\nTotal test words extracted: {len(extractor.extracted_data)}")
        summary = extractor.generate_summary()
        print(f"Summary: {summary}")
        
        # Save small test file
        extractor.save_to_hdf5(str(output_dir / "test_extraction.h5"))
        print(f"Test data saved to {output_dir}")
    else:
        print("No test data extracted")

if __name__ == "__main__":
    test_extraction()