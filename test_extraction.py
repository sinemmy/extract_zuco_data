#!/usr/bin/env python3
"""
Test script for ZuCo extraction pipeline
Tests extraction of a small sample from both ZuCo 1.0 and 2.0
"""

import logging
from pathlib import Path
import time
import h5py
import json

# Import extraction functions from the main pipeline
from zuco_extraction_pipeline import (
    extract_all,
    get_all_files,
    extract_file_zuco1,
    extract_file_zuco2,
    save_subject_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_test():
    """Quick test with just a few files"""
    logger.info("="*60)
    logger.info("Running Quick Test of ZuCo Extraction Pipeline")
    logger.info("="*60)
    
    # Use test output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Get all files
    all_files = get_all_files()
    logger.info(f"Found {len(all_files)} total files in dataset")
    
    # Test with first file from each version
    test_files = []
    for version in ["zuco1", "zuco2"]:
        version_files = [f for f in all_files if f[0] == version]
        if version_files:
            test_files.append(version_files[0])
            logger.info(f"Selected test file for {version}: {version_files[0][1]}_{version_files[0][2]}")
    
    if not test_files:
        logger.error("No test files found!")
        return
    
    # Process each test file manually
    logger.info(f"\nProcessing {len(test_files)} test files...")
    
    for version, subject_id, task_name, file_path in test_files:
        file_id = f"{version}_{subject_id}_{task_name}"
        logger.info(f"\nTesting {file_id}...")
        
        try:
            start_time = time.time()
            
            # Extract data
            if version == "zuco1":
                word_data_list = extract_file_zuco1(file_path, subject_id, task_name, show_progress=False)
            else:
                word_data_list = extract_file_zuco2(file_path, subject_id, task_name, show_progress=False)
            
            logger.info(f"  Extracted {len(word_data_list)} words")
            
            # Save first 30 words only for quick test
            if word_data_list:
                test_words = word_data_list[:30]
                output_file = output_dir / f"test_{file_id}.h5"
                save_subject_file(test_words, output_file, show_progress=False)
                logger.info(f"  Saved {len(test_words)} words to {output_file.name}")
                
                # Verify the saved file
                with h5py.File(output_file, 'r') as f:
                    logger.info(f"  Verification: File contains {len(f.keys())} word groups")
                    
                    # Check first word
                    if 'word_00000' in f:
                        word = f['word_00000']
                        logger.info(f"    First word: '{word.attrs['word_content']}'")
                        logger.info(f"    Has raw EEG: {'raw_eeg' in word}")
                        logger.info(f"    Has frequency bands: {'FFD_t1' in word}")
            
            elapsed = time.time() - start_time
            logger.info(f"  Processed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"  Error processing {file_id}: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Quick test complete!")


def test_with_pipeline():
    """Test using the main pipeline in test mode"""
    logger.info("="*60)
    logger.info("Testing with Main Pipeline (test_mode=True)")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Run extraction in test mode
    summary = extract_all(
        output_dir="test_output",
        resume=False,  # Don't resume for testing
        test_mode=True,  # Use test mode
        max_files=4  # Limit files
    )
    
    elapsed = time.time() - start_time
    
    # Display summary
    logger.info("\nTest Summary:")
    logger.info(f"  Total files processed: {summary.get('total_files', 0)}")
    logger.info(f"  Total words extracted: {summary.get('total_words', 0)}")
    logger.info(f"  Versions: {summary.get('versions', {})}")
    logger.info(f"  Time elapsed: {elapsed:.1f} seconds")
    
    # Check output files
    output_dir = Path("test_output")
    h5_files = list(output_dir.glob("*.h5"))
    logger.info(f"\nCreated {len(h5_files)} HDF5 files:")
    for h5_file in h5_files[:5]:  # Show first 5
        logger.info(f"  - {h5_file.name}")


def validate_extraction():
    """Validate extracted data quality"""
    logger.info("\n" + "="*60)
    logger.info("Validating Extracted Data")
    logger.info("="*60)
    
    output_dir = Path("test_output")
    
    # Check each HDF5 file
    for h5_file in output_dir.glob("*.h5"):
        if h5_file.name.startswith("test_"):
            continue  # Skip test files
            
        logger.info(f"\nValidating {h5_file.name}...")
        
        with h5py.File(h5_file, 'r') as f:
            n_words = f.attrs.get('n_words', len(f.keys()))
            subject_id = f.attrs.get('subject_id', 'unknown')
            task_name = f.attrs.get('task_name', 'unknown')
            
            logger.info(f"  Subject: {subject_id}, Task: {task_name}")
            logger.info(f"  Total words: {n_words}")
            
            # Check data quality
            has_eeg = 0
            has_bands = 0
            has_fixations = 0
            
            for word_name in list(f.keys())[:100]:  # Sample first 100
                word = f[word_name]
                if 'raw_eeg' in word:
                    has_eeg += 1
                if 'FFD_t1' in word:
                    has_bands += 1
                if word.attrs.get('has_fixation', False):
                    has_fixations += 1
            
            if n_words > 0:
                logger.info(f"  Sample stats (first 100 words):")
                logger.info(f"    With raw EEG: {has_eeg}")
                logger.info(f"    With frequency bands: {has_bands}")
                logger.info(f"    With fixations: {has_fixations}")


def main():
    """Run all tests"""
    
    # Quick test with manual extraction
    quick_test()
    
    # Test with pipeline
    print("\n")
    test_with_pipeline()
    
    # Validate results
    validate_extraction()
    
    logger.info("\n" + "="*60)
    logger.info("All tests complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()