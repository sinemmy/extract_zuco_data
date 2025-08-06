#!/usr/bin/env python3
"""
ZuCo Dataset Unified Extraction Pipeline
Extracts word-level EEG data from both ZuCo 1.0 and 2.0 datasets
Handles both MATLAB v7 (scipy.io) and v7.3 (HDF5) formats
Outputs to modern Python formats (HDF5/Parquet)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WordData:
    """Structure for word-level data"""
    # Identifiers
    subject_id: str
    sentence_id: int
    word_index: int
    task_name: str
    
    # Text content
    word_content: str
    sentence_content: str
    
    # EEG data - raw
    raw_eeg: Optional[np.ndarray] = None  # Shape: (n_fixations, 105)
    
    # EEG frequency bands (105 electrodes each)
    # First fixation duration (FFD)
    FFD_t1: Optional[np.ndarray] = None
    FFD_t2: Optional[np.ndarray] = None
    FFD_a1: Optional[np.ndarray] = None
    FFD_a2: Optional[np.ndarray] = None
    FFD_b1: Optional[np.ndarray] = None
    FFD_b2: Optional[np.ndarray] = None
    FFD_g1: Optional[np.ndarray] = None
    FFD_g2: Optional[np.ndarray] = None
    
    # Total reading time (TRT)
    TRT_t1: Optional[np.ndarray] = None
    TRT_t2: Optional[np.ndarray] = None
    TRT_a1: Optional[np.ndarray] = None
    TRT_a2: Optional[np.ndarray] = None
    TRT_b1: Optional[np.ndarray] = None
    TRT_b2: Optional[np.ndarray] = None
    TRT_g1: Optional[np.ndarray] = None
    TRT_g2: Optional[np.ndarray] = None
    
    # Gaze duration (GD)
    GD_t1: Optional[np.ndarray] = None
    GD_t2: Optional[np.ndarray] = None
    GD_a1: Optional[np.ndarray] = None
    GD_a2: Optional[np.ndarray] = None
    GD_b1: Optional[np.ndarray] = None
    GD_b2: Optional[np.ndarray] = None
    GD_g1: Optional[np.ndarray] = None
    GD_g2: Optional[np.ndarray] = None
    
    # Go-past time (GPT)
    GPT_t1: Optional[np.ndarray] = None
    GPT_t2: Optional[np.ndarray] = None
    GPT_a1: Optional[np.ndarray] = None
    GPT_a2: Optional[np.ndarray] = None
    GPT_b1: Optional[np.ndarray] = None
    GPT_b2: Optional[np.ndarray] = None
    GPT_g1: Optional[np.ndarray] = None
    GPT_g2: Optional[np.ndarray] = None
    
    # Eye-tracking metrics
    n_fixations: Optional[int] = None
    FFD_duration: Optional[float] = None
    TRT_duration: Optional[float] = None
    GD_duration: Optional[float] = None
    GPT_duration: Optional[float] = None
    SFD_duration: Optional[float] = None  # Single fixation duration
    
    # Data quality flags
    has_fixation: bool = False
    is_missing: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling numpy arrays"""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            else:
                data[key] = value
        return data


class ZuCoExtractor:
    """Unified extractor for both ZuCo 1.0 and 2.0"""
    
    def __init__(self, base_path: str = "zuco_data"):
        self.base_path = Path(base_path)
        self.zuco1_path = self.base_path / "zuco1.0"
        self.zuco2_path = self.base_path / "zuco2.0"
        self.extracted_data = []
        
    def extract_all(self) -> List[WordData]:
        """Extract data from both ZuCo versions"""
        logger.info("Starting extraction from both ZuCo datasets...")
        
        # Extract ZuCo 1.0
        if self.zuco1_path.exists():
            logger.info("Extracting ZuCo 1.0...")
            self.extract_zuco1()
        else:
            logger.warning(f"ZuCo 1.0 path not found: {self.zuco1_path}")
        
        # Extract ZuCo 2.0
        if self.zuco2_path.exists():
            logger.info("Extracting ZuCo 2.0...")
            self.extract_zuco2()
        else:
            logger.warning(f"ZuCo 2.0 path not found: {self.zuco2_path}")
        
        logger.info(f"Extraction complete. Total words: {len(self.extracted_data)}")
        return self.extracted_data
    
    def extract_zuco1(self):
        """Extract data from ZuCo 1.0 (MATLAB v7 format)"""
        
        # Task definitions
        tasks = [
            ("task1-SR", "SR", "Matlab files"),  # Sentiment Reading
            ("task2-NR", "NR", "Matlab files"),  # Normal Reading
            ("task3-TSR", "TSR", "Matlab files")  # Task-Specific Reading
        ]
        
        for task_dir, task_name, matlab_subdir in tasks:
            task_path = self.zuco1_path / task_dir / matlab_subdir
            
            if not task_path.exists():
                logger.warning(f"Task path not found: {task_path}")
                continue
            
            logger.info(f"Processing ZuCo 1.0 {task_name}...")
            
            # Process each subject file
            for mat_file in sorted(task_path.glob("*.mat")):
                subject_id = mat_file.stem.replace("results", "").replace(f"_{task_name}", "")
                logger.info(f"  Processing subject {subject_id}...")
                
                try:
                    # Load MATLAB file
                    data = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
                    
                    if 'sentenceData' not in data:
                        logger.warning(f"    No sentenceData in {mat_file.name}")
                        continue
                    
                    sentences = data['sentenceData']
                    if not hasattr(sentences, '__len__'):
                        sentences = [sentences]
                    
                    # Process each sentence
                    for sent_idx, sentence in enumerate(tqdm(sentences, desc=f"    {subject_id}", leave=False)):
                        if not hasattr(sentence, 'word'):
                            continue
                        
                        sentence_content = getattr(sentence, 'content', '')
                        words = sentence.word
                        
                        if not hasattr(words, '__len__'):
                            words = [words]
                        
                        # Process each word
                        for word_idx, word in enumerate(words):
                            word_data = self.extract_word_zuco1(
                                word, 
                                subject_id, 
                                sent_idx, 
                                word_idx,
                                task_name,
                                sentence_content
                            )
                            if word_data:
                                self.extracted_data.append(word_data)
                
                except Exception as e:
                    logger.error(f"    Error processing {mat_file.name}: {e}")
    
    def extract_word_zuco1(self, word, subject_id: str, sent_idx: int, 
                          word_idx: int, task_name: str, sentence_content: str) -> Optional[WordData]:
        """Extract word-level data from ZuCo 1.0 word struct"""
        
        try:
            word_data = WordData(
                subject_id=subject_id,
                sentence_id=sent_idx,
                word_index=word_idx,
                task_name=task_name,
                word_content=getattr(word, 'content', ''),
                sentence_content=sentence_content
            )
            
            # Extract raw EEG if available
            if hasattr(word, 'rawEEG'):
                raw_eeg = word.rawEEG
                if raw_eeg is not None:
                    # Convert to numpy array and handle different formats
                    if not isinstance(raw_eeg, np.ndarray):
                        try:
                            raw_eeg = np.array(raw_eeg)
                        except:
                            raw_eeg = None
                    
                    if raw_eeg is not None and raw_eeg.size > 0:
                        # Handle object arrays (MATLAB cell arrays)
                        if raw_eeg.dtype == object:
                            # Try to extract numeric data from object array
                            try:
                                # Flatten and concatenate if it's a cell array of arrays
                                if raw_eeg.ndim == 1:
                                    raw_eeg = np.concatenate([np.array(x).flatten() for x in raw_eeg if x is not None])
                                else:
                                    raw_eeg = np.array(raw_eeg.tolist(), dtype=float)
                            except:
                                raw_eeg = None
                        
                        if raw_eeg is not None and raw_eeg.size > 0:
                            # Ensure correct shape (n_fixations, 105)
                            if raw_eeg.ndim == 1:
                                # Try to reshape to (n, 105)
                                if len(raw_eeg) % 105 == 0:
                                    raw_eeg = raw_eeg.reshape(-1, 105)
                                else:
                                    raw_eeg = raw_eeg.reshape(1, -1)
                            elif raw_eeg.ndim == 2 and raw_eeg.shape[1] != 105 and raw_eeg.shape[0] == 105:
                                raw_eeg = raw_eeg.T
                            
                            # Only keep if we have valid EEG data
                            if raw_eeg.ndim == 2:
                                word_data.raw_eeg = raw_eeg.astype(np.float32)
                                word_data.has_fixation = True
            
            # Extract frequency band features
            freq_bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
            event_types = ['FFD', 'TRT', 'GD', 'GPT']
            
            for event in event_types:
                for band in freq_bands:
                    attr_name = f"{event}_{band}"
                    if hasattr(word, attr_name):
                        value = getattr(word, attr_name)
                        if value is not None and hasattr(value, '__len__'):
                            # Convert to numpy array and ensure it's 1D with 105 elements
                            value = np.asarray(value).flatten()
                            if len(value) == 105:
                                setattr(word_data, attr_name, value)
                                word_data.has_fixation = True
            
            # Extract eye-tracking metrics
            if hasattr(word, 'nFixations'):
                word_data.n_fixations = int(word.nFixations) if word.nFixations else 0
            
            for metric in ['FFD', 'TRT', 'GD', 'GPT', 'SFD']:
                if hasattr(word, metric):
                    value = getattr(word, metric)
                    if value is not None and not isinstance(value, (list, np.ndarray)):
                        setattr(word_data, f"{metric}_duration", float(value))
            
            return word_data
            
        except Exception as e:
            logger.debug(f"Error extracting word data: {e}")
            return None
    
    def extract_zuco2(self):
        """Extract data from ZuCo 2.0 (HDF5/MATLAB v7.3 format)"""
        
        # Task definitions for ZuCo 2.0
        tasks = [
            ("task1 - NR", "NR", "Matlab files"),  # Normal Reading
            ("task2 - TSR", "TSR", "Matlab files")  # Task-Specific Reading
        ]
        
        for task_dir, task_name, matlab_subdir in tasks:
            task_path = self.zuco2_path / task_dir / matlab_subdir
            
            if not task_path.exists():
                logger.warning(f"Task path not found: {task_path}")
                continue
            
            logger.info(f"Processing ZuCo 2.0 {task_name}...")
            
            # Process each subject file
            for mat_file in sorted(task_path.glob("*.mat")):
                subject_id = mat_file.stem.replace("results", "").replace(f"_{task_name}", "")
                logger.info(f"  Processing subject {subject_id}...")
                
                try:
                    # Try loading as HDF5
                    with h5py.File(mat_file, 'r') as f:
                        if 'sentenceData' in f:
                            self.extract_sentences_zuco2_hdf5(
                                f['sentenceData'], 
                                subject_id, 
                                task_name
                            )
                        else:
                            logger.warning(f"    No sentenceData in {mat_file.name}")
                
                except OSError:
                    # If HDF5 fails, try scipy.io.loadmat (some ZuCo 2.0 files might be v7)
                    try:
                        data = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
                        if 'sentenceData' in data:
                            # Process similar to ZuCo 1.0
                            self.extract_sentences_zuco2_mat(
                                data['sentenceData'],
                                subject_id,
                                task_name
                            )
                    except Exception as e:
                        logger.error(f"    Error loading {mat_file.name}: {e}")
    
    def extract_sentences_zuco2_hdf5(self, sent_group: h5py.Group, 
                                     subject_id: str, task_name: str):
        """Extract sentences from ZuCo 2.0 HDF5 format"""
        
        # In HDF5 format, sentenceData is typically a group containing datasets
        # Structure varies, so we need to explore it
        
        # Check if it has a 'word' subgroup or dataset
        if 'word' in sent_group:
            word_data = sent_group['word']
            
            # Get sentence content if available
            sentence_content = ''
            if 'content' in sent_group:
                content_data = sent_group['content']
                if isinstance(content_data, h5py.Dataset):
                    # Try to decode the content
                    try:
                        content_raw = content_data[()]
                        if content_raw.dtype.kind == 'U':  # Unicode string
                            sentence_content = str(content_raw)
                        elif content_raw.dtype.kind == 'O':  # Object (likely string references)
                            # Handle MATLAB string references
                            sentence_content = self.decode_matlab_string(content_raw, sent_group.file)
                    except:
                        pass
            
            # Process words
            if isinstance(word_data, h5py.Group):
                # Words are subgroups
                for word_key in word_data.keys():
                    word_group = word_data[word_key]
                    word_idx = int(word_key) if word_key.isdigit() else 0
                    
                    word_obj = self.extract_word_zuco2_hdf5(
                        word_group,
                        subject_id,
                        0,  # sentence_id - we don't have clear sentence indexing in HDF5
                        word_idx,
                        task_name,
                        sentence_content
                    )
                    if word_obj:
                        self.extracted_data.append(word_obj)
            
            elif isinstance(word_data, h5py.Dataset):
                # Words might be in a dataset - need to handle this case
                logger.debug("Word data is a dataset, structure needs investigation")
        
        else:
            # Alternative structure - sentenceData might directly contain sentence items
            # This needs further investigation based on actual data structure
            logger.debug(f"Alternative HDF5 structure found with keys: {list(sent_group.keys())[:10]}")
    
    def extract_word_zuco2_hdf5(self, word_group: h5py.Group, subject_id: str,
                                sent_idx: int, word_idx: int, task_name: str,
                                sentence_content: str) -> Optional[WordData]:
        """Extract word data from ZuCo 2.0 HDF5 format"""
        
        try:
            word_data = WordData(
                subject_id=subject_id,
                sentence_id=sent_idx,
                word_index=word_idx,
                task_name=task_name,
                word_content='',
                sentence_content=sentence_content
            )
            
            # Extract word content
            if 'content' in word_group:
                content = word_group['content'][()]
                if isinstance(content, bytes):
                    word_data.word_content = content.decode('utf-8')
                elif isinstance(content, str):
                    word_data.word_content = content
                else:
                    # Handle MATLAB string reference
                    word_data.word_content = self.decode_matlab_string(content, word_group.file)
            
            # Extract raw EEG
            if 'rawEEG' in word_group:
                raw_eeg = word_group['rawEEG'][()]
                if raw_eeg is not None and raw_eeg.size > 0:
                    word_data.raw_eeg = raw_eeg
                    word_data.has_fixation = True
            
            # Extract frequency bands
            freq_bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
            event_types = ['FFD', 'TRT', 'GD', 'GPT']
            
            for event in event_types:
                for band in freq_bands:
                    attr_name = f"{event}_{band}"
                    if attr_name in word_group:
                        value = word_group[attr_name][()]
                        if value is not None and value.size == 105:
                            setattr(word_data, attr_name, value)
                            word_data.has_fixation = True
            
            # Extract eye-tracking metrics
            if 'nFixations' in word_group:
                word_data.n_fixations = int(word_group['nFixations'][()])
            
            return word_data
            
        except Exception as e:
            logger.debug(f"Error extracting HDF5 word data: {e}")
            return None
    
    def extract_sentences_zuco2_mat(self, sentences, subject_id: str, task_name: str):
        """Extract sentences from ZuCo 2.0 that are in MATLAB v7 format"""
        # Similar to ZuCo 1.0 processing
        if not hasattr(sentences, '__len__'):
            sentences = [sentences]
        
        for sent_idx, sentence in enumerate(sentences):
            if not hasattr(sentence, 'word'):
                continue
            
            sentence_content = getattr(sentence, 'content', '')
            words = sentence.word
            
            if not hasattr(words, '__len__'):
                words = [words]
            
            for word_idx, word in enumerate(words):
                word_data = self.extract_word_zuco1(
                    word, 
                    subject_id, 
                    sent_idx, 
                    word_idx,
                    task_name,
                    sentence_content
                )
                if word_data:
                    self.extracted_data.append(word_data)
    
    def decode_matlab_string(self, ref, h5file) -> str:
        """Decode MATLAB string reference from HDF5"""
        try:
            if hasattr(ref, 'shape') and ref.shape == ():
                ref = ref[()]
            if isinstance(ref, h5py.Reference):
                return ''.join(chr(c) for c in h5file[ref][:].flatten())
            return str(ref)
        except:
            return ''
    
    def save_to_hdf5(self, output_path: str = "extracted_data.h5"):
        """Save extracted data to HDF5 format"""
        logger.info(f"Saving {len(self.extracted_data)} words to HDF5...")
        
        with h5py.File(output_path, 'w') as f:
            # Create groups for organization
            by_subject = defaultdict(list)
            for word in self.extracted_data:
                by_subject[word.subject_id].append(word)
            
            # Save data organized by subject
            for subject_id, words in by_subject.items():
                subject_group = f.create_group(subject_id)
                
                # Convert to structured arrays for efficient storage
                for i, word in enumerate(words):
                    word_group = subject_group.create_group(f"word_{i:05d}")
                    
                    # Save metadata
                    word_group.attrs['subject_id'] = word.subject_id
                    word_group.attrs['sentence_id'] = word.sentence_id
                    word_group.attrs['word_index'] = word.word_index
                    word_group.attrs['task_name'] = word.task_name
                    word_group.attrs['word_content'] = word.word_content
                    word_group.attrs['sentence_content'] = word.sentence_content
                    word_group.attrs['has_fixation'] = word.has_fixation
                    word_group.attrs['n_fixations'] = word.n_fixations if word.n_fixations else 0
                    
                    # Save EEG data
                    if word.raw_eeg is not None:
                        word_group.create_dataset('raw_eeg', data=word.raw_eeg, compression='gzip')
                    
                    # Save frequency bands
                    for attr_name in dir(word):
                        if any(attr_name.startswith(prefix) for prefix in ['FFD_', 'TRT_', 'GD_', 'GPT_']):
                            value = getattr(word, attr_name)
                            if value is not None and isinstance(value, np.ndarray):
                                word_group.create_dataset(attr_name, data=value, compression='gzip')
        
        logger.info(f"Data saved to {output_path}")
    
    def save_to_parquet(self, output_path: str = "extracted_data.parquet"):
        """Save extracted data to Parquet format"""
        logger.info(f"Saving {len(self.extracted_data)} words to Parquet...")
        
        # Convert to DataFrame-friendly format
        records = []
        for word in self.extracted_data:
            record = {
                'subject_id': word.subject_id,
                'sentence_id': word.sentence_id,
                'word_index': word.word_index,
                'task_name': word.task_name,
                'word_content': word.word_content,
                'sentence_content': word.sentence_content,
                'has_fixation': word.has_fixation,
                'n_fixations': word.n_fixations,
                'FFD_duration': word.FFD_duration,
                'TRT_duration': word.TRT_duration,
                'GD_duration': word.GD_duration,
                'GPT_duration': word.GPT_duration,
                'SFD_duration': word.SFD_duration,
            }
            
            # Add flattened EEG features (for DataFrame compatibility)
            # Store as separate columns or as serialized arrays
            if word.raw_eeg is not None:
                record['has_raw_eeg'] = True
                record['raw_eeg_shape'] = word.raw_eeg.shape
            else:
                record['has_raw_eeg'] = False
            
            # Flag which frequency bands are available
            for event in ['FFD', 'TRT', 'GD', 'GPT']:
                for band in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']:
                    attr_name = f"{event}_{band}"
                    value = getattr(word, attr_name)
                    record[f"has_{attr_name}"] = value is not None
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Metadata saved to {output_path}")
        
        # For actual EEG data, we might want to save separately due to size
        # or use a different format that better handles array data
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics of extracted data"""
        summary = {
            'total_words': len(self.extracted_data),
            'subjects': defaultdict(int),
            'tasks': defaultdict(int),
            'words_with_fixations': 0,
            'words_with_raw_eeg': 0,
            'words_with_frequency_bands': 0,
            'missing_data_count': 0
        }
        
        for word in self.extracted_data:
            summary['subjects'][word.subject_id] += 1
            summary['tasks'][word.task_name] += 1
            
            if word.has_fixation:
                summary['words_with_fixations'] += 1
            if word.raw_eeg is not None:
                summary['words_with_raw_eeg'] += 1
            if word.FFD_t1 is not None:  # Check if frequency bands exist
                summary['words_with_frequency_bands'] += 1
            if word.is_missing:
                summary['missing_data_count'] += 1
        
        summary['subjects'] = dict(summary['subjects'])
        summary['tasks'] = dict(summary['tasks'])
        
        return summary


def main():
    """Main extraction pipeline"""
    
    # Create output directory
    output_dir = Path("extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = ZuCoExtractor()
    
    # Extract all data
    logger.info("="*60)
    logger.info("Starting ZuCo Data Extraction Pipeline")
    logger.info("="*60)
    
    extracted_data = extractor.extract_all()
    
    # Generate and save summary
    summary = extractor.generate_summary()
    logger.info("\nExtraction Summary:")
    logger.info(f"  Total words extracted: {summary['total_words']}")
    logger.info(f"  Subjects: {len(summary['subjects'])}")
    logger.info(f"  Tasks: {list(summary['tasks'].keys())}")
    logger.info(f"  Words with fixations: {summary['words_with_fixations']}")
    logger.info(f"  Words with raw EEG: {summary['words_with_raw_eeg']}")
    logger.info(f"  Words with frequency bands: {summary['words_with_frequency_bands']}")
    
    # Save summary to JSON
    with open(output_dir / "extraction_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save extracted data
    if extracted_data:
        logger.info("\nSaving extracted data...")
        extractor.save_to_hdf5(str(output_dir / "zuco_extracted.h5"))
        extractor.save_to_parquet(str(output_dir / "zuco_metadata.parquet"))
        logger.info("Extraction pipeline complete!")
    else:
        logger.warning("No data was extracted!")
    
    return extracted_data


if __name__ == "__main__":
    data = main()