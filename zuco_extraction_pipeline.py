#!/usr/bin/env python3
"""
ZuCo Dataset Unified Extraction Pipeline
Extracts word-level EEG data from both ZuCo 1.0 and 2.0 datasets
Handles both MATLAB v7 (scipy.io) and v7.3 (HDF5) formats
Outputs to modern Python formats (HDF5/Parquet)

Features:
- Separate HDF5 files per subject-task (e.g., zuco1_ZAB_SR.h5)
- Resume capability with checkpoint tracking
- Progress bars for all operations
- Optimized compression (lzf for speed)
- Modular functions for testing
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
import time
from datetime import datetime

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


class CheckpointManager:
    """Manages extraction progress and resumption"""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.progress = self.load_checkpoint()
    
    def load_checkpoint(self) -> Dict:
        """Load existing checkpoint or create new one"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    # Ensure required keys exist (handle old checkpoint formats)
                    if 'completed_files' not in data:
                        data['completed_files'] = []
                    if 'last_updated' not in data:
                        data['last_updated'] = None
                    if 'total_words_extracted' not in data:
                        data['total_words_extracted'] = 0
                    if 'extraction_stats' not in data:
                        data['extraction_stats'] = {}
                    return data
            except (json.JSONDecodeError, KeyError):
                # If checkpoint is corrupted, start fresh
                logger.warning("Checkpoint file corrupted, starting fresh")
                return {
                    'completed_files': [],
                    'last_updated': None,
                    'total_words_extracted': 0,
                    'extraction_stats': {}
                }
        return {
            'completed_files': [],  # List of completed file identifiers
            'last_updated': None,
            'total_words_extracted': 0,
            'extraction_stats': {}
        }
    
    def save_checkpoint(self):
        """Save current progress"""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_file_complete(self, file_id: str) -> bool:
        """Check if a file is already complete"""
        return file_id in self.progress['completed_files']
    
    def mark_file_complete(self, file_id: str, word_count: int):
        """Mark a file as complete"""
        if file_id not in self.progress['completed_files']:
            self.progress['completed_files'].append(file_id)
        self.progress['total_words_extracted'] += word_count
        self.save_checkpoint()
    
    def get_incomplete_files(self, all_files: List[Tuple[str, str, str, Path]]) -> List[Tuple[str, str, str, Path]]:
        """Filter out already completed files"""
        incomplete = []
        for version, subject_id, task_name, file_path in all_files:
            file_id = f"{version}_{subject_id}_{task_name}"
            if not self.is_file_complete(file_id):
                incomplete.append((version, subject_id, task_name, file_path))
        return incomplete


def extract_word_zuco1(word, subject_id: str, sent_idx: int, 
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
                            setattr(word_data, attr_name, value.astype(np.float32))
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


def extract_file_zuco1(mat_file: Path, subject_id: str, task_name: str, 
                       show_progress: bool = True) -> List[WordData]:
    """Extract data from a single ZuCo 1.0 file"""
    word_data_list = []
    
    try:
        # Load MATLAB file
        data = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
        
        if 'sentenceData' not in data:
            logger.warning(f"No sentenceData in {mat_file.name}")
            return word_data_list
        
        sentences = data['sentenceData']
        if not hasattr(sentences, '__len__'):
            sentences = [sentences]
        
        # Process each sentence with optional progress bar
        sentence_iter = sentences
        if show_progress:
            sentence_iter = tqdm(sentences, desc=f"  {subject_id}", leave=False, unit="sent")
        
        for sent_idx, sentence in enumerate(sentence_iter):
            if not hasattr(sentence, 'word'):
                continue
            
            sentence_content = getattr(sentence, 'content', '')
            words = sentence.word
            
            if not hasattr(words, '__len__'):
                words = [words]
            
            # Process each word
            for word_idx, word in enumerate(words):
                word_data = extract_word_zuco1(
                    word, 
                    subject_id, 
                    sent_idx, 
                    word_idx,
                    task_name,
                    sentence_content
                )
                if word_data:
                    word_data_list.append(word_data)
    
    except Exception as e:
        logger.error(f"Error processing {mat_file.name}: {e}")
    
    return word_data_list


def extract_word_zuco2_hdf5(word_ref, h5file, subject_id: str, sent_idx: int,
                            word_idx: int, task_name: str, sentence_content: str) -> Optional[WordData]:
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
        
        # Dereference the word object
        word_obj = h5file[word_ref]
        
        # Extract word content
        if 'content' in word_obj:
            content_ref = word_obj['content'][()]
            if isinstance(content_ref, h5py.Reference):
                # Dereference and decode
                content_data = h5file[content_ref][()]
                if content_data.dtype.kind == 'U':
                    word_data.word_content = str(content_data)
                else:
                    # Character array - convert to string
                    word_data.word_content = ''.join(chr(c) for c in content_data.flatten() if c > 0)
            elif isinstance(content_ref, np.ndarray):
                # Direct character array
                word_data.word_content = ''.join(chr(c) for c in content_ref.flatten() if c > 0)
        
        # Extract raw EEG
        if 'rawEEG' in word_obj:
            raw_eeg_ref = word_obj['rawEEG'][()]
            if isinstance(raw_eeg_ref, h5py.Reference):
                raw_eeg = h5file[raw_eeg_ref][()]
                if raw_eeg.size > 0:
                    # Ensure correct shape
                    if raw_eeg.ndim == 2:
                        if raw_eeg.shape[0] == 105:
                            raw_eeg = raw_eeg.T
                        word_data.raw_eeg = raw_eeg.astype(np.float32)
                        word_data.has_fixation = True
        
        # Extract frequency bands
        freq_bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
        event_types = ['FFD', 'TRT', 'GD', 'GPT']
        
        for event in event_types:
            for band in freq_bands:
                attr_name = f"{event}_{band}"
                if attr_name in word_obj:
                    band_ref = word_obj[attr_name][()]
                    if isinstance(band_ref, h5py.Reference):
                        band_data = h5file[band_ref][()]
                        if band_data.size == 105:
                            setattr(word_data, attr_name, band_data.flatten().astype(np.float32))
                            word_data.has_fixation = True
        
        # Extract eye-tracking metrics
        if 'nFixations' in word_obj:
            nfix_ref = word_obj['nFixations'][()]
            if isinstance(nfix_ref, h5py.Reference):
                word_data.n_fixations = int(h5file[nfix_ref][()])
            else:
                word_data.n_fixations = int(nfix_ref) if nfix_ref else 0
        
        # Extract durations
        for metric in ['FFD', 'TRT', 'GD', 'GPT', 'SFD']:
            if metric in word_obj:
                dur_ref = word_obj[metric][()]
                if isinstance(dur_ref, h5py.Reference):
                    dur_val = h5file[dur_ref][()]
                    if dur_val.size == 1:
                        setattr(word_data, f"{metric}_duration", float(dur_val))
                elif not isinstance(dur_ref, (list, np.ndarray)):
                    setattr(word_data, f"{metric}_duration", float(dur_ref))
        
        return word_data
        
    except Exception as e:
        logger.debug(f"Error extracting HDF5 word data: {e}")
        return None


def extract_file_zuco2(mat_file: Path, subject_id: str, task_name: str,
                       show_progress: bool = True) -> List[WordData]:
    """Extract data from a single ZuCo 2.0 file
    
    ZuCo 2.0 has a different structure where each sentence's words are grouped together.
    Based on the official scripts in zuco2.0/scripts/python_reader/
    """
    word_data_list = []
    
    try:
        # ZuCo 2.0 uses HDF5 format
        with h5py.File(mat_file, 'r') as f:
            if 'sentenceData' not in f:
                logger.warning(f"No sentenceData in {mat_file.name}")
                return word_data_list
            
            sent_group = f['sentenceData']
            
            # Check if it's a group (ZuCo 2.0 structure)
            if isinstance(sent_group, h5py.Group):
                # This is the aggregated structure with word data
                if 'word' not in sent_group or 'content' not in sent_group:
                    logger.warning(f"Missing word or content data in {mat_file.name}")
                    return word_data_list
                
                word_data_refs = sent_group['word']
                content_refs = sent_group['content']
                
                n_sentences = word_data_refs.shape[0]
                
                # Process each sentence
                sentence_iter = range(n_sentences)
                if show_progress:
                    sentence_iter = tqdm(sentence_iter, desc=f"  {subject_id}", leave=False, unit="sent")
                
                for sent_idx in sentence_iter:
                    # Get sentence content
                    sentence_content = ''
                    if sent_idx < content_refs.shape[0]:
                        content_ref = content_refs[sent_idx, 0] if len(content_refs.shape) > 1 else content_refs[sent_idx]
                        if isinstance(content_ref, h5py.Reference):
                            content_data = f[content_ref][()]
                            # Decode the sentence string
                            sentence_content = ''.join(chr(c) for c in content_data.flatten() if c > 0)
                    
                    # Get word group for this sentence
                    word_group_ref = word_data_refs[sent_idx, 0] if len(word_data_refs.shape) > 1 else word_data_refs[sent_idx]
                    
                    if not isinstance(word_group_ref, h5py.Reference):
                        continue
                    
                    # Dereference to get the word group
                    word_group = f[word_group_ref]
                    
                    if not isinstance(word_group, h5py.Group):
                        continue
                    
                    # Extract all words from this sentence
                    words = extract_words_from_sentence_zuco2(
                        f, word_group, subject_id, sent_idx, task_name, sentence_content
                    )
                    
                    word_data_list.extend(words)
            
            else:
                logger.warning(f"Unexpected sentenceData structure in {mat_file.name}")
    
    except OSError:
        # If HDF5 fails, try scipy.io.loadmat (some files might be v7)
        logger.debug(f"Trying to load {mat_file.name} as MATLAB v7")
        try:
            return extract_file_zuco1(mat_file, subject_id, task_name, show_progress)
        except Exception as e:
            logger.error(f"Error loading {mat_file.name}: {e}")
    
    except Exception as e:
        logger.error(f"Error processing {mat_file.name}: {e}")
    
    return word_data_list


def extract_words_from_sentence_zuco2(h5file, word_group: h5py.Group, subject_id: str,
                                      sent_idx: int, task_name: str, sentence_content: str) -> List[WordData]:
    """Extract all words from a sentence in ZuCo 2.0 format
    
    Based on extract_word_level_data() from data_loading_helpers.py
    """
    word_data_list = []
    
    try:
        # Check if we have the expected structure
        if 'content' not in word_group:
            return word_data_list
        
        # Get arrays for all words in this sentence
        content_data = word_group['content']
        
        # Check for EEG data
        has_eeg = 'rawEEG' in word_group
        
        if has_eeg:
            raw_eeg_data = word_group['rawEEG']
            n_words = raw_eeg_data.shape[0]
            
            # Get eye-tracking metrics if available
            ffd_data = word_group.get('FFD', None)
            trt_data = word_group.get('TRT', None)
            gd_data = word_group.get('GD', None)
            gpt_data = word_group.get('GPT', None)
            sfd_data = word_group.get('SFD', None)
            nfix_data = word_group.get('nFixations', None)
            
            # Get frequency bands
            freq_bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
            event_types = ['FFD', 'TRT', 'GD', 'GPT']
            
            # Process each word
            for word_idx in range(n_words):
                word_data = WordData(
                    subject_id=subject_id,
                    sentence_id=sent_idx,
                    word_index=word_idx,
                    task_name=task_name,
                    word_content='',
                    sentence_content=sentence_content
                )
                
                # Get word content
                if word_idx < content_data.shape[0]:
                    content_ref = content_data[word_idx, 0] if len(content_data.shape) > 1 else content_data[word_idx]
                    if isinstance(content_ref, h5py.Reference):
                        word_text_data = h5file[content_ref][()]
                        # Decode word string using the official method
                        word_data.word_content = ''.join(chr(c) for c in word_text_data.flatten())
                
                # Get raw EEG
                if word_idx < raw_eeg_data.shape[0]:
                    eeg_ref = raw_eeg_data[word_idx, 0] if len(raw_eeg_data.shape) > 1 else raw_eeg_data[word_idx]
                    if isinstance(eeg_ref, h5py.Reference):
                        eeg_obj = h5file[eeg_ref]
                        if isinstance(eeg_obj, h5py.Dataset):
                            raw_eeg = eeg_obj[()]
                            if raw_eeg.size > 0:
                                # Handle the nested structure for multiple fixations
                                if raw_eeg.dtype == object:
                                    # Multiple fixations stored as references
                                    fixation_data = []
                                    for fix_ref in raw_eeg.flatten():
                                        if isinstance(fix_ref, h5py.Reference):
                                            fix_data = h5file[fix_ref][()]
                                            fixation_data.append(fix_data)
                                    if fixation_data:
                                        word_data.raw_eeg = np.array(fixation_data).astype(np.float32)
                                else:
                                    # Direct array
                                    if raw_eeg.ndim == 2 and raw_eeg.shape[1] == 105:
                                        word_data.raw_eeg = raw_eeg.astype(np.float32)
                                    elif raw_eeg.ndim == 2 and raw_eeg.shape[0] == 105:
                                        word_data.raw_eeg = raw_eeg.T.astype(np.float32)
                                
                                word_data.has_fixation = True
                
                # Get frequency bands
                for event in event_types:
                    for band in freq_bands:
                        attr_name = f"{event}_{band}"
                        if attr_name in word_group:
                            band_data = word_group[attr_name]
                            if word_idx < band_data.shape[0]:
                                band_ref = band_data[word_idx, 0] if len(band_data.shape) > 1 else band_data[word_idx]
                                if isinstance(band_ref, h5py.Reference):
                                    band_values = h5file[band_ref][()]
                                    if band_values.size == 105:
                                        setattr(word_data, attr_name, band_values.flatten().astype(np.float32))
                                        word_data.has_fixation = True
                
                # Get eye-tracking metrics
                if nfix_data is not None and word_idx < nfix_data.shape[0]:
                    nfix_ref = nfix_data[word_idx, 0] if len(nfix_data.shape) > 1 else nfix_data[word_idx]
                    if isinstance(nfix_ref, h5py.Reference):
                        nfix_val = h5file[nfix_ref][()]
                        if nfix_val.size > 0:
                            word_data.n_fixations = int(nfix_val.flatten()[0])
                
                # Get durations
                for metric_name, metric_data in [('FFD', ffd_data), ('TRT', trt_data), 
                                                 ('GD', gd_data), ('GPT', gpt_data), ('SFD', sfd_data)]:
                    if metric_data is not None and word_idx < metric_data.shape[0]:
                        metric_ref = metric_data[word_idx, 0] if len(metric_data.shape) > 1 else metric_data[word_idx]
                        if isinstance(metric_ref, h5py.Reference):
                            metric_val = h5file[metric_ref][()]
                            if metric_val.size > 0:
                                setattr(word_data, f"{metric_name}_duration", float(metric_val.flatten()[0]))
                
                # Only add words with actual content
                if word_data.word_content and word_data.word_content.strip():
                    word_data_list.append(word_data)
        
        else:
            # No EEG data, just extract word content
            n_words = content_data.shape[0]
            for word_idx in range(n_words):
                content_ref = content_data[word_idx, 0] if len(content_data.shape) > 1 else content_data[word_idx]
                if isinstance(content_ref, h5py.Reference):
                    word_text_data = h5file[content_ref][()]
                    word_content = ''.join(chr(c) for c in word_text_data.flatten())
                    
                    if word_content and word_content.strip():
                        word_data = WordData(
                            subject_id=subject_id,
                            sentence_id=sent_idx,
                            word_index=word_idx,
                            task_name=task_name,
                            word_content=word_content,
                            sentence_content=sentence_content
                        )
                        word_data_list.append(word_data)
    
    except Exception as e:
        logger.debug(f"Error extracting words from sentence: {e}")
    
    return word_data_list


def save_subject_file(word_data_list: List[WordData], output_path: Path,
                     show_progress: bool = True):
    """Save word data to a single HDF5 file"""
    
    with h5py.File(output_path, 'w') as f:
        # Add metadata
        f.attrs['n_words'] = len(word_data_list)
        f.attrs['subject_id'] = word_data_list[0].subject_id if word_data_list else ''
        f.attrs['task_name'] = word_data_list[0].task_name if word_data_list else ''
        f.attrs['created'] = datetime.now().isoformat()
        
        # Save words with optional progress bar
        word_iter = enumerate(word_data_list)
        if show_progress:
            word_iter = enumerate(tqdm(word_data_list, 
                                     desc=f"    Saving", 
                                     leave=False, unit="word"))
        
        for i, word in word_iter:
            word_group = f.create_group(f"word_{i:05d}")
            
            # Save metadata
            word_group.attrs['subject_id'] = word.subject_id
            word_group.attrs['sentence_id'] = word.sentence_id
            word_group.attrs['word_index'] = word.word_index
            word_group.attrs['task_name'] = word.task_name
            word_group.attrs['word_content'] = word.word_content
            word_group.attrs['sentence_content'] = word.sentence_content
            word_group.attrs['has_fixation'] = word.has_fixation
            word_group.attrs['n_fixations'] = word.n_fixations if word.n_fixations else 0
            
            # Save eye-tracking durations
            for duration_type in ['FFD', 'TRT', 'GD', 'GPT', 'SFD']:
                duration_attr = f"{duration_type}_duration"
                if hasattr(word, duration_attr):
                    value = getattr(word, duration_attr)
                    if value is not None:
                        word_group.attrs[duration_attr] = value
            
            # Save EEG data with lzf compression (faster than gzip)
            if word.raw_eeg is not None:
                word_group.create_dataset('raw_eeg', data=word.raw_eeg, compression='lzf')
            
            # Save frequency bands
            for attr_name in dir(word):
                if any(attr_name.startswith(prefix) for prefix in ['FFD_', 'TRT_', 'GD_', 'GPT_']):
                    value = getattr(word, attr_name)
                    if value is not None and isinstance(value, np.ndarray):
                        word_group.create_dataset(attr_name, data=value, compression='lzf')


def get_all_files(base_path: str = "zuco_data") -> List[Tuple[str, str, str, Path]]:
    """Get all subject files to process
    Returns: List of (version, subject_id, task_name, file_path) tuples
    """
    base = Path(base_path)
    all_files = []
    
    # ZuCo 1.0 files
    zuco1_path = base / "zuco1.0"
    if zuco1_path.exists():
        tasks = [
            ("task1-SR", "SR", "Matlab files"),
            ("task2-NR", "NR", "Matlab files"),
            ("task3-TSR", "TSR", "Matlab files")
        ]
        
        for task_dir, task_name, matlab_subdir in tasks:
            task_path = zuco1_path / task_dir / matlab_subdir
            if task_path.exists():
                for mat_file in sorted(task_path.glob("*.mat")):
                    subject_id = mat_file.stem.replace("results", "").replace(f"_{task_name}", "")
                    all_files.append(("zuco1", subject_id, task_name, mat_file))
    
    # ZuCo 2.0 files
    zuco2_path = base / "zuco2.0"
    if zuco2_path.exists():
        tasks = [
            ("task1 - NR", "NR", "Matlab files"),
            ("task2 - TSR", "TSR", "Matlab files")
        ]
        
        for task_dir, task_name, matlab_subdir in tasks:
            task_path = zuco2_path / task_dir / matlab_subdir
            if task_path.exists():
                for mat_file in sorted(task_path.glob("*.mat")):
                    subject_id = mat_file.stem.replace("results", "").replace(f"_{task_name}", "")
                    all_files.append(("zuco2", subject_id, task_name, mat_file))
    
    return all_files


def extract_all(output_dir: str = "extracted_data", 
                resume: bool = True,
                test_mode: bool = False,
                max_files: Optional[int] = None):
    """
    Main extraction function
    
    Args:
        output_dir: Directory to save extracted files
        resume: Whether to resume from checkpoint
        test_mode: If True, process only a subset for testing
        max_files: Maximum number of files to process (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize checkpoint manager if resuming
    checkpoint = None
    if resume:
        checkpoint = CheckpointManager(output_path / "extraction_checkpoint.json")
    
    # Get all files
    all_files = get_all_files()
    logger.info(f"Found {len(all_files)} total files")
    
    # Filter for test mode or max_files
    if test_mode:
        # Just process first file from each version/task combo for testing
        test_files = []
        seen_combos = set()
        for version, subject_id, task_name, file_path in all_files:
            combo = (version, task_name)
            if combo not in seen_combos:
                test_files.append((version, subject_id, task_name, file_path))
                seen_combos.add(combo)
                if len(test_files) >= 4:  # Get samples from both versions
                    break
        all_files = test_files
        logger.info(f"Test mode: Processing {len(all_files)} sample files")
    
    if max_files:
        all_files = all_files[:max_files]
        logger.info(f"Limited to {max_files} files")
    
    # Filter out completed files if resuming
    if checkpoint:
        incomplete_files = checkpoint.get_incomplete_files(all_files)
        if len(incomplete_files) < len(all_files):
            logger.info(f"Resuming: {len(all_files) - len(incomplete_files)} files already completed")
            all_files = incomplete_files
    
    # Collect metadata for parquet
    all_metadata = []
    
    # Process each file
    with tqdm(total=len(all_files), desc="Overall Progress", unit="file") as pbar:
        for version, subject_id, task_name, file_path in all_files:
            file_id = f"{version}_{subject_id}_{task_name}"
            pbar.set_description(f"Processing {file_id}")
            
            # Extract data
            word_data_list = []
            try:
                if version == "zuco1":
                    word_data_list = extract_file_zuco1(file_path, subject_id, task_name)
                else:  # zuco2
                    word_data_list = extract_file_zuco2(file_path, subject_id, task_name)
                
                if word_data_list:
                    # Save to separate HDF5 file
                    output_file = output_path / f"{file_id}.h5"
                    save_subject_file(word_data_list, output_file)
                    
                    # Collect metadata
                    for word in word_data_list:
                        all_metadata.append({
                            'file_id': file_id,
                            'version': version,
                            'subject_id': word.subject_id,
                            'sentence_id': word.sentence_id,
                            'word_index': word.word_index,
                            'task_name': word.task_name,
                            'word_content': word.word_content,
                            'has_fixation': word.has_fixation,
                            'n_fixations': word.n_fixations,
                            'FFD_duration': word.FFD_duration,
                            'TRT_duration': word.TRT_duration,
                            'GD_duration': word.GD_duration,
                            'GPT_duration': word.GPT_duration,
                            'SFD_duration': word.SFD_duration,
                            'has_raw_eeg': word.raw_eeg is not None,
                            'raw_eeg_shape': word.raw_eeg.shape if word.raw_eeg is not None else None
                        })
                    
                    # Update checkpoint
                    if checkpoint:
                        checkpoint.mark_file_complete(file_id, len(word_data_list))
                    
                    logger.info(f"  Saved {len(word_data_list)} words to {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
            
            pbar.update(1)
    
    # Save metadata to parquet
    if all_metadata:
        logger.info(f"Saving metadata for {len(all_metadata)} words to Parquet...")
        df = pd.DataFrame(all_metadata)
        df.to_parquet(output_path / "zuco_metadata.parquet", index=False, compression='snappy')
    
    # Generate and save summary
    summary = generate_summary(output_path)
    # Convert numpy types to Python native types for JSON serialization
    summary_json = json.loads(json.dumps(summary, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))
    with open(output_path / "extraction_summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    logger.info("Extraction complete!")
    return summary


def generate_summary(output_dir: Path) -> Dict:
    """Generate summary statistics from extracted HDF5 files"""
    summary = {
        'total_words': 0,
        'total_files': 0,
        'versions': defaultdict(int),
        'subjects': defaultdict(int),
        'tasks': defaultdict(int),
        'words_with_fixations': 0,
        'words_with_raw_eeg': 0,
        'words_with_frequency_bands': 0,
        'extraction_completed': datetime.now().isoformat()
    }
    
    # Process each HDF5 file
    for h5_file in output_dir.glob("*.h5"):
        if h5_file.name == "extraction_checkpoint.json":
            continue
            
        summary['total_files'] += 1
        
        # Parse filename to get version, subject, task
        parts = h5_file.stem.split('_')
        if len(parts) >= 3:
            version = parts[0]
            subject_id = parts[1]
            task_name = parts[2]
            
            summary['versions'][version] += 1
            
            with h5py.File(h5_file, 'r') as f:
                n_words = f.attrs.get('n_words', len(f.keys()))
                summary['total_words'] += n_words
                summary['subjects'][subject_id] += n_words
                summary['tasks'][task_name] += n_words
                
                # Sample some words for statistics
                for word_name in list(f.keys())[:100]:  # Sample first 100
                    word_group = f[word_name]
                    if word_group.attrs.get('has_fixation', False):
                        summary['words_with_fixations'] += 1
                    if 'raw_eeg' in word_group:
                        summary['words_with_raw_eeg'] += 1
                    if 'FFD_t1' in word_group:
                        summary['words_with_frequency_bands'] += 1
    
    # Convert defaultdicts to regular dicts
    summary['versions'] = dict(summary['versions'])
    summary['subjects'] = dict(summary['subjects'])
    summary['tasks'] = dict(summary['tasks'])
    
    return summary


def main():
    """Main extraction pipeline"""
    
    logger.info("="*60)
    logger.info("ZuCo Data Extraction Pipeline")
    logger.info("With separate files per subject-task")
    logger.info("="*60)
    
    start_time = time.time()
    summary = extract_all()
    elapsed_time = time.time() - start_time
    
    logger.info("="*60)
    logger.info(f"Extraction Summary:")
    logger.info(f"  Total files: {summary.get('total_files', 0)}")
    logger.info(f"  Total words: {summary.get('total_words', 0)}")
    logger.info(f"  Time elapsed: {elapsed_time:.1f} seconds")
    logger.info("="*60)


if __name__ == "__main__":
    main()