#!/usr/bin/env python3
"""
ZuCo Dataset Reconnaissance Script
Explores and documents the structure of both ZuCo 1.0 and 2.0 datasets
to identify version-specific differences and data inconsistencies.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import json
from collections import defaultdict
import scipy.io as sio
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ZuCoReconnaissance:
    """Explores ZuCo dataset structure across versions"""
    
    def __init__(self, base_path: str = "zuco_data"):
        self.base_path = Path(base_path)
        self.zuco1_path = self.base_path / "zuco1.0"
        self.zuco2_path = self.base_path / "zuco2.0"
        self.report = defaultdict(dict)
        
    def safe_load_mat(self, filepath: Path) -> Optional[Dict]:
        """Safely load a .mat file with error handling
        
        Handles both:
        - MATLAB v7 format (ZuCo 1.0) - uses scipy.io.loadmat
        - MATLAB v7.3 format (ZuCo 2.0) - HDF5-based, uses h5py
        """
        try:
            # First try scipy.io.loadmat (for v7 and earlier)
            data = sio.loadmat(str(filepath), struct_as_record=False, squeeze_me=True)
            print(f"  Loaded as MATLAB v7 format")
            return data
        except NotImplementedError:
            # This happens with v7.3 files, try h5py
            try:
                import h5py
                print(f"  Detected MATLAB v7.3 format, using h5py...")
                return self.load_mat_v73(filepath)
            except Exception as e:
                print(f"  Error with h5py: {e}")
                return None
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            return None
    
    def load_mat_v73(self, filepath: Path) -> Dict:
        """Load MATLAB v7.3 files using h5py"""
        import h5py
        
        # For now, just return the h5py File object wrapped in a dict
        # We'll explore it differently since it's HDF5
        f = h5py.File(filepath, 'r')
        return {'_h5file': f, '_keys': list(f.keys())}
    
    def explore_mat_structure(self, data: Dict, prefix: str = "") -> Dict:
        """Recursively explore the structure of loaded mat data"""
        import h5py
        
        structure = {}
        
        # Special handling for h5py files
        if '_h5file' in data:
            h5file = data['_h5file']
            for key in data['_keys']:
                if key.startswith('#'):
                    continue
                try:
                    item = h5file[key]
                    structure[key] = {
                        'type': 'h5py.Group' if isinstance(item, h5py.Group) else 'h5py.Dataset',
                        'shape': item.shape if hasattr(item, 'shape') else None,
                        'dtype': str(item.dtype) if hasattr(item, 'dtype') else None
                    }
                    if isinstance(item, h5py.Group):
                        structure[key]['keys'] = list(item.keys())
                except Exception as e:
                    structure[key] = {'type': 'error', 'error': str(e)}
            return structure
        
        # Regular scipy.io.loadmat data
        for key, value in data.items():
            # Skip metadata keys
            if key.startswith('__'):
                continue
                
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Get type and shape info
            value_type = type(value).__name__
            
            if hasattr(value, 'shape'):
                shape = value.shape
            elif hasattr(value, '__len__') and not isinstance(value, str):
                try:
                    shape = (len(value),)
                except:
                    shape = None
            else:
                shape = None
                
            structure[key] = {
                'type': value_type,
                'shape': shape
            }
            
            # Handle special types
            if value_type == 'matlab.struct':
                # MATLAB struct - explore fields
                if hasattr(value, '_fieldnames'):
                    fields = value._fieldnames
                    structure[key]['fields'] = fields
                    structure[key]['nested'] = {}
                    for field in fields:
                        try:
                            field_value = getattr(value, field)
                            structure[key]['nested'][field] = self.get_field_info(field_value)
                        except:
                            structure[key]['nested'][field] = {'type': 'error'}
                            
            elif isinstance(value, np.ndarray) and value.dtype == object:
                # Array of objects/structs
                if len(value) > 0:
                    sample = value[0] if value.ndim == 1 else value.flat[0]
                    if hasattr(sample, '_fieldnames'):
                        structure[key]['element_fields'] = sample._fieldnames
                        structure[key]['sample_structure'] = {}
                        for field in sample._fieldnames:
                            try:
                                field_value = getattr(sample, field)
                                structure[key]['sample_structure'][field] = self.get_field_info(field_value)
                            except:
                                structure[key]['sample_structure'][field] = {'type': 'error'}
                                
        return structure
    
    def get_field_info(self, value: Any) -> Dict:
        """Get basic info about a field value"""
        info = {'type': type(value).__name__}
        
        if hasattr(value, 'shape'):
            info['shape'] = value.shape
        elif hasattr(value, '__len__') and not isinstance(value, str):
            info['shape'] = (len(value),)
            
        # For nested structs, get field names
        if hasattr(value, '_fieldnames'):
            info['fields'] = value._fieldnames
            
        # Sample string content
        if isinstance(value, str):
            info['sample'] = value[:50] if len(value) > 50 else value
            
        return info
    
    def analyze_zuco1_structure(self):
        """Analyze ZuCo 1.0 data structure"""
        print("\n" + "="*60)
        print("ANALYZING ZUCO 1.0 STRUCTURE")
        print("="*60)
        
        self.report['zuco1'] = {
            'tasks': {},
            'subjects': set(),
            'file_patterns': defaultdict(list)
        }
        
        # Task 1: SR (Sentiment Reading)
        task1_path = self.zuco1_path / "task1-SR" / "Matlab files"
        if task1_path.exists():
            print("\nTask 1 - SR (Sentiment Reading):")
            print("-" * 40)
            
            for mat_file in sorted(task1_path.glob("*.mat")):
                subject = mat_file.stem.replace("results", "").replace("_SR", "")
                self.report['zuco1']['subjects'].add(subject)
                
                print(f"\nLoading {mat_file.name}...")
                data = self.safe_load_mat(mat_file)
                
                if data:
                    structure = self.explore_mat_structure(data)
                    self.report['zuco1']['tasks']['SR'] = structure
                    
                    # Print summary
                    for key, info in structure.items():
                        print(f"  {key}: {info['type']}", end="")
                        if info.get('shape'):
                            print(f" {info['shape']}", end="")
                        if info.get('fields'):
                            print(f" Fields: {info['fields']}", end="")
                        print()
                    
                    # Only need one file for structure
                    break
        
        # Task 2: NR (Normal Reading)
        task2_path = self.zuco1_path / "task2-NR" / "Matlab files"
        if task2_path.exists():
            print("\nTask 2 - NR (Normal Reading):")
            print("-" * 40)
            
            for mat_file in sorted(task2_path.glob("*.mat")):
                subject = mat_file.stem.replace("results", "").replace("_NR", "")
                self.report['zuco1']['subjects'].add(subject)
                
                print(f"\nLoading {mat_file.name}...")
                data = self.safe_load_mat(mat_file)
                
                if data:
                    structure = self.explore_mat_structure(data)
                    self.report['zuco1']['tasks']['NR'] = structure
                    
                    # Print summary
                    for key, info in structure.items():
                        print(f"  {key}: {info['type']}", end="")
                        if info.get('shape'):
                            print(f" {info['shape']}", end="")
                        if info.get('fields'):
                            print(f" Fields: {info['fields']}", end="")
                        print()
                    
                    # Only need one file for structure
                    break
        
        # Task 3: TSR (Task-Specific Reading)
        task3_path = self.zuco1_path / "task3-TSR" / "Matlab files"
        if task3_path.exists():
            print("\nTask 3 - TSR (Task-Specific Reading):")
            print("-" * 40)
            
            for mat_file in sorted(task3_path.glob("*.mat")):
                subject = mat_file.stem.replace("results", "").replace("_TSR", "")
                self.report['zuco1']['subjects'].add(subject)
                
                print(f"\nLoading {mat_file.name}...")
                data = self.safe_load_mat(mat_file)
                
                if data:
                    structure = self.explore_mat_structure(data)
                    self.report['zuco1']['tasks']['TSR'] = structure
                    
                    # Print summary
                    for key, info in structure.items():
                        print(f"  {key}: {info['type']}", end="")
                        if info.get('shape'):
                            print(f" {info['shape']}", end="")
                        if info.get('fields'):
                            print(f" Fields: {info['fields']}", end="")
                        print()
                    
                    # Only need one file for structure
                    break
    
    def analyze_zuco2_structure(self):
        """Analyze ZuCo 2.0 data structure"""
        print("\n" + "="*60)
        print("ANALYZING ZUCO 2.0 STRUCTURE")
        print("="*60)
        
        self.report['zuco2'] = {
            'tasks': {},
            'subjects': set(),
            'file_patterns': defaultdict(list)
        }
        
        # Task 1: NR (Normal Reading) in ZuCo 2.0
        task1_path = self.zuco2_path / "task1 - NR" / "Matlab files"
        if task1_path.exists():
            print("\nTask 1 - NR (Normal Reading):")
            print("-" * 40)
            
            for mat_file in sorted(task1_path.glob("*.mat")):
                subject = mat_file.stem.replace("results", "").replace("_NR", "")
                self.report['zuco2']['subjects'].add(subject)
                
                print(f"\nLoading {mat_file.name}...")
                data = self.safe_load_mat(mat_file)
                
                if data:
                    structure = self.explore_mat_structure(data)
                    self.report['zuco2']['tasks']['NR'] = structure
                    
                    # Print summary
                    for key, info in structure.items():
                        print(f"  {key}: {info['type']}", end="")
                        if info.get('shape'):
                            print(f" {info['shape']}", end="")
                        if info.get('fields'):
                            print(f" Fields: {info['fields']}", end="")
                        print()
                    
                    # Only need one file for structure
                    break
        
        # Task 2: TSR (Task-Specific Reading) in ZuCo 2.0
        task2_path = self.zuco2_path / "task2 - TSR" / "Matlab files"
        if task2_path.exists():
            print("\nTask 2 - TSR (Task-Specific Reading):")
            print("-" * 40)
            
            for mat_file in sorted(task2_path.glob("*.mat")):
                subject = mat_file.stem.replace("results", "").replace("_TSR", "")
                self.report['zuco2']['subjects'].add(subject)
                
                print(f"\nLoading {mat_file.name}...")
                data = self.safe_load_mat(mat_file)
                
                if data:
                    structure = self.explore_mat_structure(data)
                    self.report['zuco2']['tasks']['TSR'] = structure
                    
                    # Print summary
                    for key, info in structure.items():
                        print(f"  {key}: {info['type']}", end="")
                        if info.get('shape'):
                            print(f" {info['shape']}", end="")
                        if info.get('fields'):
                            print(f" Fields: {info['fields']}", end="")
                        print()
                    
                    # Only need one file for structure
                    break
    
    def deep_dive_sentence_structure(self):
        """Do a deep dive into the sentence/word structure"""
        import h5py
        
        print("\n" + "="*60)
        print("DEEP DIVE: SENTENCE AND WORD STRUCTURE")
        print("="*60)
        
        # Sample from ZuCo 1.0
        print("\nZuCo 1.0 Sample (Task 1 - SR):")
        print("-" * 40)
        
        sample_file = self.zuco1_path / "task1-SR" / "Matlab files" / "resultsZAB_SR.mat"
        if sample_file.exists():
            data = self.safe_load_mat(sample_file)
            if data and 'sentenceData' in data:
                sentences = data['sentenceData']
                
                # Check if it's an array
                if hasattr(sentences, '__len__'):
                    print(f"Number of sentences: {len(sentences)}")
                    
                    # Examine first sentence
                    if len(sentences) > 0:
                        sent = sentences[0] if isinstance(sentences, np.ndarray) else sentences
                        print(f"\nFirst sentence structure:")
                        
                        if hasattr(sent, '_fieldnames'):
                            for field in sent._fieldnames:
                                field_value = getattr(sent, field)
                                print(f"  {field}:", end=" ")
                                
                                if isinstance(field_value, str):
                                    print(f"'{field_value[:50]}...'" if len(field_value) > 50 else f"'{field_value}'")
                                elif hasattr(field_value, 'shape'):
                                    print(f"array {field_value.shape}")
                                elif hasattr(field_value, '__len__'):
                                    print(f"length {len(field_value)}")
                                    
                                    # If it's words, examine first word
                                    if field == 'word' and len(field_value) > 0:
                                        word = field_value[0]
                                        if hasattr(word, '_fieldnames'):
                                            print(f"    First word fields: {word._fieldnames}")
                                            
                                            # Sample some word fields
                                            for wfield in ['content', 'rawEEG', 'FFD', 'GD', 'GPT', 'TRT', 
                                                         'FFD_t1', 'FFD_a1', 'FFD_b1', 'FFD_g1']:
                                                if wfield in word._fieldnames:
                                                    try:
                                                        wvalue = getattr(word, wfield)
                                                        if isinstance(wvalue, str):
                                                            print(f"      {wfield}: '{wvalue}'")
                                                        elif hasattr(wvalue, 'shape'):
                                                            print(f"      {wfield}: array {wvalue.shape}")
                                                        elif wvalue is None or (hasattr(wvalue, '__len__') and len(wvalue) == 0):
                                                            print(f"      {wfield}: empty/None")
                                                        else:
                                                            print(f"      {wfield}: {type(wvalue).__name__}")
                                                    except:
                                                        print(f"      {wfield}: <error accessing>")
                                else:
                                    print(f"{type(field_value).__name__}")
        
        # Sample from ZuCo 2.0
        print("\n\nZuCo 2.0 Sample (Task 1 - NR):")
        print("-" * 40)
        
        sample_file = self.zuco2_path / "task1 - NR" / "Matlab files" / "resultsYAC_NR.mat"
        if sample_file.exists():
            data = self.safe_load_mat(sample_file)
            if data:
                if '_h5file' in data:
                    # Handle h5py format
                    h5file = data['_h5file']
                    print(f"Main data keys: {data['_keys']}")
                    
                    # Explore sentenceData if it exists
                    if 'sentenceData' in data['_keys']:
                        sent_data = h5file['sentenceData']
                        print(f"\nsentenceData: {sent_data}")
                        print(f"  Type: {type(sent_data)}")
                        print(f"  Shape: {sent_data.shape if hasattr(sent_data, 'shape') else 'N/A'}")
                        
                        # If it's a dataset, try to access first element
                        if isinstance(sent_data, h5py.Dataset):
                            print(f"  Dtype: {sent_data.dtype}")
                        elif isinstance(sent_data, h5py.Group):
                            print(f"  Keys: {list(sent_data.keys())[:10]}")  # First 10 keys
                else:
                    # Find the main data field (might not be 'sentenceData')
                    main_keys = [k for k in data.keys() if not k.startswith('__')]
                    print(f"Main data keys: {main_keys}")
                    
                    for key in main_keys:
                        value = data[key]
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            print(f"\n{key}: length {len(value)}")
                            
                            # If it looks like sentence data
                            if len(value) > 0:
                                item = value[0] if isinstance(value, np.ndarray) else value
                                if hasattr(item, '_fieldnames'):
                                    print(f"  First item fields: {item._fieldnames}")
                                    
                                    # Check for word field
                                    if 'word' in item._fieldnames:
                                        words = getattr(item, 'word')
                                        if hasattr(words, '__len__') and len(words) > 0:
                                            word = words[0]
                                            if hasattr(word, '_fieldnames'):
                                                print(f"    First word fields: {word._fieldnames}")
    
    def compare_versions(self):
        """Compare ZuCo 1.0 and 2.0 structures"""
        print("\n" + "="*60)
        print("VERSION COMPARISON")
        print("="*60)
        
        # Compare subjects
        print("\nSubjects:")
        print(f"  ZuCo 1.0: {sorted(self.report['zuco1']['subjects'])}")
        print(f"  ZuCo 2.0: {sorted(self.report['zuco2']['subjects'])}")
        
        # Compare task structure
        print("\nTask Structure:")
        print(f"  ZuCo 1.0 tasks: {list(self.report['zuco1']['tasks'].keys())}")
        print(f"  ZuCo 2.0 tasks: {list(self.report['zuco2']['tasks'].keys())}")
        
        # Field differences
        if self.report['zuco1']['tasks'] and self.report['zuco2']['tasks']:
            print("\nField Differences:")
            
            # Get a task from each version
            zuco1_task = list(self.report['zuco1']['tasks'].values())[0] if self.report['zuco1']['tasks'] else {}
            zuco2_task = list(self.report['zuco2']['tasks'].values())[0] if self.report['zuco2']['tasks'] else {}
            
            zuco1_fields = set(zuco1_task.keys())
            zuco2_fields = set(zuco2_task.keys())
            
            print(f"  Fields only in ZuCo 1.0: {zuco1_fields - zuco2_fields}")
            print(f"  Fields only in ZuCo 2.0: {zuco2_fields - zuco1_fields}")
            print(f"  Common fields: {zuco1_fields & zuco2_fields}")
    
    def save_report(self):
        """Save the reconnaissance report"""
        output_file = "zuco_structure_report.json"
        
        # Convert sets to lists for JSON serialization
        def serialize(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=serialize)
        
        print(f"\nReport saved to {output_file}")
    
    def run(self):
        """Run the complete reconnaissance"""
        print("Starting ZuCo Dataset Reconnaissance...")
        
        # Check paths exist
        if not self.base_path.exists():
            print(f"Error: {self.base_path} does not exist!")
            return
        
        if not self.zuco1_path.exists():
            print(f"Warning: ZuCo 1.0 path not found at {self.zuco1_path}")
        else:
            self.analyze_zuco1_structure()
        
        if not self.zuco2_path.exists():
            print(f"Warning: ZuCo 2.0 path not found at {self.zuco2_path}")
        else:
            self.analyze_zuco2_structure()
        
        # Deep dive into structure
        self.deep_dive_sentence_structure()
        
        # Compare versions
        self.compare_versions()
        
        # Save report
        self.save_report()
        
        print("\nReconnaissance complete!")


if __name__ == "__main__":
    recon = ZuCoReconnaissance()
    recon.run()