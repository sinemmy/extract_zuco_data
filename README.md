# ZuCo EEG Datasets as consistent HDF5 files

A robust Python pipeline for extracting and modernizing word-level EEG data from the ZuCo (Zurich Cognitive Language Processing Corpus) and ZuCo 2.0 datasets. This tool converts MATLAB files into organized HDF5 format with preserved sentence context, perfect for EEG-language model alignment research.

Full extracted data can be found on [ZuCo 2.0 fork on OSF](https://osf.io/wq6ps/)

### Verify Downloaded Data

After downloading the extracted HDF5 files from OSF fork, you can verify the extraction statistics:

```bash
# Recalculate extraction summary to verify data coverage (default: extracted_data/)
python src/recalculate_summary.py

# Specify custom input directory
python src/recalculate_summary.py --input-dir /path/to/hdf5/files

# Specify custom output file
python src/recalculate_summary.py --output /path/to/summary.json

# Example: Analyze files in a different location and save summary elsewhere
python src/recalculate_summary.py --input-dir ~/Downloads/zuco_extracted --output ~/analysis/zuco_summary.json
```

This verification script:
- Analyzes all HDF5 files to count words with EEG data, frequency bands, and fixation information
- Automatically compares with any existing `extraction_summary_full.json` in the directory
- Shows differences between old and new summaries (useful for verifying consistency)
- Expected coverage is approximately 50-80% of words having EEG recordings

When an existing summary is found, the script will display changes like:
```
Differences from existing summary:
  • Total words: 154,173 → 154,173
  • Words with raw EEG: 107,831 (69.9%) → 107,831 (69.9%)
  • New files added: 2
  • Files with EEG count changes: 5
```
# ZuCo EEG Dataset Extraction Pipeline (from original dataset)

## Features

- ✅ **Dual Version Support**: Handles both ZuCo 1.0 and ZuCo 2.0 datasets
- ✅ **Sentence Context Preservation**: Maintains full sentence text and word positions for LLM alignment
- ✅ **Resume Capability**: Checkpoint system allows interrupting and resuming extraction
- ✅ **Efficient Storage**: Separate HDF5 files per subject-task with LZF compression
- ✅ **Progress Tracking**: Real-time progress bars for all operations
- ✅ **Robust Error Handling**: Gracefully handles missing data and format variations

## Prerequisites

- Python 3.8 (recommended for MATLAB compatibility) for HDF5 file creation. (Python 3.13+ verified compatible with existing HDF5)
- Conda or Miniconda
- ~50GB free disk space for full dataset
- ZuCo dataset files (download instructions below)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/extract_zuco_data.git
cd extract_zuco_data
```

### 2. Download ZuCo Dataset

Create the data directory structure and download the datasets:

```bash
# Create the data directory
mkdir -p zuco_data/zuco1.0
mkdir -p zuco_data/zuco2.0
```

#### Download ZuCo 1.0
1. Go to https://osf.io/q3zws/files/osfstorage
2. Download all folders:
   - `task1-SR/`
   - `task2-NR/`
   - `task3-TSR/`
3. Place them in `zuco_data/zuco1.0/`

#### Download ZuCo 2.0
1. Go to https://osf.io/2urht/files/osfstorage
2. Download all folders:
   - `task1 - NR/`
   - `task2 - TSR/`
   - `preprocessed/` (optional)
   - `scripts/` (contains useful reference scripts)
3. Place them in `zuco_data/zuco2.0/`

Your directory structure should look like:
```
extract_zuco_data/
├── zuco_data/
│   ├── zuco1.0/
│   │   ├── task1-SR/
│   │   │   └── Matlab files/
│   │   │       ├── results01_SR_ZAB.mat
│   │   │       ├── results02_SR_ZDM.mat
│   │   │       └── ...
│   │   ├── task2-NR/
│   │   │   └── Matlab files/
│   │   └── task3-TSR/
│   │       └── Matlab files/
│   └── zuco2.0/
│       ├── task1 - NR/
│       │   └── Matlab files/
│       │       ├── resultsYAC_NR.mat
│       │       ├── resultsYAG_NR.mat
│       │       └── ...
│       ├── task2 - TSR/
│       │   └── Matlab files/
│       └── scripts/
│           └── python_reader/
└── src/
    └── [all Python scripts]
```

### 3. Set Up Python Environment

#### Option A: Using the provided setup script
```bash
bash setup_environment.sh
```

#### Option B: Manual setup
```bash
# Create conda environment (Python 3.8 recommended)
conda create --prefix .conda/zuco_extract python=3.8
conda activate .conda/zuco_extract

# Or for Python 3.13+ compatibility
conda create --prefix .conda/zuco_py313 python=3.13
conda activate .conda/zuco_py313

# Install dependencies
pip install scipy numpy pandas matplotlib h5py
pip install tables pyarrow tqdm
```

## Usage

### Quick Test

Run a quick test to verify everything is working:

```bash
# Activate environment
conda activate .conda/zuco_extract

# Run test extraction (processes 4 sample files)
python src/test_extraction.py
```

This will create test files in `test_output/` and verify both ZuCo 1.0 and 2.0 extraction.

### Python 3.13 Compatibility Verification (Optional)

To verify the extracted data works with modern Python versions:

```bash
# Using Python 3.13 environment
conda activate .conda/zuco_py313

# Test HDF5 compatibility
python src/test_python313_compat.py

# Verify alignment across versions
python src/verify_alignment.py
```

### Full Extraction

Extract the complete dataset:

```bash
# Run full extraction with progress bars
python src/zuco_extraction_pipeline.py

# Monitor progress in another terminal
tail -f extraction_log.txt
```

**Expected runtime**: 30-45 minutes for all 72 files
**Output size**: ~5-10GB (compressed HDF5)



### Resume Interrupted Extraction

The pipeline automatically saves checkpoints. If interrupted, simply run the same command again:

```bash
python src/zuco_extraction_pipeline.py
# Will skip already completed files and resume from last checkpoint
```

## Output Structure

### File Organization

The pipeline creates separate HDF5 files for each subject-task combination:

```
extracted_data/
├── zuco1_ZAB_SR.h5       # ZuCo 1.0: 12 subjects × 3 tasks = 36 files
├── zuco1_ZAB_NR.h5
├── zuco1_ZAB_TSR.h5
├── ...
├── zuco2_YAC_NR.h5       # ZuCo 2.0: 18 subjects × 2 tasks = 36 files
├── zuco2_YAC_TSR.h5
├── ...
├── extraction_summary.json     # Statistics and metadata
└── extraction_checkpoint.json  # Resume tracking
```

### HDF5 Data Structure

Each HDF5 file contains word-level data with preserved sentence context:

```
zuco1_ZAB_SR.h5
├── word_00000/
│   ├── Attributes:
│   │   ├── word_content: "Henry"
│   │   ├── sentence_id: 0
│   │   ├── sentence_content: "Henry Ford, with his son..."
│   │   ├── word_index: 0
│   │   ├── task_name: "SR"
│   │   └── subject_id: "ZAB"
│   ├── raw_eeg         # Shape: (n_fixations, 105 electrodes)
│   ├── frequency_bands/
│   │   ├── FFD_t1      # Theta band 1 (4-6 Hz)
│   │   ├── FFD_t2      # Theta band 2 (6.5-8 Hz)
│   │   ├── FFD_a1      # Alpha band 1 (8.5-10 Hz)
│   │   ├── FFD_a2      # Alpha band 2 (10.5-13 Hz)
│   │   ├── FFD_b1      # Beta band 1 (13.5-18 Hz)
│   │   ├── FFD_b2      # Beta band 2 (18.5-30 Hz)
│   │   ├── FFD_g1      # Gamma band 1 (30.5-40 Hz)
│   │   └── FFD_g2      # Gamma band 2 (40-49.5 Hz)
│   └── eye_tracking/
│       ├── FFD         # First fixation duration
│       ├── TRT         # Total reading time
│       ├── GD          # Gaze duration
│       ├── GPT         # Go-past time
│       ├── SFD         # Single fixation duration
│       └── nFixations  # Number of fixations
├── word_00001/
└── ...
```

## Data Access Examples

### Python - Load extracted data

```python
import h5py
import numpy as np

# Load a specific subject-task file
with h5py.File('extracted_data/zuco1_ZAB_SR.h5', 'r') as f:
    # Access first word
    word_group = f['word_00000']
    
    # Get word and sentence information
    word = word_group.attrs['word_content']
    sentence = word_group.attrs['sentence_content']
    word_position = word_group.attrs['word_index']
    
    # Load EEG data
    raw_eeg = word_group['raw_eeg'][:]  # Shape: (n_fixations, 105)
    
    # Load frequency bands
    theta1 = word_group['frequency_bands/FFD_t1'][:]  # Shape: (105,)
    
    # Load eye-tracking metrics
    ffd = word_group['eye_tracking/FFD'][()]  # Scalar value
    
    print(f"Word: '{word}' at position {word_position}")
    print(f"Sentence: {sentence[:50]}...")
    print(f"Raw EEG shape: {raw_eeg.shape}")
```

### Reconstruct sentences for LLM processing

```python
import h5py
from collections import defaultdict

def get_sentences_from_file(filepath):
    """Reconstruct sentences from word-level data"""
    sentences = defaultdict(list)
    
    with h5py.File(filepath, 'r') as f:
        for word_key in f.keys():
            word_group = f[word_key]
            sent_id = word_group.attrs['sentence_id']
            word_idx = word_group.attrs['word_index']
            word_content = word_group.attrs['word_content']
            
            sentences[sent_id].append((word_idx, word_content))
    
    # Sort words by position and reconstruct
    reconstructed = {}
    for sent_id, words in sentences.items():
        words.sort(key=lambda x: x[0])
        reconstructed[sent_id] = ' '.join([w[1] for w in words])
    
    return reconstructed
```

## Dataset Information

### ZuCo 1.0
- **Subjects**: 12 native English speakers
- **Tasks**: 
  - SR: Sentiment Reading (400 sentences)
  - NR: Normal Reading (300 sentences)
  - TSR: Task-Specific Reading (407 sentences)
- **Known missing data**: ZDN, ZJS, ZPH, ZGW (partial recordings)

### ZuCo 2.0
- **Subjects**: 18 native English speakers
- **Tasks**:
  - NR: Normal Reading (349 sentences)
  - TSR: Task-Specific Reading (349 sentences)
- **Improvements**: Single-session recording (no session effects)

## Technical Details

### Requirements
- **Python 3.8+**: Optimal compatibility with MATLAB file formats (tested with 3.8 and 3.13)
- **Memory**: ~4GB RAM for processing
- **Storage**: ~50GB for raw data, ~10GB for extracted HDF5 files

### Performance
- **Extraction speed**: ~20-30 seconds per subject-task file
- **Compression**: LZF compression (5x faster than gzip)
- **Parallel processing**: Not implemented (files are independent, can run multiple instances)

## Troubleshooting

### Common Issues

1. **Memory errors during extraction**
   - Solution: The pipeline uses incremental saving to minimize memory usage
   - If issues persist, process fewer files at a time

2. **Missing data warnings**
   - Expected for subjects ZDN, ZJS, ZPH, ZGW in ZuCo 1.0
   - Pipeline handles these gracefully

3. **Deprecation warnings**
   - `nFixations` empty array warning is harmless and can be ignored

4. **HDF5 file corruption**
   - Delete the corrupted file and re-run extraction
   - Checkpoint system will resume from that file

## Citation

If you use this extraction pipeline in your research, please cite:

```bibtex
@article{hollenstein2018zuco,
  title={ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  author={Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius and Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  journal={Scientific Data},
  volume={5},
  pages={180291},
  year={2018}
}

@article{hollenstein2020zuco2,
  title={ZuCo 2.0: A dataset of physiological recordings during English and Dutch reading},
  author={Hollenstein, Nora and Troendle, Marius and Zhang, Ce and Langer, Nicolas},
  journal={arXiv preprint arXiv:2004.14254},
  year={2020}
}
```

## License

This extraction pipeline is provided under the MIT License. The ZuCo dataset itself has its own licensing terms - please refer to the official ZuCo documentation.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- ZuCo dataset creators at University of Zurich
- Official ZuCo 2.0 Python scripts provided helpful structure insights
