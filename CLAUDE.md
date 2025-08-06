# ZuCo Dataset Extraction Project Context

## Project Overview
Implementing a simplified version of the Goldstein et al. 2024 approach for mapping between EEG signals and language model embeddings using the ZuCo dataset. Goal is to extract and modernize word-level EEG data from Matlab .mat files into a robust Python pipeline.

## Current Status
- **Date Started**: 2025-08-06
- **Current Phase**: Full extraction complete
- **Environment**: Python 3.8 conda environment created in `.conda/zuco_extract`
- **Status**: 
  - ✅ Successfully extracted 256,838 words from ZuCo 1.0 (all subjects, all tasks)
  - ✅ Handles missing data gracefully (ZDN, ZJS, ZPH, ZGW known gaps)
  - ⏳ ZuCo 2.0 HDF5 structure identified but needs custom extraction
- **Performance**: ~10-15 min for full ZuCo 1.0 extraction, HDF5 save takes 5+ min due to compression

## Environment Setup

### Recommended Python Version
**Python 3.8** - Chosen for compatibility with:
- MATLAB 2016b .mat files
- ZuCo 1.0: MATLAB v7 format (scipy.io.loadmat)
- ZuCo 2.0: MATLAB v7.3 format (HDF5-based, requires h5py)
- Close to ZuCo benchmark's Python 3.7.16
- Well-tested scipy versions for older MATLAB formats

### Create Conda Environment
```bash
conda create -n zuco_extract python=3.8
conda activate zuco_extract
```

### Required Dependencies
```bash
pip install scipy numpy pandas matplotlib h5py
pip install tables  # For HDF5 support
pip install pyarrow  # For Parquet support
```

## Dataset Structure

### ZuCo 1.0
- **Location**: `zuco_data/zuco1.0/`
- **Subjects**: 12 (ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, ZPH)
- **Tasks**:
  - Task 1 (SR): Sentiment Reading - `task1-SR/Matlab files/`
  - Task 2 (NR): Normal Reading (relations) - `task2-NR/Matlab files/`
  - Task 3 (TSR): Task-Specific Reading - `task3-TSR/Matlab files/`
- **Known Issues**:
  - Missing data: ZDN (151-250, 400), ZJS (1-50), ZPH (51-100), ZGW (179-225)
  - Different sessions for different tasks (potential session effects)

### ZuCo 2.0
- **Location**: `zuco_data/zuco2.0/`
- **Subjects**: ~18 (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YMH, YMS, YRH, YRK, YRP, YSD, YSL, YTL)
- **Tasks**:
  - Task 1: Normal Reading (NR) - `task1 - NR/Matlab files/`
  - Task 2: Task-Specific Reading (TSR) - `task2 - TSR/Matlab files/`
- **Advantages**: Both paradigms recorded in single session (better for comparison)

## Expected Data Structure (from documentation)

### Word-Level Data Contains:
- **Text**: word content, sentence context, position
- **Raw EEG**: 105 electrode values per fixation
- **Frequency Bands**: 8 bands (t1, t2, a1, a2, b1, b2, g1, g2) × 105 electrodes
- **Eye-tracking Events**: FFD, TRT, GD, SFD, GPT, nFixations
- **Metadata**: subject ID, session info, task type

## Code Files Created

### 1. `zuco_reconnaissance.py`
- Explores both ZuCo 1.0 and 2.0 structures
- Documents differences between versions
- Identifies missing data patterns
- Creates structure report (saves to `zuco_structure_report.json`)

## TODO List
1. ✅ Explore zuco_data folder structure
2. ✅ Create reconnaissance script to probe .mat file structures
3. ✅ Run reconnaissance and analyze results
4. ✅ Set up conda environment with Python 3.8
5. ✅ Document structural differences between ZuCo 1.0 and 2.0
6. ✅ Identify missing data patterns and inconsistencies
7. ✅ Build unified extraction pipeline for both versions
8. ✅ Successfully extract full ZuCo 1.0 dataset
9. ⏳ Implement ZuCo 2.0 HDF5 extraction (different structure than expected)
10. ⏳ Optimize save performance (consider lzf compression or chunking)

## Key Findings from Reconnaissance

### Format Differences
- **ZuCo 1.0**: MATLAB v7 format - loads with `scipy.io.loadmat`
  - sentenceData is a numpy array of MATLAB structs
  - Each sentence has word array with detailed EEG features
  
- **ZuCo 2.0**: MATLAB v7.3 format (HDF5) - requires `h5py`
  - sentenceData is an HDF5 Group
  - Different structure, data organized as HDF5 datasets/groups
  
### Data Structure (ZuCo 1.0)
- 400 sentences in Task 1 (SR)
- 300 sentences in Task 2 (NR) 
- 407 sentences in Task 3 (TSR)
- Each sentence contains:
  - `word` array with word-level data
  - Frequency bands: t1, t2, a1, a2, b1, b2, g1, g2 (8 bands × 105 electrodes)
  - `rawData`: 105 × N timepoints
  - Eye-tracking: stored in word structures

## Next Actions
1. Set up conda environment with Python 3.9
2. Install dependencies
3. Run `zuco_reconnaissance.py` to explore data structures
4. Analyze the output report
5. Design unified extraction pipeline based on findings

## Key Technical Considerations
- **MATLAB Format Differences**:
  - ZuCo 1.0: MATLAB v7 format - use `scipy.io.loadmat`
  - ZuCo 2.0: MATLAB v7.3 format (HDF5) - use `h5py`
- **scipy.io.loadmat parameters**: Use `struct_as_record=False, squeeze_me=True` for better handling of MATLAB structs
- **Version differences**: Need version-aware logic to handle structural differences
- **Missing data**: Implement graceful handling of missing fixations/recordings
- **Output format**: Target HDF5 or Parquet for modern Python compatibility

## Commands to Run
```bash
# Activate environment
conda activate /Users/oshun/Documents/GitHub/extract_zuco_data/.conda/zuco_extract

# Run full extraction (with resume capability)
./.conda/zuco_extract/bin/python zuco_extraction_pipeline.py

# Run test extraction (quick validation)
./.conda/zuco_extract/bin/python test_extraction.py

# Check extraction results
./.conda/zuco_extract/bin/python -c "
import h5py
from pathlib import Path
for f in Path('extracted_data').glob('*.h5'):
    with h5py.File(f, 'r') as h:
        print(f'{f.name}: {h.attrs.get(\"n_words\", 0)} words')
"
```

## Pipeline Architecture (Refactored 2025-08-06)

### Key Features
1. **Separate HDF5 files** per subject-task (e.g., `zuco1_ZAB_SR.h5`, `zuco2_YAC_NR.h5`)
2. **Resume capability** with checkpoint tracking (`extraction_checkpoint.json`)
3. **Progress bars** using tqdm for all operations
4. **Optimized compression** using lzf instead of gzip (much faster)
5. **Modular functions** for testing and importing
6. **Incremental saving** to avoid memory issues

### File Naming Convention
- ZuCo 1.0: `zuco1_<SUBJECT>_<TASK>.h5` (e.g., `zuco1_ZAB_SR.h5`)
- ZuCo 2.0: `zuco2_<SUBJECT>_<TASK>.h5` (e.g., `zuco2_YAC_NR.h5`)

### Output Structure
```
extracted_data/
├── zuco1_ZAB_SR.h5       # Individual subject-task files
├── zuco1_ZAB_NR.h5
├── zuco1_ZAB_TSR.h5
├── zuco2_YAC_NR.h5        # ZuCo 2.0 files
├── zuco_metadata.parquet  # Combined metadata for quick queries
├── extraction_summary.json # Statistics
└── extraction_checkpoint.json # Resume tracking
```

### Data Structure in HDF5 Files
Each HDF5 file contains word-level data with **full sentence context**:
```
zuco1_ZAB_SR.h5
├── word_00000/
│   ├── Attributes:
│   │   ├── word_content: "Henry"
│   │   ├── sentence_id: 0
│   │   ├── sentence_content: "Henry Ford, with his son..."  # Full sentence
│   │   ├── word_index: 0  # Position in sentence
│   │   ├── task_name: "SR"
│   │   └── subject_id: "ZAB"
│   ├── raw_eeg  # (n_fixations, 105) array
│   ├── FFD_t1   # (105,) frequency band
│   └── ...
├── word_00001/
└── ...
```

### Important for LLM Alignment
- **Full context preserved**: Each word knows its sentence and position
- **Sentence tracking**: `sentence_id` links words to their sentence
- **Word order**: `word_index` preserves word position in sentence
- **Complete text**: `sentence_content` provides full context for embeddings
- **Easy grouping**: Can reconstruct sentences from words for LLM processing

## Known Issues
1. **ZuCo 2.0 extraction**: HDF5 structure completely different from ZuCo 1.0
   - Each sentence's words are grouped together in a single HDF5 group
   - Word content is in nested references that need special decoding
   - Solution found in official scripts: `zuco2.0/scripts/python_reader/`
2. **Deprecation warning**: Empty array in `nFixations` (harmless)

## ZuCo 2.0 Structure Discovery

### Key Differences from ZuCo 1.0
- **ZuCo 1.0**: Each sentence has a `word` array with individual word objects
- **ZuCo 2.0**: Each sentence has ONE `word` group containing ALL words together

### ZuCo 2.0 Word Data Structure
```
sentenceData/
├── word[0]/  # ALL words for sentence 0
│   ├── content[n]  # Array of references to word strings
│   ├── rawEEG[n]   # Raw EEG for each word
│   ├── FFD_t1[n]   # Frequency bands for each word
│   └── ...
├── word[1]/  # ALL words for sentence 1
│   └── ...
```

### Official Script Location
- Found in: `zuco2.0/scripts/python_reader/`
- Key files:
  - `read_matlab_files.py` - Shows how to read the data
  - `data_loading_helpers.py` - Contains `extract_word_level_data()` function
  - Uses `load_matlab_string()` to decode word content: `u''.join(chr(c) for c in matlab_object)`

## Notes for Next Session
- ZuCo 1.0 extraction working perfectly
- ZuCo 2.0 structure understood - need to implement based on official scripts
- Pipeline is modular and can be imported for custom workflows
- Test script validates extraction quality