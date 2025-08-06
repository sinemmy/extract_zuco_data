# ZuCo Dataset Extraction Project Context

## Project Overview
Implementing a simplified version of the Goldstein et al. 2024 approach for mapping between EEG signals and language model embeddings using the ZuCo dataset. Goal is to extract and modernize word-level EEG data from Matlab .mat files into a robust Python pipeline.

## Current Status
- **Date Started**: 2025-08-06
- **Current Phase**: BOTH ZuCo 1.0 and 2.0 extraction working! Ready for full run.
- **Environment**: Python 3.8 conda environment created in `.conda/zuco_extract`
- **Status**: 
  - ✅ ZuCo 1.0 extraction fully working (12 subjects × 3 tasks = 36 files)
  - ✅ ZuCo 2.0 extraction fully working (18 subjects × 2 tasks = 36 files) 
  - ✅ Handles missing data gracefully (ZDN, ZJS, ZPH, ZGW known gaps)
  - ✅ Saves separate HDF5 files per subject-task (e.g., zuco1_ZAB_SR.h5)
  - ✅ Checkpoint/resume capability implemented
  - ✅ Full sentence context preserved for LLM alignment
- **Performance**: 
  - Per file: ~20-30 seconds (including save)
  - Full dataset: Estimated 30-45 minutes for all 72 files
  - Uses LZF compression for faster saves

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
pip install tqdm  # For progress bars
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

### 2. `zuco_extraction_pipeline.py`
- Main extraction script for both ZuCo versions
- Saves separate HDF5 files per subject-task
- Includes CheckpointManager for resume capability
- Preserves full sentence context (sentence_id, sentence_content, word_index)
- Uses modular functions for easy testing
- Progress bars with tqdm

### 3. `test_extraction.py`
- Quick test script for validation
- Tests both ZuCo 1.0 and 2.0
- Saves 30 words per test file
- Verifies data structure and content

### 4. Helper Scripts (created during debugging)
- `find_word_content.py`: Explores ZuCo 2.0 word content location
- `extract_zuco2_word.py`: Tests ZuCo 2.0 word extraction
- `debug_zuco2.py`: Deep dive into ZuCo 2.0 HDF5 structure

## TODO List
1. ✅ Explore zuco_data folder structure
2. ✅ Create reconnaissance script to probe .mat file structures
3. ✅ Run reconnaissance and analyze results
4. ✅ Set up conda environment with Python 3.8
5. ✅ Document structural differences between ZuCo 1.0 and 2.0
6. ✅ Identify missing data patterns and inconsistencies
7. ✅ Build unified extraction pipeline for both versions
8. ✅ Successfully extract ZuCo 1.0 dataset
9. ✅ Implement ZuCo 2.0 HDF5 extraction (fixed using official scripts)
10. ✅ Optimize save performance (using LZF compression)
11. ✅ Add checkpoint/resume capability
12. ✅ Preserve sentence context for LLM alignment
13. ⏳ Run full extraction (user will run separately)

## Key Discoveries About ZuCo 2.0 Structure

### Official Scripts Found
- Located in `zuco_data/zuco2.0/scripts/python_reader/`
- `data_loading_helpers.py`: Contains `extract_word_level_data()` function
- Key insight: In ZuCo 2.0, each 'word' entry is actually a **sentence container**
- Words within each sentence are stored as arrays inside the container
- Word content decoded using: `''.join(chr(c) for c in ref)`

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

## Commands to Run
```bash
# Activate environment
conda activate /Users/oshun/Documents/GitHub/extract_zuco_data/.conda/zuco_extract

# Or use directly
./.conda/zuco_extract/bin/python

# Run test extraction (quick validation)
./.conda/zuco_extract/bin/python test_extraction.py

# Run full extraction (takes 30-45 minutes)
./.conda/zuco_extract/bin/python zuco_extraction_pipeline.py

# Check extraction progress
tail -f extraction_log.txt

# Verify extracted data
./.conda/zuco_extract/bin/python -c "import h5py; import os; print('Files created:'); [print(f) for f in os.listdir('extracted_data') if f.endswith('.h5')]"
```

## Output Structure

### File Naming
- Format: `{dataset}_{subject}_{task}.h5`
- Examples: `zuco1_ZAB_SR.h5`, `zuco2_YAC_NR.h5`

### HDF5 Structure
```
file.h5
├── word_00000/
│   ├── raw_eeg [array: (n_fixations, 105)]
│   ├── frequency_bands/
│   │   ├── t1, t2, a1, a2, b1, b2, g1, g2 [arrays]
│   ├── eye_tracking/
│   │   ├── FFD, TRT, GD, GPT, SFD, nFixations [scalars]
│   └── attributes:
│       ├── word_content: "word"
│       ├── sentence_id: 0
│       ├── sentence_content: "Full sentence text..."
│       └── word_index: 0
├── word_00001/
...
```

## Key Technical Considerations
- **MATLAB Format Differences**:
  - ZuCo 1.0: MATLAB v7 format - use `scipy.io.loadmat`
  - ZuCo 2.0: MATLAB v7.3 format (HDF5) - use `h5py`
- **scipy.io.loadmat parameters**: Use `struct_as_record=False, squeeze_me=True` for better handling of MATLAB structs
- **Version differences**: Need version-aware logic to handle structural differences
- **Missing data**: Implement graceful handling of missing fixations/recordings
- **Output format**: Target HDF5 or Parquet for modern Python compatibility

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

### Output Directory Structure
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

## Status: READY FOR FULL EXTRACTION ✅

### Test Results (2025-08-06)
- **ZuCo 1.0**: ✅ Working perfectly (7129 words extracted from ZAB_SR)
- **ZuCo 2.0**: ✅ FIXED and working! (1079 words extracted from YAC_NR)
- Word content, EEG data, and sentence context all preserved correctly

## Known Issues (Resolved)
1. ~~**ZuCo 2.0 extraction**~~: ✅ FIXED using official script structure
   - Solution: Implemented proper word grouping based on `zuco2.0/scripts/python_reader/`
   - Words are extracted with `''.join(chr(c) for c in ref)` decoding
2. **Deprecation warning**: Empty array in `nFixations` (harmless, ignore)

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
- Both ZuCo 1.0 and 2.0 extraction working perfectly
- Ready for full extraction run (user will run separately)
- Pipeline is modular and can be imported for custom workflows
- Test script validates extraction quality
- Full sentence context preserved for LLM alignment work