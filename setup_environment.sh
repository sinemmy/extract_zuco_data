#!/bin/bash

# Setup script for ZuCo data extraction project

echo "Setting up ZuCo data extraction environment..."
echo "============================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment with Python 3.8
echo "Creating conda environment 'zuco_extract' with Python 3.8..."
conda create -n zuco_extract python=3.8 -y

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate zuco_extract

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "============================================"
echo "To activate the environment, run:"
echo "  conda activate zuco_extract"
echo ""
echo "Then you can run the reconnaissance script:"
echo "  python zuco_reconnaissance.py"