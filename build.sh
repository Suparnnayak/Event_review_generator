#!/usr/bin/env bash

# Exit on error
set -o errexit

# Update the package list and install the gfortran compiler and other build tools.
# This ensures gfortran is available before any Python packages try to compile.
echo "--> Installing system dependencies..."
apt-get update -y
apt-get install -y gfortran build-essential

# Install Python dependencies from requirements.txt
echo "--> Installing Python dependencies..."
pip install -r requirements.txt

# Download necessary NLTK data
echo "--> Downloading NLTK data..."
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

echo "--> Build process completed successfully!"
