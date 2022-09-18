#!/bin/bash
# This script collect training data from a given directory by looking
# for all the files in that directory that end wih ".mid" and converting
# them to a five-track pianoroll dataset.
# Usage: ./generate_data.sh [INPUT_DIR] [OUTPUT_FILENAME]
python "../preprocessing/collect_data.py" -i "F:\Projets\Music Generator\datasets\lpd_5_cleansed" -o "../data/lpd5_cleansed"