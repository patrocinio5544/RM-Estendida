#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# PI-CAI 220 - MedNeXt Dataset Downloader (PRODUCTION)
# =========================================================

FILE_ID="1zU7dQcDF65Ag2bDZ-5neL4raTte8SawN"
FILE_NAME="pi_cai220_mednext.tar.zst"

DEST_DIR="data/nnUNet_raw/nnUNet_raw_data/Task220_PI-CAI"
mkdir -p "$DEST_DIR"

echo "======================================"
echo "Downloading PI-CAI MedNeXt dataset..."
echo "======================================"

# download
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$FILE_NAME"

echo "======================================"
echo "Extracting dataset..."
echo "======================================"

tar -I zstd -xf "$FILE_NAME" -C data/nnUNet_raw/nnUNet_raw_data/

echo "======================================"
echo "Cleaning up..."
echo "======================================"

rm -f "$FILE_NAME"

echo "======================================"
echo "DONE ✔ Dataset ready for MedNeXt"
echo "Location: $DEST_DIR"
echo "======================================"
