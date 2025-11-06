#!/usr/bin/env python3
"""
Script to fetch and organize U-CALTECH and U-CIFAR datasets from Google Drive.
Downloads UHSR.zip, extracts nested zips, and organizes .npz files into train/test splits.
"""

import os
import zipfile
import shutil
import argparse
import re
from pathlib import Path
from typing import Tuple
import gdown

# Google Drive file ID
UHSR_ZIP_ID = "1zLVEX-Q3rvbe2nZvwF5ZCTjyX22VzxED"
UHSR_ZIP_URL = f"https://drive.google.com/uc?id={UHSR_ZIP_ID}"


def download_from_gdrive(url: str, output_path: str) -> None:
    """Download file from Google Drive."""
    print(f"Downloading from Google Drive to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to download {output_path}")
    print(f"Downloaded successfully: {output_path}")


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def _extract_number(filename: str) -> int:
    """
    Extract numeric value from filename for proper numerical sorting.
    Handles patterns like '0.npz', '123.npz', 'file_456.npz', etc.
    """
    # Extract all numbers from filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Use the first number found (or largest if multiple)
        # For names like '123.npz', this returns 123
        return int(numbers[0])
    # If no number found, return -1 to put at the end
    return -1


def split_files(src_dir: str, train_dir: str, test_dir: str) -> Tuple[int, int]:
    """
    Split .npz files from src_dir into train and test directories.
    Files 0-4999 go to train, files 5000+ go to test.
    Files are sorted numerically by name to ensure correct ordering.
    
    Args:
        src_dir: Source directory containing .npz files
        train_dir: Directory to move training files to
        test_dir: Directory to move test files to
    
    Returns:
        Tuple of (num_train_files, num_test_files)
    """
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all .npz files and sort numerically by name
    npz_files = [f for f in os.listdir(src_dir) if f.endswith('.npz')]
    npz_files = sorted(npz_files, key=_extract_number)
    total_files = len(npz_files)
    
    if total_files == 0:
        raise ValueError(f"No .npz files found in {src_dir}")
    
    # Split: indices 0-4999 for train, 5000+ for test
    train_files = npz_files[0:5000]
    test_files = npz_files[5000:]
    
    train_count = len(train_files)
    test_count = len(test_files)
    
    print(f"Found {total_files} .npz files. Splitting: {train_count} train (indices 0-4999), {test_count} test (indices 5000+)")
    
    # Move files to respective directories
    for file in train_files:
        shutil.move(
            os.path.join(src_dir, file),
            os.path.join(train_dir, file)
        )
    
    for file in test_files:
        shutil.move(
            os.path.join(src_dir, file),
            os.path.join(test_dir, file)
        )
    
    print(f"Moved {len(train_files)} files to {train_dir}")
    print(f"Moved {len(test_files)} files to {test_dir}")
    
    return train_count, test_count


def clean_up_zips(zip_files: list) -> None:
    """Remove zip files and temporary directories."""
    print("\nCleaning up zip files...")
    for zip_file in zip_files:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"Removed {zip_file}")
    
    # Remove temporary extraction directories
    temp_dirs = ['UHSR', 'U-CALTECH', 'U-CIFAR']
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")


def main(keep_zips: bool = False):
    """
    Main function to fetch and organize data.
    
    Args:
        keep_zips: If True, keep zip files instead of deleting them
    """
    print("=" * 60)
    print("SpikeCLIP Data Fetching Script")
    print("=" * 60)
    
    # Step 1: Download UHSR.zip
    uhsr_zip = "UHSR.zip"
    if not os.path.exists(uhsr_zip):
        download_from_gdrive(UHSR_ZIP_URL, uhsr_zip)
    else:
        print(f"{uhsr_zip} already exists, skipping download")
    
    # Step 2: Extract UHSR.zip
    extract_zip(uhsr_zip, ".")
    
    # Verify UHSR directory exists
    if not os.path.exists("UHSR"):
        raise FileNotFoundError("UHSR directory not found after extraction")
    
    # Step 3: Find and extract nested zips
    ucaltech_zip = os.path.join("UHSR", "U-CALTECH.zip")
    ucifar_zip = os.path.join("UHSR", "U-CIFAR.zip")
    
    if not os.path.exists(ucaltech_zip):
        raise FileNotFoundError(f"{ucaltech_zip} not found in UHSR directory")
    if not os.path.exists(ucifar_zip):
        raise FileNotFoundError(f"{ucifar_zip} not found in UHSR directory")
    
    # Extract U-CALTECH.zip
    extract_zip(ucaltech_zip, ".")
    
    # Extract U-CIFAR.zip
    extract_zip(ucifar_zip, ".")
    
    # Step 4: Verify extracted directories
    if not os.path.exists("U-CALTECH"):
        raise FileNotFoundError("U-CALTECH directory not found after extraction")
    if not os.path.exists("U-CIFAR"):
        raise FileNotFoundError("U-CIFAR directory not found after extraction")
    
    # Step 5: Create data directory structure
    data_dir = Path("data")
    ucaltech_train = data_dir / "u-caltech" / "train"
    ucaltech_test = data_dir / "u-caltech" / "test"
    ucifar_train = data_dir / "u-cifar" / "train"
    ucifar_test = data_dir / "u-cifar" / "test"
    
    print("\n" + "=" * 60)
    print("Organizing U-CALTECH data...")
    print("=" * 60)
    split_files("U-CALTECH", str(ucaltech_train), str(ucaltech_test))
    
    print("\n" + "=" * 60)
    print("Organizing U-CIFAR data...")
    print("=" * 60)
    split_files("U-CIFAR", str(ucifar_train), str(ucifar_test))
    
    # Step 6: Clean up
    if not keep_zips:
        zip_files = [uhsr_zip, ucaltech_zip, ucifar_zip]
        clean_up_zips(zip_files)
    else:
        print("\nKeeping zip files as requested")
    
    print("\n" + "=" * 60)
    print("Data organization complete!")
    print("=" * 60)
    print(f"U-CALTECH data: {ucaltech_train} (train), {ucaltech_test} (test)")
    print(f"U-CIFAR data: {ucifar_train} (train), {ucifar_test} (test)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and organize U-CALTECH and U-CIFAR datasets. "
                    "Files are split with indices 0-4999 for train and the rest for test."
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip files instead of deleting them"
    )
    
    args = parser.parse_args()
    
    main(keep_zips=args.keep_zips)

