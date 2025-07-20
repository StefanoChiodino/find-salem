#!/usr/bin/env python3
"""
Dataset splitting script for Find Salem project.
Splits Salem photos into train/test sets and collects matching other cat photos.
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import urllib.request
import json
import time

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory, excluding videos and other formats."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Skip system files
            if not file_path.name.startswith('.'):
                image_files.append(file_path)
    
    return image_files

def split_salem_photos(train_dir: Path, test_dir: Path, test_ratio: float = 0.2) -> Tuple[int, int]:
    """
    Split Salem photos between train and test directories.
    
    Args:
        train_dir: Training directory containing Salem photos
        test_dir: Test directory to move some photos to
        test_ratio: Fraction of photos to move to test (default 0.2 = 20%)
    
    Returns:
        Tuple of (remaining_in_train, moved_to_test)
    """
    # Get all image files
    image_files = get_image_files(train_dir)
    
    print(f"📊 Found {len(image_files)} image files in {train_dir}")
    
    if len(image_files) == 0:
        print("❌ No image files found!")
        return 0, 0
    
    # Calculate how many to move to test
    num_test = int(len(image_files) * test_ratio)
    if num_test == 0:
        num_test = 1  # At least move one file
    
    # Randomly select files for test set
    random.shuffle(image_files)
    test_files = image_files[:num_test]
    
    # Ensure test directory exists
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files to test directory
    moved_count = 0
    for file_path in test_files:
        try:
            destination = test_dir / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_count += 1
            print(f"✅ Moved {file_path.name} to test set")
        except Exception as e:
            print(f"❌ Error moving {file_path.name}: {e}")
    
    remaining_count = len(image_files) - moved_count
    
    print(f"\n📈 Dataset split complete:")
    print(f"   Training set: {remaining_count} images")
    print(f"   Test set: {moved_count} images")
    
    return remaining_count, moved_count

def download_other_cat_photos(train_count: int, test_count: int) -> bool:
    """
    Download other black cat photos to match Salem's dataset size.
    
    Args:
        train_count: Number of photos needed for training set
        test_count: Number of photos needed for test set
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🔍 Need to collect {train_count} training + {test_count} test photos of other cats")
    
    # Create directories
    train_other_dir = Path("data/train/other_cats")
    test_other_dir = Path("data/test/other_cats")
    train_other_dir.mkdir(parents=True, exist_ok=True)
    test_other_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample URLs for black cat photos (these would need to be real URLs)
    # For now, we'll create placeholder instructions
    
    print("\n📝 Manual collection needed:")
    print(f"   Please collect {train_count} photos for: {train_other_dir.absolute()}")
    print(f"   Please collect {test_count} photos for: {test_other_dir.absolute()}")
    
    print("\n🌐 Recommended sources:")
    print("   • Unsplash: https://unsplash.com/s/photos/black-cat")
    print("   • Pexels: https://www.pexels.com/search/black%20cat/")
    print("   • Pixabay: https://pixabay.com/images/search/black%20cat/")
    
    print("\n💡 Tips:")
    print("   • Download high-quality images (>400x400 pixels)")
    print("   • Ensure good variety in poses, lighting, and backgrounds")
    print("   • Avoid images with multiple cats or unclear subjects")
    print("   • Save as JPG format with descriptive names")
    
    return True

def create_collection_script():
    """Create a simple script to help with manual collection."""
    script_content = '''#!/bin/bash
# Quick collection helper script

echo "🐱 Black Cat Photo Collection Helper"
echo "=================================="

echo "📊 Current dataset status:"
echo "Salem training photos: $(ls -1 data/train/salem/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Salem test photos: $(ls -1 data/test/salem/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Other cats training: $(ls -1 data/train/other_cats/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Other cats test: $(ls -1 data/test/other_cats/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"

echo ""
echo "🌐 Quick download commands (replace URLs with actual image URLs):"
echo "cd data/train/other_cats"
echo "curl -o black_cat_01.jpg 'https://example.com/black_cat_1.jpg'"
echo "curl -o black_cat_02.jpg 'https://example.com/black_cat_2.jpg'"
echo "# ... add more URLs as needed"
'''
    
    with open("collect_photos.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("collect_photos.sh", 0o755)
    print("✅ Created collect_photos.sh helper script")

def main():
    """Main function to split dataset and collect other cat photos."""
    print("🐱 Find Salem Dataset Splitter")
    print("=" * 40)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    salem_train_dir = Path("data/train/salem")
    salem_test_dir = Path("data/test/salem")
    
    # Check if Salem training directory exists
    if not salem_train_dir.exists():
        print(f"❌ Salem training directory not found: {salem_train_dir}")
        print("Please make sure you've added Salem photos to data/train/salem/")
        return
    
    # Split Salem photos
    print("🔄 Splitting Salem photos into train/test sets...")
    train_count, test_count = split_salem_photos(salem_train_dir, salem_test_dir)
    
    if train_count == 0 and test_count == 0:
        print("❌ No photos to split!")
        return
    
    # Collect other cat photos
    print("\n🔄 Setting up collection for other cat photos...")
    download_other_cat_photos(train_count, test_count)
    
    # Create helper script
    create_collection_script()
    
    print("\n🎉 Dataset preparation complete!")
    print("\n📋 Summary:")
    print(f"   Salem training photos: {train_count}")
    print(f"   Salem test photos: {test_count}")
    print(f"   Need {train_count} other cat training photos")
    print(f"   Need {test_count} other cat test photos")
    
    print("\n🚀 Next steps:")
    print("1. Collect other black cat photos using the provided sources")
    print("2. Run ./collect_photos.sh to check dataset balance")
    print("3. Start training when both categories have sufficient photos")

if __name__ == "__main__":
    main()
