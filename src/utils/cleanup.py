#!/usr/bin/env python3
"""
Dataset cleanup utility for FastAI training.
Removes any corrupted or missing image files.
"""

import os
from pathlib import Path
from PIL import Image

def check_image_file(image_path):
    """Check if an image file is valid and readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image is valid
        return True
    except Exception as e:
        print(f"❌ Corrupted image: {image_path} - {e}")
        return False

def cleanup_dataset(data_dir="data"):
    """Clean up dataset by removing corrupted/missing images"""
    print("🧹 Cleaning up dataset...")
    print("=" * 40)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    total_files = 0
    corrupted_files = 0
    
    # Check all subdirectories
    for subset in ['train', 'test']:
        subset_path = data_path / subset
        if not subset_path.exists():
            continue
            
        print(f"\n📁 Checking {subset} directory...")
        
        for category in ['salem', 'other_cats']:
            category_path = subset_path / category
            if not category_path.exists():
                continue
            
            print(f"  📂 {category}...")
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(category_path.glob(f"*{ext}"))
            
            category_total = len(image_files)
            category_corrupted = 0
            
            print(f"     Found {category_total} image files")
            
            # Check each image
            for image_path in image_files:
                total_files += 1
                
                if not image_path.exists():
                    print(f"❌ Missing file: {image_path}")
                    corrupted_files += 1
                    category_corrupted += 1
                    continue
                
                if not check_image_file(image_path):
                    # Remove corrupted file
                    try:
                        os.remove(image_path)
                        print(f"🗑️ Removed corrupted file: {image_path}")
                        corrupted_files += 1
                        category_corrupted += 1
                    except Exception as e:
                        print(f"❌ Could not remove {image_path}: {e}")
            
            valid_files = category_total - category_corrupted
            print(f"     ✅ {valid_files} valid files, ❌ {category_corrupted} corrupted/removed")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Total files checked: {total_files}")
    print(f"   Corrupted/removed: {corrupted_files}")
    print(f"   Valid files: {total_files - corrupted_files}")
    
    if corrupted_files == 0:
        print("✅ Dataset is clean!")
    else:
        print(f"🧹 Cleaned up {corrupted_files} problematic files")

if __name__ == "__main__":
    cleanup_dataset()
