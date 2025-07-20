#!/usr/bin/env python3
"""
Simple Dataset Splitter for Salem Cat Project
Properly splits balanced dataset into 80/20 train/test
"""

import shutil
import random
from pathlib import Path
from typing import List

def get_all_images_from_category(category: str) -> List[Path]:
    """Get all images from both train and test folders for a category"""
    images = []
    
    # Check train folder
    train_path = Path(f"data/train/{category}")
    if train_path.exists():
        images.extend([f for f in train_path.glob("*.jpg") if f.is_file()])
        images.extend([f for f in train_path.glob("*.JPG") if f.is_file()])
        images.extend([f for f in train_path.glob("*.jpeg") if f.is_file()])
    
    # Check test folder  
    test_path = Path(f"data/test/{category}")
    if test_path.exists():
        images.extend([f for f in test_path.glob("*.jpg") if f.is_file()])
        images.extend([f for f in test_path.glob("*.JPG") if f.is_file()])
        images.extend([f for f in test_path.glob("*.jpeg") if f.is_file()])
    
    return images

def split_and_move_images(category: str, images: List[Path]):
    """Split images 80/20 and move to proper train/test folders"""
    
    # Shuffle images for random split
    random.shuffle(images)
    
    # Calculate split (80% train, 20% test)
    total_count = len(images)
    train_count = int(total_count * 0.8)
    test_count = total_count - train_count
    
    print(f"📊 {category}: {total_count} total → {train_count} train, {test_count} test")
    
    # Create clean directories
    train_dir = Path(f"data/train/{category}")
    test_dir = Path(f"data/test/{category}")
    
    # Clear existing images (but keep .gitkeep files)
    for folder in [train_dir, test_dir]:
        if folder.exists():
            for img_file in folder.glob("*.jpg"):
                img_file.unlink()
            for img_file in folder.glob("*.JPG"):
                img_file.unlink()
            for img_file in folder.glob("*.jpeg"):
                img_file.unlink()
    
    # Split images
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    # Move train images
    for i, img_path in enumerate(train_images):
        ext = img_path.suffix.lower()
        new_name = f"{category}_{i:03d}{ext}"
        new_path = train_dir / new_name
        shutil.copy2(img_path, new_path)
    
    # Move test images  
    for i, img_path in enumerate(test_images):
        ext = img_path.suffix.lower()
        new_name = f"{category}_test_{i:03d}{ext}"
        new_path = test_dir / new_name
        shutil.copy2(img_path, new_path)
    
    return train_count, test_count

def main():
    """Split dataset into proper 80/20 train/test"""
    print("🐱 Simple Dataset Splitter for Salem Cat Project")
    print("=" * 50)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    results = {}
    
    # Process Salem photos
    print("\n📸 Processing Salem photos...")
    salem_images = get_all_images_from_category("salem")
    if salem_images:
        train_count, test_count = split_and_move_images("salem", salem_images)
        results["salem"] = {"train": train_count, "test": test_count}
    else:
        print("⚠️ No Salem photos found!")
        results["salem"] = {"train": 0, "test": 0}
    
    # Process other cat photos
    print("\n📸 Processing other cat photos...")
    other_images = get_all_images_from_category("other_cats")
    if other_images:
        train_count, test_count = split_and_move_images("other_cats", other_images)
        results["other_cats"] = {"train": train_count, "test": test_count}
    else:
        print("⚠️ No other cat photos found!")
        results["other_cats"] = {"train": 0, "test": 0}
    
    # Summary
    print(f"\n🎉 Dataset splitting complete!")
    print(f"📊 Final dataset split:")
    
    total_train = 0
    total_test = 0
    
    for category, counts in results.items():
        train_count = counts["train"]
        test_count = counts["test"]
        total_count = train_count + test_count
        
        if total_count > 0:
            train_pct = (train_count / total_count) * 100
            test_pct = (test_count / total_count) * 100
            print(f"   {category}: {train_count} train ({train_pct:.1f}%), {test_count} test ({test_pct:.1f}%)")
        else:
            print(f"   {category}: No images")
        
        total_train += train_count
        total_test += test_count
    
    total_all = total_train + total_test
    if total_all > 0:
        print(f"   TOTAL: {total_train} train ({(total_train/total_all)*100:.1f}%), {total_test} test ({(total_test/total_all)*100:.1f}%)")
        print(f"📈 Dataset ready for training with {total_all} images!")
    else:
        print("⚠️ No images found in dataset!")

if __name__ == "__main__":
    main()
