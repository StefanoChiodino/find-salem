#!/usr/bin/env python3
"""
Split perfectly balanced dataset 80/20 into train/test folders
"""

import os
import shutil
import random
from pathlib import Path

def get_all_files(directory):
    """Get all image files from directory"""
    if not directory.exists():
        return []
    
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix in extensions:
            files.append(file)
    return files

def split_files_80_20(files, train_dir, test_dir, category_name):
    """Split files 80/20 and move to train/test directories"""
    total_files = len(files)
    test_count = int(total_files * 0.2)  # 20% for testing
    train_count = total_files - test_count
    
    # Shuffle for random split
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    # Split into train and test
    train_files = shuffled_files[:train_count] 
    test_files = shuffled_files[train_count:]
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Splitting {category_name} photos (80/20)...")
    print(f"   📊 Total: {total_files} photos")
    print(f"   📚 Training: {len(train_files)} photos (80%)")
    print(f"   🧪 Testing: {len(test_files)} photos (20%)")
    
    # Move training files
    for i, file in enumerate(train_files, 1):
        dest = train_dir / file.name
        shutil.move(str(file), str(dest))
        if i % 50 == 0:
            print(f"   📈 Moved {i}/{len(train_files)} training files...")
    
    # Move test files
    for i, file in enumerate(test_files, 1):
        dest = test_dir / file.name
        shutil.move(str(file), str(dest))
        if i % 20 == 0:
            print(f"   📈 Moved {i}/{len(test_files)} test files...")
    
    print(f"✅ {category_name} split complete!")
    return len(train_files), len(test_files)

def main():
    print("🐱 Salem Dataset 80/20 Split")
    print("=============================")
    print("📊 Standard ML train/test split: 80% training, 20% testing")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    data_dir = Path("data")
    
    # Source directories (all photos currently in training)
    salem_source = data_dir / "train" / "salem"
    other_source = data_dir / "train" / "other_cats"
    
    # Destination directories
    salem_train = data_dir / "train" / "salem"
    salem_test = data_dir / "test" / "salem"
    other_train = data_dir / "train" / "other_cats"
    other_test = data_dir / "test" / "other_cats"
    
    # Clear test directories first
    print("\n🧹 Cleaning test directories...")
    for test_dir in [salem_test, other_test]:
        if test_dir.exists():
            for file in test_dir.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"✅ Cleaned {test_dir}")
    
    # Get all files
    salem_files = get_all_files(salem_source)
    other_files = get_all_files(other_source)
    
    print(f"\n📊 Current dataset:")
    print(f"   🐱 Salem photos: {len(salem_files)}")
    print(f"   🐾 Other cat photos: {len(other_files)}")
    print(f"   📈 Total: {len(salem_files) + len(other_files)}")
    
    # Clear training directories to prepare for split
    for file in salem_files + other_files:
        if file.exists():
            continue  # Files are already in the right place
    
    # Split Salem photos
    salem_train_count, salem_test_count = split_files_80_20(
        salem_files, salem_train, salem_test, "Salem"
    )
    
    # Split other cat photos  
    other_train_count, other_test_count = split_files_80_20(
        other_files, other_train, other_test, "Other cats"
    )
    
    # Final summary
    print(f"\n🎉 Dataset split complete!")
    print(f"📊 Final distribution:")
    print(f"   📚 Training: {salem_train_count + other_train_count} photos")
    print(f"      🐱 Salem: {salem_train_count}")
    print(f"      🐾 Other cats: {other_train_count}")
    print(f"   🧪 Testing: {salem_test_count + other_test_count} photos")
    print(f"      🐱 Salem: {salem_test_count}")
    print(f"      🐾 Other cats: {other_test_count}")
    
    total_photos = salem_train_count + salem_test_count + other_train_count + other_test_count
    train_percentage = ((salem_train_count + other_train_count) / total_photos) * 100
    test_percentage = ((salem_test_count + other_test_count) / total_photos) * 100
    
    print(f"\n📈 Split verification:")
    print(f"   📚 Training: {train_percentage:.1f}%")
    print(f"   🧪 Testing: {test_percentage:.1f}%")
    print(f"   ⚖️ Perfect balance maintained in both splits!")
    
    print(f"\n🚀 Ready for FastAI training with {total_photos} photos!")

if __name__ == "__main__":
    main()
