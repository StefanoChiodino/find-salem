#!/usr/bin/env python3
"""
Safe Dataset Splitter - NEVER touches Salem photos
Only splits other cat photos into proper 80/20 train/test split
"""

import shutil
import random
from pathlib import Path
from typing import List

def safe_split_other_cats():
    """Safely split only other cat photos into 80/20 train/test - NEVER touches Salem photos"""
    print("🐱 Safe Dataset Splitter")
    print("⚠️  NEVER touches Salem photos - only splits other cat photos!")
    print("=" * 60)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Only work with other cat photos
    train_other_dir = Path("data/train/other_cats")
    test_other_dir = Path("data/test/other_cats")
    
    # Ensure test directory exists
    test_other_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all other cat photos currently in training folder
    if not train_other_dir.exists():
        print("❌ No training other_cats directory found!")
        return
    
    all_other_cat_photos = [f for f in train_other_dir.glob("*.jpg") if f.is_file()]
    
    if not all_other_cat_photos:
        print("❌ No other cat photos found in training folder!")
        return
    
    print(f"📊 Found {len(all_other_cat_photos)} other cat photos to split")
    
    # Shuffle for random split
    random.shuffle(all_other_cat_photos)
    
    # Calculate split (20% for testing)
    total_count = len(all_other_cat_photos)
    test_count = max(1, int(total_count * 0.2))  # At least 1 for testing
    train_count = total_count - test_count
    
    print(f"🎯 Target split: {train_count} train ({train_count/total_count*100:.1f}%), {test_count} test ({test_count/total_count*100:.1f}%)")
    
    # Move photos to test folder
    test_photos = all_other_cat_photos[:test_count]
    
    print(f"\n📤 Moving {len(test_photos)} photos to test folder...")
    
    for i, photo_path in enumerate(test_photos):
        # Create new filename for test folder
        new_name = f"other_cat_test_{i:03d}.jpg"
        new_path = test_other_dir / new_name
        
        # Move to test folder
        shutil.move(str(photo_path), str(new_path))
        print(f"   ✅ Moved {photo_path.name} → {new_name}")
    
    # Rename remaining training photos for consistency
    remaining_train_photos = [f for f in train_other_dir.glob("*.jpg") if f.is_file()]
    
    print(f"\n📝 Renaming {len(remaining_train_photos)} remaining training photos...")
    
    # Create temporary names to avoid conflicts
    temp_names = []
    for i, photo_path in enumerate(remaining_train_photos):
        temp_name = f"temp_other_cat_{i:03d}.jpg"
        temp_path = train_other_dir / temp_name
        shutil.move(str(photo_path), str(temp_path))
        temp_names.append(temp_path)
    
    # Rename to final names
    for i, temp_path in enumerate(temp_names):
        final_name = f"other_cat_{i:03d}.jpg" 
        final_path = train_other_dir / final_name
        shutil.move(str(temp_path), str(final_path))
    
    # Final count verification
    final_train_count = len([f for f in train_other_dir.glob("*.jpg") if f.is_file()])
    final_test_count = len([f for f in test_other_dir.glob("*.jpg") if f.is_file()])
    
    print(f"\n🎉 Dataset split complete!")
    print(f"📊 Final other cat photo distribution:")
    print(f"   Training: {final_train_count} photos")
    print(f"   Testing: {final_test_count} photos")
    print(f"   Total: {final_train_count + final_test_count} photos")
    
    # Show Salem photos remain untouched
    salem_train_count = len([f for f in Path("data/train/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    salem_test_count = len([f for f in Path("data/test/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    
    print(f"\n✅ Salem photos remain completely untouched:")
    print(f"   Salem training: {salem_train_count} photos")
    print(f"   Salem testing: {salem_test_count} photos")
    print(f"   Salem total: {salem_train_count + salem_test_count} photos")
    
    if final_train_count > 0 and final_test_count > 0:
        print(f"\n🚀 Dataset ready for training!")
    else:
        print(f"\n⚠️  Need more other cat photos for proper training")
    
    return {
        'other_train': final_train_count,
        'other_test': final_test_count,
        'salem_train': salem_train_count,
        'salem_test': salem_test_count
    }

def main():
    """Main function to safely split dataset"""
    results = safe_split_other_cats()
    
    if results:
        total_images = sum(results.values())
        print(f"\n📈 Complete dataset summary: {total_images} total images")
        
        # Check if dataset is balanced enough for training
        salem_total = results['salem_train'] + results['salem_test'] 
        other_total = results['other_train'] + results['other_test']
        
        if other_total < salem_total * 0.5:
            needed = int(salem_total * 0.8) - other_total  # Aim for 80% balance
            print(f"\n💡 Recommendation: Collect ~{needed} more other cat images for better balance")
        else:
            print(f"\n✅ Dataset balance is sufficient for training!")

if __name__ == "__main__":
    main()
