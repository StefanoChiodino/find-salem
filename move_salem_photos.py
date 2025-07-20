#!/usr/bin/env python3
"""
Move Salem Photos for Proper 80/20 Split
Moves 20% of Salem photos from train to test folder (NEVER deletes, only moves)
"""

import shutil
import random
from pathlib import Path

def move_salem_photos_to_test():
    """Move 20% of Salem photos to test folder for proper dataset split"""
    print("🐱 Moving Salem Photos to Test Folder")
    print("⚠️  NEVER deletes - only moves photos between folders!")
    print("=" * 60)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Salem directories
    salem_train_dir = Path("data/train/salem")
    salem_test_dir = Path("data/test/salem")
    
    # Ensure test directory exists
    salem_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Salem photos currently in training folder
    if not salem_train_dir.exists():
        print("❌ No training Salem directory found!")
        return
    
    all_salem_photos = [f for f in salem_train_dir.glob("*") if f.is_file() and not f.name.startswith('.')]
    
    if not all_salem_photos:
        print("❌ No Salem photos found in training folder!")
        return
    
    print(f"📊 Found {len(all_salem_photos)} Salem photos to split")
    
    # Shuffle for random split
    random.shuffle(all_salem_photos)
    
    # Calculate split (20% for testing)
    total_count = len(all_salem_photos)
    test_count = int(total_count * 0.2)
    train_count = total_count - test_count
    
    print(f"🎯 Target split: {train_count} train ({train_count/total_count*100:.1f}%), {test_count} test ({test_count/total_count*100:.1f}%)")
    
    # Move photos to test folder
    test_photos = all_salem_photos[:test_count]
    
    print(f"\n📤 Moving {len(test_photos)} Salem photos to test folder...")
    
    for photo_path in test_photos:
        # Keep original filename
        new_path = salem_test_dir / photo_path.name
        
        # Move to test folder
        shutil.move(str(photo_path), str(new_path))
        print(f"   ✅ Moved {photo_path.name} → test folder")
    
    # Final count verification
    final_train_count = len([f for f in salem_train_dir.glob("*") if f.is_file() and not f.name.startswith('.')])
    final_test_count = len([f for f in salem_test_dir.glob("*") if f.is_file() and not f.name.startswith('.')])
    
    print(f"\n🎉 Salem photo split complete!")
    print(f"📊 Final Salem photo distribution:")
    print(f"   Training: {final_train_count} photos")
    print(f"   Testing: {final_test_count} photos")
    print(f"   Total: {final_train_count + final_test_count} photos")
    
    # Show other cat photos status
    other_train_count = len([f for f in Path("data/train/other_cats").glob("*.jpg") if f.is_file()])
    other_test_count = len([f for f in Path("data/test/other_cats").glob("*.jpg") if f.is_file()])
    
    print(f"\n📊 Other cat photo status:")
    print(f"   Training: {other_train_count} photos")
    print(f"   Testing: {other_test_count} photos")
    print(f"   Total: {other_train_count + other_test_count} photos")
    
    print(f"\n🚀 Complete dataset now properly split for training!")
    print(f"📈 Total dataset: {final_train_count + final_test_count + other_train_count + other_test_count} images")
    
    return {
        'salem_train': final_train_count,
        'salem_test': final_test_count,
        'other_train': other_train_count,
        'other_test': other_test_count
    }

def main():
    """Main function to move Salem photos"""
    results = move_salem_photos_to_test()
    
    if results:
        # Check dataset balance
        salem_total = results['salem_train'] + results['salem_test']
        other_total = results['other_train'] + results['other_test']
        
        print(f"\n📊 Dataset balance:")
        print(f"   Salem photos: {salem_total}")
        print(f"   Other cat photos: {other_total}")
        
        if other_total < salem_total * 0.5:
            needed = salem_total - other_total
            print(f"\n💡 Recommendation: Collect ~{needed} more other cat images for perfect balance")
        else:
            print(f"\n✅ Dataset balance is good for training!")

if __name__ == "__main__":
    main()
