#!/usr/bin/env python3
"""
Merge all photos back into training folders and assess dataset balance
"""

import os
import shutil
from pathlib import Path

def get_image_count(directory):
    """Count image files in directory"""
    if not directory.exists():
        return 0
    
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0
    for file in directory.iterdir():
        if file.is_file() and file.suffix in extensions:
            count += 1
    return count

def move_files(src_dir, dest_dir, category_name):
    """Move all files from src to dest directory"""
    if not src_dir.exists():
        print(f"⚠️  {src_dir} doesn't exist, skipping...")
        return 0
    
    # Ensure destination exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for file in src_dir.iterdir():
        if file.is_file() and file.suffix in extensions:
            dest_file = dest_dir / file.name
            # Handle name conflicts
            counter = 1
            while dest_file.exists():
                name_parts = file.name.rsplit('.', 1)
                if len(name_parts) == 2:
                    dest_file = dest_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    dest_file = dest_dir / f"{file.name}_{counter}"
                counter += 1
            
            shutil.move(str(file), str(dest_file))
            moved_count += 1
    
    if moved_count > 0:
        print(f"📁 Moved {moved_count} {category_name} photos from test to training")
    
    return moved_count

def main():
    print("🐱 Merging All Photos Back to Training")
    print("=====================================")
    
    data_dir = Path("data")
    
    # Define directories
    salem_train = data_dir / "train" / "salem"
    salem_test = data_dir / "test" / "salem"
    other_train = data_dir / "train" / "other_cats"
    other_test = data_dir / "test" / "other_cats"
    
    # Count current photos
    print("📊 Current photo distribution:")
    salem_train_count = get_image_count(salem_train)
    salem_test_count = get_image_count(salem_test)
    other_train_count = get_image_count(other_train)
    other_test_count = get_image_count(other_test)
    
    print(f"   Salem training: {salem_train_count}")
    print(f"   Salem test: {salem_test_count}")
    print(f"   Other cats training: {other_train_count}")
    print(f"   Other cats test: {other_test_count}")
    
    total_salem = salem_train_count + salem_test_count
    total_other = other_train_count + other_test_count
    
    print(f"\n📈 Current totals:")
    print(f"   🐱 Salem photos: {total_salem}")
    print(f"   🐾 Other cat photos: {total_other}")
    print(f"   📊 Total dataset: {total_salem + total_other}")
    
    if total_other > 0:
        ratio = total_salem / total_other
        print(f"   ⚖️  Current ratio: {ratio:.2f}:1 (Salem:Other)")
    
    # Move test photos to training
    print(f"\n🔄 Moving all photos to training folders...")
    salem_moved = move_files(salem_test, salem_train, "Salem")
    other_moved = move_files(other_test, other_train, "other cat")
    
    # Final counts
    final_salem = get_image_count(salem_train)
    final_other = get_image_count(other_train)
    
    print(f"\n🎯 Final training dataset:")
    print(f"   🐱 Salem photos: {final_salem}")
    print(f"   🐾 Other cat photos: {final_other}")
    print(f"   📊 Total: {final_salem + final_other}")
    
    # Calculate balance
    if final_salem > final_other:
        needed = final_salem - final_other
        print(f"\n📊 Balance Analysis:")
        print(f"   ❌ Need {needed} more other cat photos to match Salem count")
        print(f"   🎯 Target: {final_salem} other cat photos (currently {final_other})")
        print(f"   📈 Progress: {(final_other/final_salem)*100:.1f}% of target reached")
    elif final_other > final_salem:
        excess = final_other - final_salem
        print(f"\n📊 Balance Analysis:")
        print(f"   ✅ Dataset is over-balanced! {excess} excess other cat photos")
        print(f"   🎯 Perfect balance at {final_salem} photos each")
    else:
        print(f"\n📊 Balance Analysis:")
        print(f"   ✅ Perfect balance! {final_salem} photos each")
    
    print(f"\n🚀 Ready for training with {final_salem + final_other} total photos!")

if __name__ == "__main__":
    main()
