#!/usr/bin/env python3
"""
Deduplicate training folders and report final counts
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

def get_file_hash(filepath):
    """Get MD5 hash of file for duplicate detection"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {filepath}: {e}")
        return None

def dedupe_directory(directory, category_name):
    """Remove duplicate files from directory based on hash"""
    if not directory.exists():
        print(f"⚠️  {directory} doesn't exist, skipping...")
        return 0, 0
    
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Get all image files
    image_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix in extensions:
            image_files.append(file)
    
    print(f"\n🔍 Deduplicating {category_name} photos...")
    print(f"📊 Found {len(image_files)} total files")
    
    # Group files by hash
    hash_to_files = defaultdict(list)
    processed = 0
    
    for file in image_files:
        file_hash = get_file_hash(file)
        if file_hash:
            hash_to_files[file_hash].append(file)
            processed += 1
            if processed % 50 == 0:
                print(f"📈 Processed {processed}/{len(image_files)} files...")
    
    # Remove duplicates (keep first file, remove others)
    duplicates_removed = 0
    unique_count = 0
    
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            # Keep the first file, remove the rest
            kept_file = files[0]
            for dup_file in files[1:]:
                print(f"🗑️  Removing duplicate: {dup_file.name}")
                try:
                    dup_file.unlink()
                    duplicates_removed += 1
                except Exception as e:
                    print(f"❌ Error removing {dup_file}: {e}")
        unique_count += 1
    
    final_count = len([f for f in directory.iterdir() 
                      if f.is_file() and f.suffix in extensions])
    
    print(f"✅ {category_name} deduplication complete:")
    print(f"   📊 Original: {len(image_files)} files")
    print(f"   🗑️  Duplicates removed: {duplicates_removed}")
    print(f"   ✨ Final unique files: {final_count}")
    
    return len(image_files), final_count

def main():
    print("🐱 Dataset Deduplication")
    print("========================")
    
    data_dir = Path("data")
    salem_train = data_dir / "train" / "salem"
    other_train = data_dir / "train" / "other_cats"
    
    # Dedupe Salem photos
    salem_original, salem_final = dedupe_directory(salem_train, "Salem")
    
    # Dedupe other cat photos  
    other_original, other_final = dedupe_directory(other_train, "Other cats")
    
    # Final summary
    print(f"\n📊 FINAL DATASET SUMMARY")
    print(f"========================")
    print(f"🐱 Salem photos: {salem_final}")
    print(f"🐾 Other cat photos: {other_final}")
    print(f"📈 Total dataset: {salem_final + other_final}")
    
    total_duplicates = (salem_original - salem_final) + (other_original - other_final)
    print(f"🗑️  Total duplicates removed: {total_duplicates}")
    
    # Balance analysis
    if salem_final > other_final:
        needed = salem_final - other_final
        print(f"\n📊 Balance Analysis:")
        print(f"   ❌ Need {needed} more other cat photos to match Salem count")
        print(f"   🎯 Target: {salem_final} other cat photos (currently {other_final})")
        print(f"   📈 Progress: {(other_final/salem_final)*100:.1f}% of target reached")
    elif other_final > salem_final:
        excess = other_final - salem_final
        print(f"\n📊 Balance Analysis:")
        print(f"   ✅ Dataset is over-balanced! {excess} excess other cat photos")
    else:
        print(f"\n📊 Balance Analysis:")
        print(f"   ✅ Perfect balance! {salem_final} photos each")
    
    print(f"\n🚀 Dataset ready for training!")

if __name__ == "__main__":
    main()
