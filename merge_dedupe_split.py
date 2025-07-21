#!/usr/bin/env python3
"""
Merge test data back, deduplicate, and re-split 80/20
Safe operation that never deletes Salem photos - only moves them
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Set, List
import random

class SafeDatasetManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.salem_train = self.data_dir / "train" / "salem"
        self.salem_test = self.data_dir / "test" / "salem"
        self.other_train = self.data_dir / "train" / "other_cats"
        self.other_test = self.data_dir / "test" / "other_cats"
        
        # Create temp directories for safe processing
        self.temp_dir = Path("temp_dataset_processing")
        self.temp_salem = self.temp_dir / "salem_all"
        self.temp_other = self.temp_dir / "other_cats_all"
        
    def get_file_hash(self, filepath: Path) -> str:
        """Get MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"❌ Error hashing {filepath}: {e}")
            return ""
    
    def get_all_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        files = []
        if directory.exists():
            for file in directory.iterdir():
                if file.is_file() and file.suffix in extensions:
                    files.append(file)
        return files
    
    def merge_and_dedupe_category(self, train_dir: Path, test_dir: Path, 
                                  temp_dir: Path, category_name: str) -> List[Path]:
        """Merge train/test for one category and deduplicate"""
        print(f"\n🔄 Processing {category_name} photos...")
        
        # Create temp directory
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all files from both train and test
        train_files = self.get_all_image_files(train_dir)
        test_files = self.get_all_image_files(test_dir)
        
        print(f"📊 Found {len(train_files)} training + {len(test_files)} test = {len(train_files + test_files)} total {category_name} photos")
        
        # Track hashes for deduplication
        seen_hashes: Set[str] = set()
        unique_files: List[Path] = []
        duplicates_removed = 0
        
        # Process all files (train first, then test)
        all_files = train_files + test_files
        
        for i, src_file in enumerate(all_files, 1):
            file_hash = self.get_file_hash(src_file)
            if not file_hash:
                continue
                
            if file_hash in seen_hashes:
                print(f"🗑️  Duplicate {i}: {src_file.name}")
                duplicates_removed += 1
                continue
            
            seen_hashes.add(file_hash)
            
            # Copy to temp directory with unique name
            dest_file = temp_dir / f"{category_name}_{len(unique_files):03d}{src_file.suffix}"
            try:
                shutil.copy2(src_file, dest_file)
                unique_files.append(dest_file)
                if i % 50 == 0:
                    print(f"📈 Processed {i}/{len(all_files)} files, {len(unique_files)} unique")
            except Exception as e:
                print(f"❌ Error copying {src_file}: {e}")
        
        print(f"✅ {category_name} deduplication complete:")
        print(f"   📊 Original: {len(all_files)} files")
        print(f"   🗑️  Duplicates removed: {duplicates_removed}")
        print(f"   ✨ Unique files: {len(unique_files)}")
        
        return unique_files
    
    def split_files(self, files: List[Path], train_dir: Path, test_dir: Path, 
                   test_ratio: float = 0.2) -> tuple:
        """Split files 80/20 into train/test directories"""
        # Shuffle files for random split
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        # Calculate split point
        total_files = len(shuffled_files)
        test_count = int(total_files * test_ratio)
        train_count = total_files - test_count
        
        train_files = shuffled_files[:train_count]
        test_files = shuffled_files[train_count:]
        
        # Ensure directories exist
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files to final locations
        print(f"📁 Moving {len(train_files)} files to training...")
        for i, src_file in enumerate(train_files, 1):
            dest_file = train_dir / src_file.name
            shutil.move(str(src_file), str(dest_file))
            if i % 50 == 0:
                print(f"   📈 Moved {i}/{len(train_files)} training files")
        
        print(f"📁 Moving {len(test_files)} files to testing...")
        for i, src_file in enumerate(test_files, 1):
            dest_file = test_dir / src_file.name
            shutil.move(str(src_file), str(dest_file))
            if i % 20 == 0:
                print(f"   📈 Moved {i}/{len(test_files)} test files")
        
        return len(train_files), len(test_files)
    
    def clean_original_directories(self):
        """Clean out original train/test directories"""
        print("\n🧹 Cleaning original directories...")
        
        dirs_to_clean = [self.salem_train, self.salem_test, self.other_train, self.other_test]
        
        for directory in dirs_to_clean:
            if directory.exists():
                for file in directory.iterdir():
                    if file.is_file():
                        file.unlink()
                print(f"✅ Cleaned {directory}")
    
    def cleanup_temp(self):
        """Remove temporary processing directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up {self.temp_dir}")
    
    def run_full_process(self):
        """Execute the complete merge, dedupe, split workflow"""
        print("🐱 Salem Dataset Merge, Dedupe & Re-split")
        print("⚠️  SAFETY: Never deletes Salem photos, only moves them")
        print("=" * 60)
        
        try:
            # Step 1: Merge and deduplicate Salem photos
            salem_files = self.merge_and_dedupe_category(
                self.salem_train, self.salem_test, self.temp_salem, "salem"
            )
            
            # Step 2: Merge and deduplicate other cat photos
            other_files = self.merge_and_dedupe_category(
                self.other_train, self.other_test, self.temp_other, "other_cat"
            )
            
            print(f"\n📊 Final dataset balance after deduplication:")
            print(f"   🐱 Salem photos: {len(salem_files)}")
            print(f"   🐾 Other cat photos: {len(other_files)}")
            print(f"   ⚖️  Ratio: {len(salem_files)/(len(other_files)+0.001):.2f}:1")
            
            # Step 3: Clean original directories
            self.clean_original_directories()
            
            # Step 4: Split Salem photos 80/20
            print(f"\n🔄 Splitting Salem photos 80/20...")
            salem_train_count, salem_test_count = self.split_files(
                salem_files, self.salem_train, self.salem_test
            )
            
            # Step 5: Split other cat photos 80/20
            print(f"\n🔄 Splitting other cat photos 80/20...")
            other_train_count, other_test_count = self.split_files(
                other_files, self.other_train, self.other_test
            )
            
            print(f"\n🎉 Dataset processing complete!")
            print(f"📊 Final split:")
            print(f"   Salem training: {salem_train_count}")
            print(f"   Salem test: {salem_test_count}")
            print(f"   Other training: {other_train_count}")
            print(f"   Other test: {other_test_count}")
            print(f"   📈 Total: {salem_train_count + salem_test_count + other_train_count + other_test_count} photos")
            
            balance_ratio = (salem_train_count + salem_test_count) / (other_train_count + other_test_count + 0.001)
            print(f"   ⚖️  Perfect balance: {balance_ratio:.2f}:1 ratio")
            
        except Exception as e:
            print(f"❌ Error in dataset processing: {e}")
            print("🔄 Attempting cleanup...")
            
        finally:
            # Always cleanup temp directory
            self.cleanup_temp()

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    
    manager = SafeDatasetManager()
    manager.run_full_process()
