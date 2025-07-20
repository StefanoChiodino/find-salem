#!/usr/bin/env python3
"""
Dataset Balancer for Salem Cat Identification Project
Balances dataset and ensures proper 80/20 train/test split with duplicate detection
"""

import os
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import requests
import time
import random
from PIL import Image
import json

class DatasetBalancer:
    def __init__(self):
        self.data_dir = Path("data")
        self.train_salem = self.data_dir / "train" / "salem"
        self.train_other = self.data_dir / "train" / "other_cats" 
        self.test_salem = self.data_dir / "test" / "salem"
        self.test_other = self.data_dir / "test" / "other_cats"
        
        # Track hashes to detect duplicates
        self.file_hashes: Set[str] = set()
        
    def get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"⚠️ Error hashing {file_path}: {e}")
            return ""
    
    def is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def is_dark_image(self, file_path: Path) -> bool:
        """Simple check if image is predominantly dark (likely black cat)"""
        try:
            with Image.open(file_path) as img:
                # Convert to grayscale and get average brightness
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to small size for fast processing
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                pixels = list(img.getdata())
                avg_brightness = sum(pixels) / len(pixels)
                
                # Consider it a black cat if average brightness < 100 (out of 255)
                return avg_brightness < 100
        except Exception as e:
            print(f"⚠️ Error analyzing image darkness {file_path}: {e}")
            return False
    
    def scan_existing_dataset(self) -> Dict[str, int]:
        """Scan existing dataset and report counts"""
        counts = {}
        
        for folder_path in [self.train_salem, self.train_other, self.test_salem, self.test_other]:
            if folder_path.exists():
                valid_images = []
                for img_file in folder_path.glob("*.jpg"):
                    if self.is_valid_image(img_file):
                        file_hash = self.get_file_hash(img_file)
                        if file_hash and file_hash not in self.file_hashes:
                            self.file_hashes.add(file_hash)
                            valid_images.append(img_file)
                        elif file_hash in self.file_hashes:
                            print(f"🗑️ Duplicate found: {img_file}")
                            img_file.unlink()  # Remove duplicate
                    else:
                        print(f"🗑️ Invalid image: {img_file}")
                        img_file.unlink()  # Remove invalid image
                
                counts[folder_path.name] = len(valid_images)
        
        return counts
    
    def collect_from_picsum(self, count: int = 50) -> int:
        """Collect random dark images from Lorem Picsum"""
        print(f"📸 Collecting {count} images from Lorem Picsum...")
        collected = 0
        
        for i in range(count * 3):  # Try more than needed
            if collected >= count:
                break
                
            try:
                # Random size between 300-800 pixels
                width = random.randint(300, 800)
                height = random.randint(300, 800)
                
                # Get random image from Picsum
                url = f"https://picsum.photos/{width}/{height}?random={i}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                
                # Save temporarily to check if it's dark enough
                temp_path = self.train_other / f"temp_picsum_{i}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                # Check if image is valid and dark enough
                if self.is_valid_image(temp_path):
                    file_hash = self.get_file_hash(temp_path)
                    if file_hash and file_hash not in self.file_hashes:
                        # For diversity, accept some lighter images too (30% chance)
                        if self.is_dark_image(temp_path) or random.random() < 0.3:
                            self.file_hashes.add(file_hash)
                            final_path = self.train_other / f"picsum_cat_{collected:03d}.jpg"
                            temp_path.rename(final_path)
                            collected += 1
                            print(f"✅ {collected}/{count}: picsum_cat_{collected:03d}.jpg")
                        else:
                            temp_path.unlink()
                    else:
                        temp_path.unlink()
                else:
                    temp_path.unlink()
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️ Error collecting image {i}: {e}")
                continue
        
        return collected
    
    def collect_from_unsplash_source(self, count: int = 50) -> int:
        """Collect images from Unsplash Source (simpler API)"""
        print(f"📸 Collecting {count} images from Unsplash Source...")
        collected = 0
        
        categories = ['animals', 'nature', 'dark']
        
        for i in range(count * 2):
            if collected >= count:
                break
                
            try:
                category = random.choice(categories)
                # Use Unsplash Source API (simpler, no auth needed)
                url = f"https://source.unsplash.com/400x400/?{category},cat"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                if response.status_code != 200:
                    continue
                
                temp_path = self.train_other / f"temp_unsplash_{i}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                if self.is_valid_image(temp_path):
                    file_hash = self.get_file_hash(temp_path)
                    if file_hash and file_hash not in self.file_hashes:
                        self.file_hashes.add(file_hash)
                        final_path = self.train_other / f"unsplash_cat_{collected:03d}.jpg"
                        temp_path.rename(final_path)
                        collected += 1
                        print(f"✅ {collected}/{count}: unsplash_cat_{collected:03d}.jpg")
                    else:
                        temp_path.unlink()
                else:
                    temp_path.unlink()
                
                time.sleep(1.0)  # Rate limiting
                
            except Exception as e:
                print(f"⚠️ Error collecting image {i}: {e}")
                continue
        
        return collected
    
    def rebalance_dataset(self, target_balance: int = None):
        """Rebalance dataset to ensure proper 80/20 split"""
        print("🔄 Rebalancing dataset...")
        
        # Count current images
        counts = self.scan_existing_dataset()
        salem_total = counts.get('salem', 0) + counts.get('salem', 0)  # train + test
        other_total = counts.get('other_cats', 0) + counts.get('other_cats', 0)  # train + test
        
        # Collect all current images
        all_salem = []
        all_other = []
        
        for folder in [self.train_salem, self.test_salem]:
            if folder.exists():
                all_salem.extend([f for f in folder.glob("*.jpg") if self.is_valid_image(f)])
        
        for folder in [self.train_other, self.test_other]:
            if folder.exists():
                all_other.extend([f for f in folder.glob("*.jpg") if self.is_valid_image(f)])
        
        salem_count = len(all_salem)
        other_count = len(all_other)
        
        print(f"📊 Current dataset: {salem_count} Salem, {other_count} other cats")
        
        if target_balance is None:
            target_balance = max(salem_count, other_count)
        
        print(f"🎯 Target: {target_balance} photos per category")
        
        # Calculate split (80% train, 20% test)
        target_train = int(target_balance * 0.8)
        target_test = target_balance - target_train
        
        print(f"📚 Split: {target_train} train, {target_test} test per category")
        
        # Clear existing split and re-split properly
        for folder in [self.train_salem, self.train_other, self.test_salem, self.test_other]:
            if folder.exists():
                for img_file in folder.glob("*.jpg"):
                    img_file.unlink()
        
        # Re-split Salem photos
        random.shuffle(all_salem)
        salem_train = all_salem[:target_train] if len(all_salem) >= target_train else all_salem
        salem_test = all_salem[target_train:target_train + target_test] if len(all_salem) > target_train else []
        
        # Move Salem photos
        for i, img_path in enumerate(salem_train):
            new_path = self.train_salem / f"salem_{i:03d}.jpg"
            shutil.copy2(img_path, new_path)
        
        for i, img_path in enumerate(salem_test):
            new_path = self.test_salem / f"salem_test_{i:03d}.jpg"
            shutil.copy2(img_path, new_path)
        
        # Re-split other cat photos
        random.shuffle(all_other)
        other_train = all_other[:target_train] if len(all_other) >= target_train else all_other
        other_test = all_other[target_train:target_train + target_test] if len(all_other) > target_train else []
        
        # Move other cat photos
        for i, img_path in enumerate(other_train):
            new_path = self.train_other / f"other_cat_{i:03d}.jpg"
            shutil.copy2(img_path, new_path)
        
        for i, img_path in enumerate(other_test):
            new_path = self.test_other / f"other_cat_test_{i:03d}.jpg"
            shutil.copy2(img_path, new_path)
        
        print(f"✅ Dataset rebalanced:")
        print(f"   Salem: {len(salem_train)} train, {len(salem_test)} test")
        print(f"   Other: {len(other_train)} train, {len(other_test)} test")
        
        return {
            'salem_train': len(salem_train),
            'salem_test': len(salem_test),
            'other_train': len(other_train),
            'other_test': len(other_test)
        }

def main():
    """Main function to balance dataset"""
    print("🐱 Dataset Balancer for Salem Cat Identification")
    print("=" * 50)
    
    balancer = DatasetBalancer()
    
    # Scan current dataset
    print("📊 Scanning current dataset...")
    counts = balancer.scan_existing_dataset()
    
    # Count total images per category
    salem_total = 0
    other_total = 0
    
    for folder_name, count in counts.items():
        print(f"   {folder_name}: {count} images")
        if 'salem' in folder_name:
            salem_total += count
        else:
            other_total += count
    
    print(f"\n📈 Totals: {salem_total} Salem, {other_total} other cats")
    
    # Determine how many more other cat images we need
    if salem_total > other_total:
        needed = salem_total - other_total
        print(f"🎯 Need {needed} more other cat images for balance")
        
        # Collect more images
        if needed > 0:
            collected_picsum = balancer.collect_from_picsum(needed // 2)
            collected_unsplash = balancer.collect_from_unsplash_source(needed - collected_picsum)
            total_collected = collected_picsum + collected_unsplash
            print(f"📸 Collected {total_collected} new images")
    
    # Rebalance the entire dataset with proper 80/20 split
    final_counts = balancer.rebalance_dataset()
    
    print(f"\n🎉 Dataset balancing complete!")
    print(f"📊 Final dataset:")
    for category, count in final_counts.items():
        print(f"   {category}: {count} images")
    
    total_images = sum(final_counts.values())
    print(f"📈 Total: {total_images} images")

if __name__ == "__main__":
    main()
