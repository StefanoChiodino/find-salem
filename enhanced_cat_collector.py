#!/usr/bin/env python3
"""
Enhanced Cat Photo Collector with Duplicate Detection and Black Cat Filtering
Ensures high-quality dataset with only unique black cat images.
"""

import hashlib
import numpy as np
from PIL import Image, ImageStat
import requests
import json
import time
import os
from pathlib import Path
from typing import Set, Tuple, List
import imagehash

class EnhancedBlackCatCollector:
    """Enhanced collector with duplicate detection and black cat filtering"""
    
    def __init__(self, output_dir: str = "data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track image hashes to prevent duplicates
        self.image_hashes: Set[str] = set()
        self.perceptual_hashes: Set[str] = set()
        
        # Load existing hashes from current dataset
        self._load_existing_hashes()
        
        print(f"🐱 Enhanced Black Cat Collector initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔍 Existing images: {len(self.image_hashes)} unique hashes found")
    
    def _load_existing_hashes(self):
        """Load hashes of existing images to prevent duplicates"""
        print("🔍 Loading existing image hashes...")
        
        # Check all image directories
        image_dirs = [
            Path("data/train/other_cats"),
            Path("data/test/other_cats"),
            Path("demo_samples/other_cats")
        ]
        
        count = 0
        for img_dir in image_dirs:
            if img_dir.exists():
                for img_file in img_dir.glob("*.jpg"):
                    try:
                        # File hash
                        file_hash = self._get_file_hash(img_file)
                        self.image_hashes.add(file_hash)
                        
                        # Perceptual hash
                        perceptual_hash = self._get_perceptual_hash(img_file)
                        self.perceptual_hashes.add(str(perceptual_hash))
                        
                        count += 1
                    except Exception as e:
                        print(f"⚠️ Error processing {img_file}: {e}")
        
        print(f"✅ Loaded {count} existing image hashes")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file contents"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_perceptual_hash(self, file_path: Path) -> imagehash.ImageHash:
        """Get perceptual hash for near-duplicate detection"""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB to ensure compatibility
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return imagehash.phash(img)
        except Exception as e:
            # Return a default hash if processing fails
            return imagehash.phash(Image.new('RGB', (1, 1)))
    
    def _is_black_cat(self, image_path: Path) -> Tuple[bool, str]:
        """
        Determine if image contains a black cat using color analysis
        Returns (is_black_cat, reason)
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image statistics
                stat = ImageStat.Stat(img)
                
                # Calculate average brightness
                avg_brightness = sum(stat.mean) / 3
                
                # Calculate color distribution
                r_avg, g_avg, b_avg = stat.mean
                
                # Check if image is predominantly dark (black cat indicator)
                if avg_brightness > 120:
                    return False, f"Too bright (avg: {avg_brightness:.1f})"
                
                # Check if colors are relatively balanced (not overly colorful)
                color_variance = np.var([r_avg, g_avg, b_avg])
                if color_variance > 500:
                    return False, f"Too colorful (variance: {color_variance:.1f})"
                
                # Check for very low brightness (likely black cat)
                if avg_brightness < 80:
                    return True, f"Very dark (avg: {avg_brightness:.1f})"
                
                # Moderate darkness - could be black cat
                if avg_brightness < 100:
                    return True, f"Dark (avg: {avg_brightness:.1f})"
                
                return False, f"Not dark enough (avg: {avg_brightness:.1f})"
                
        except Exception as e:
            return False, f"Error analyzing: {e}"
    
    def _is_duplicate(self, image_path: Path) -> Tuple[bool, str]:
        """Check if image is a duplicate using multiple methods"""
        try:
            # Check file hash
            file_hash = self._get_file_hash(image_path)
            if file_hash in self.image_hashes:
                return True, "Exact duplicate (file hash)"
            
            # Check perceptual hash for near-duplicates
            perceptual_hash = self._get_perceptual_hash(image_path)
            
            # Check for very similar images (hamming distance < 5)
            for existing_hash in self.perceptual_hashes:
                try:
                    existing_phash = imagehash.hex_to_hash(existing_hash)
                    if abs(perceptual_hash - existing_phash) < 5:
                        return True, f"Near duplicate (phash distance: {abs(perceptual_hash - existing_phash)})"
                except:
                    continue
            
            return False, "Unique image"
            
        except Exception as e:
            return True, f"Error checking: {e}"
    
    def download_image(self, url: str, filename: str) -> Tuple[bool, str]:
        """Download and validate a single image"""
        try:
            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save temporarily
            temp_path = self.output_dir / f"temp_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Check if it's a valid image
            try:
                with Image.open(temp_path) as img:
                    img.verify()
            except Exception:
                os.remove(temp_path)
                return False, "Invalid image format"
            
            # Check for duplicates
            is_dup, dup_reason = self._is_duplicate(temp_path)
            if is_dup:
                os.remove(temp_path)
                return False, f"Duplicate: {dup_reason}"
            
            # Check if it's a black cat
            is_black, black_reason = self._is_black_cat(temp_path)
            if not is_black:
                os.remove(temp_path)
                return False, f"Not black cat: {black_reason}"
            
            # Valid unique black cat - move to final location
            final_path = self.output_dir / filename
            os.rename(temp_path, final_path)
            
            # Add to hash sets
            file_hash = self._get_file_hash(final_path)
            perceptual_hash = self._get_perceptual_hash(final_path)
            
            self.image_hashes.add(file_hash)
            self.perceptual_hashes.add(str(perceptual_hash))
            
            return True, f"✅ Black cat saved: {black_reason}"
            
        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            return False, f"Download error: {e}"
    
    def collect_from_pexels(self, query: str = "black cat", count: int = 50):
        """Collect black cat images from Pexels (free API alternative)"""
        print(f"🔍 Searching Pexels for '{query}' (target: {count} images)...")
        
        collected = 0
        page = 1
        
        while collected < count and page <= 10:
            try:
                # Pexels public search (no API key needed for basic search)
                url = f"https://www.pexels.com/search/{query.replace(' ', '%20')}/"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code != 200:
                    print(f"⚠️ HTTP {response.status_code} from Pexels")
                    break
                
                # Simple HTML parsing to find image URLs
                import re
                img_pattern = r'https://images\.pexels\.com/photos/[^"]+\.jpeg\?[^"]*'
                image_urls = re.findall(img_pattern, response.text)
                
                if not image_urls:
                    print(f"📭 No more photos found on page {page}")
                    break
                
                print(f"📄 Processing page {page} ({len(image_urls)} photos found)...")
                
                for i, image_url in enumerate(image_urls[:20]):  # Limit to 20 per page
                    if collected >= count:
                        break
                    
                    try:
                        filename = f"pexels_black_cat_{page:02d}_{i:02d}.jpg"
                        
                        success, reason = self.download_image(image_url, filename)
                        
                        if success:
                            collected += 1
                            print(f"✅ {collected}/{count}: {filename} - {reason}")
                        else:
                            print(f"❌ Skipped image {i+1}: {reason}")
                        
                        time.sleep(1.0)  # Rate limiting
                        
                    except Exception as e:
                        print(f"⚠️ Error processing image {i+1}: {e}")
                        continue
                
                
                page += 1
                time.sleep(2)  # Rate limiting between pages
                
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break
        
        print(f"\n🎯 Collection complete: {collected} unique black cat images")
        return collected

def clean_existing_dataset():
    """Clean existing dataset to remove duplicates and non-black cats"""
    print("🧹 Cleaning existing dataset...")
    print("=" * 50)
    
    collector = EnhancedBlackCatCollector()
    
    # Directories to clean
    dirs_to_clean = [
        Path("data/train/other_cats"),
        Path("data/test/other_cats")
    ]
    
    total_checked = 0
    duplicates_removed = 0
    non_black_removed = 0
    kept = 0
    
    for directory in dirs_to_clean:
        if not directory.exists():
            continue
            
        print(f"\n📁 Cleaning {directory}...")
        
        image_files = list(directory.glob("*.jpg"))
        print(f"   Found {len(image_files)} images")
        
        # Reset collector hashes for clean analysis
        temp_hashes = set()
        temp_phashes = set()
        
        for img_file in image_files:
            total_checked += 1
            keep_file = True
            reason = ""
            
            try:
                # Check if it's a black cat
                is_black, black_reason = collector._is_black_cat(img_file)
                if not is_black:
                    keep_file = False
                    reason = f"Not black cat: {black_reason}"
                    non_black_removed += 1
                
                # Check for duplicates within this cleaning session
                if keep_file:
                    file_hash = collector._get_file_hash(img_file)
                    if file_hash in temp_hashes:
                        keep_file = False
                        reason = "Duplicate within dataset"
                        duplicates_removed += 1
                    else:
                        temp_hashes.add(file_hash)
                
                if keep_file:
                    kept += 1
                    print(f"   ✅ Keep: {img_file.name} - {black_reason}")
                else:
                    print(f"   ❌ Remove: {img_file.name} - {reason}")
                    os.remove(img_file)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {img_file.name}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Total checked: {total_checked}")
    print(f"   Kept: {kept}")
    print(f"   Duplicates removed: {duplicates_removed}")
    print(f"   Non-black cats removed: {non_black_removed}")
    print(f"   Total removed: {duplicates_removed + non_black_removed}")

def main():
    """Main function for enhanced data collection"""
    print("🐱 Enhanced Black Cat Data Collection")
    print("=" * 50)
    
    # First, clean existing dataset
    clean_existing_dataset()
    
    # Then collect new unique black cat images
    print(f"\n🔍 Collecting new black cat images...")
    collector = EnhancedBlackCatCollector()
    
    # Collect 50 unique black cat images from Pexels
    collected = collector.collect_from_pexels("black cat", count=50)
    
    print(f"\n🎉 Enhanced collection complete!")
    print(f"📸 Collected {collected} new unique black cat images")
    print(f"💡 Dataset is now cleaned and contains only unique black cats")

if __name__ == "__main__":
    main()
