#!/usr/bin/env python3
"""
FastAI Book-Style Image Collector with Enhanced Duplicate Detection
Based on fastbook Chapter 1 approach with improvements for Salem project
NEVER TOUCHES SALEM PHOTOS - only collects other black cat images
"""

import os
import hashlib
import requests
import time
import random
from pathlib import Path
from typing import Set, List, Tuple
from PIL import Image
import json
from urllib.parse import quote_plus

# Try to import fastbook dependencies
try:
    from fastai.vision.all import *
    from fastbook import *
    FASTAI_AVAILABLE = True
except ImportError:
    FASTAI_AVAILABLE = False
    print("⚠️ FastAI not available, using custom implementation")

class FastAIStyleCollector:
    """Enhanced collector using FastAI book approach with duplicate detection"""
    
    def __init__(self, output_dir: str = "data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Duplicate detection
        self.file_hashes: Set[str] = set()
        self.downloaded_urls: Set[str] = set()
        
        print("⚠️  SAFETY: This collector NEVER touches Salem photos!")
        
        # Load existing hashes
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing file hashes for duplicate detection"""
        print("🔍 Loading existing image hashes...")
        
        # Only check other_cats directories - NEVER salem folders
        for data_path in ["data/train/other_cats", "data/test/other_cats", "demo_samples/other_cats"]:
            data_dir = Path(data_path)
            if data_dir.exists():
                for img_file in data_dir.glob("*.jpg"):
                    if self._is_valid_image(img_file):
                        file_hash = self._get_file_hash(img_file)
                        if file_hash:
                            self.file_hashes.add(file_hash)
        
        print(f"✅ Loaded {len(self.file_hashes)} existing image hashes")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
                return True
        except Exception:
            return False
    
    def _is_likely_black_cat(self, file_path: Path) -> Tuple[bool, str]:
        """Simple check if image might be a black cat"""
        try:
            with Image.open(file_path) as img:
                # Basic size check
                width, height = img.size
                if width < 100 or height < 100:
                    return False, "Too small"
                
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Sample center pixels for darkness check
                center_x, center_y = width // 2, height // 2
                sample_size = min(50, width // 4, height // 4)
                
                crop_box = (
                    center_x - sample_size//2,
                    center_y - sample_size//2, 
                    center_x + sample_size//2,
                    center_y + sample_size//2
                )
                
                center_crop = img.crop(crop_box)
                pixels = list(center_crop.getdata())
                
                # Calculate average brightness
                avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
                
                # We want darker images but not solid black
                if avg_brightness < 40:
                    return False, f"Too dark {avg_brightness:.1f}"
                
                if avg_brightness > 130:
                    return False, f"Too bright {avg_brightness:.1f}"
                
                return True, f"Good candidate {width}x{height}, brightness: {avg_brightness:.1f}"
                
        except Exception as e:
            return False, f"Analysis error: {e}"
    
    def search_images_ddg_custom(self, term: str, max_images: int = 30) -> List[str]:
        """Custom DuckDuckGo image search (fallback if fastbook not available)"""
        print(f"🦆 Searching DuckDuckGo for '{term}'...")
        
        try:
            # DuckDuckGo image search
            search_url = "https://duckduckgo.com/"
            
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Get search token first
            params = {'q': term}
            resp = session.get(search_url, params=params)
            
            if 'vqd=' in resp.text:
                import re
                vqd = re.search(r'vqd=([^&]+)', resp.text).group(1)
                
                # Now search for images
                image_url = "https://duckduckgo.com/i.js"
                params = {
                    'l': 'us-en',
                    'o': 'json', 
                    'q': term,
                    'vqd': vqd,
                    'f': ',,,',
                    'p': '1'
                }
                
                resp = session.get(image_url, params=params)
                
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get('results', [])
                    
                    image_urls = []
                    for result in results[:max_images]:
                        img_url = result.get('image')
                        if img_url and img_url not in self.downloaded_urls:
                            image_urls.append(img_url)
                    
                    print(f"📋 Found {len(image_urls)} DuckDuckGo image URLs")
                    return image_urls
            
            print("⚠️ Could not extract search token from DuckDuckGo")
            return []
            
        except Exception as e:
            print(f"❌ DuckDuckGo search error: {e}")
            return []
    
    def search_images_fastai(self, term: str, max_images: int = 30) -> List[str]:
        """Use fastbook's search_images_ddg if available"""
        if not FASTAI_AVAILABLE:
            return self.search_images_ddg_custom(term, max_images)
        
        try:
            print(f"📚 Using FastAI book method to search for '{term}'...")
            
            # Use fastbook's search_images_ddg function
            urls = search_images_ddg(term, max_images=max_images)
            
            # Filter out URLs we've already tried
            new_urls = [url for url in urls if url not in self.downloaded_urls]
            
            print(f"📋 Found {len(new_urls)} new FastAI image URLs")
            return new_urls
            
        except Exception as e:
            print(f"❌ FastAI search error: {e}")
            # Fallback to custom implementation
            return self.search_images_ddg_custom(term, max_images)
    
    def download_and_validate_image(self, url: str, filename: str) -> Tuple[bool, str]:
        """Download image with validation - FastAI book style"""
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            # Mark URL as attempted
            self.downloaded_urls.add(url)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png']):
                return False, f"Not a supported image: {content_type}"
            
            # Download to temp file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate image
            if not self._is_valid_image(temp_path):
                temp_path.unlink()
                return False, "Invalid image file"
            
            # Check for duplicates using file hash
            file_hash = self._get_file_hash(temp_path)
            if not file_hash:
                temp_path.unlink()
                return False, "Could not hash file"
            
            if file_hash in self.file_hashes:
                temp_path.unlink()
                return False, "Duplicate image (hash match)"
            
            # Check if it looks like a black cat
            is_good_candidate, reason = self._is_likely_black_cat(temp_path)
            if not is_good_candidate:
                temp_path.unlink()
                return False, f"Not suitable: {reason}"
            
            # All checks passed - move to final location
            temp_path.rename(final_path)
            self.file_hashes.add(file_hash)
            
            return True, reason
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Download error: {e}"
    
    def collect_black_cats(self, target_count: int = 50) -> int:
        """Collect black cat images using FastAI book approach"""
        print(f"🐱 Collecting {target_count} black cat images using FastAI book method")
        print("=" * 60)
        
        # Search terms optimized for black cats
        search_terms = [
            "black cat",
            "black kitten", 
            "black domestic cat",
            "dark cat",
            "black feline",
            "black cat portrait"
        ]
        
        collected = 0
        all_urls = []
        
        # Collect URLs from all search terms
        for term in search_terms:
            if collected >= target_count:
                break
            
            urls = self.search_images_fastai(term, max_images=20)
            all_urls.extend(urls)
            
            # Rate limiting
            time.sleep(2)
        
        # Remove duplicates and shuffle
        unique_urls = list(set(all_urls))
        random.shuffle(unique_urls)
        
        print(f"📊 Total unique URLs to try: {len(unique_urls)}")
        
        # Download and validate images
        for i, url in enumerate(unique_urls):
            if collected >= target_count:
                break
            
            filename = f"black_cat_{collected:03d}.jpg"
            
            success, reason = self.download_and_validate_image(url, filename)
            
            if success:
                collected += 1
                print(f"✅ {collected}/{target_count}: {filename} - {reason}")
            else:
                print(f"❌ Skipped URL {i+1}: {reason}")
            
            # Rate limiting
            time.sleep(1.5)
            
            # Progress update every 10 attempts
            if (i + 1) % 10 == 0:
                print(f"📈 Progress: {i+1} URLs tried, {collected} images collected")
        
        return collected
    
    def remove_failed_images(self):
        """Remove any failed/corrupted images (FastAI book style cleanup)"""
        print("🧹 Cleaning up failed images...")
        
        removed = 0
        for img_file in self.output_dir.glob("*.jpg"):
            if not self._is_valid_image(img_file):
                print(f"🗑️ Removing failed image: {img_file.name}")
                img_file.unlink()
                removed += 1
        
        print(f"✅ Removed {removed} failed images")
        return removed

def main():
    """Main collection function using FastAI book approach"""
    print("🐱 FastAI Book-Style Black Cat Collector")
    print("⚠️  NEVER touches Salem photos - only collects other black cats")
    print("=" * 60)
    
    collector = FastAIStyleCollector()
    
    # Collect images using FastAI book method
    target_count = 60  # Reasonable target for balancing dataset
    collected = collector.collect_black_cats(target_count)
    
    # Clean up any failed images (FastAI book recommends this)
    removed = collector.remove_failed_images()
    
    # Final count
    final_count = len([f for f in collector.output_dir.glob("*.jpg") if collector._is_valid_image(f)])
    
    print(f"\n🎉 Collection complete!")
    print(f"📊 Results:")
    print(f"   Images attempted: {collected}")
    print(f"   Failed images removed: {removed}")  
    print(f"   Final valid images: {final_count}")
    print(f"💾 Images saved to: {collector.output_dir}")
    print(f"⚠️  Salem photos remain completely untouched!")
    
    if final_count > 30:
        print(f"✅ Good collection! Ready for training")
    else:
        print(f"💡 May need to run again for more images")

if __name__ == "__main__":
    main()
