#!/usr/bin/env python3
"""
High-Quality Pixabay Cat Photo Collector
Focused on collecting high-quality black cat images from Pixabay
"""

import os
import hashlib
import requests
import time
import random
from pathlib import Path
from typing import Set, Tuple
from PIL import Image
import json

class PixabayCollector:
    def __init__(self, output_dir: str = "data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_hashes: Set[str] = set()
        
        # Load existing hashes to avoid duplicates
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing file hashes to detect duplicates"""
        print("🔍 Loading existing image hashes...")
        
        # Check all data directories for existing images
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
        """Get MD5 hash of file for duplicate detection"""
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
    
    def _is_good_quality_image(self, file_path: Path) -> Tuple[bool, str]:
        """Check if image meets quality criteria"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Quality criteria
                if width < 200 or height < 200:
                    return False, "Too small (min 200px)"
                
                if width * height < 40000:  # Less than 200x200 effective
                    return False, "Low resolution"
                
                # Check if image is predominantly dark (good for black cats)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Sample pixels to check darkness
                img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
                pixels = list(img_small.getdata())
                
                # Calculate average darkness
                avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
                
                # We want darker images (black cats) but not completely black
                if avg_brightness < 50:
                    return False, "Too dark/black"
                
                if avg_brightness > 150:
                    return False, "Too bright/light"
                
                return True, f"Good quality {width}x{height}, brightness: {avg_brightness:.1f}"
                
        except Exception as e:
            return False, f"Error analyzing: {e}"
    
    def collect_from_pixabay_api(self, count: int = 50, api_key: str = None) -> int:
        """Collect from Pixabay API (requires free API key)"""
        if not api_key:
            print("⚠️ No Pixabay API key provided. Get free key at: https://pixabay.com/api/docs/")
            return 0
        
        print(f"📸 Collecting {count} images from Pixabay API...")
        collected = 0
        page = 1
        
        search_terms = [
            "black cat", "dark cat", "black kitten", 
            "black feline", "black domestic cat", "dark kitten"
        ]
        
        while collected < count and page <= 10:
            try:
                search_term = random.choice(search_terms)
                
                url = "https://pixabay.com/api/"
                params = {
                    'key': api_key,
                    'q': search_term,
                    'image_type': 'photo',
                    'category': 'animals',
                    'min_width': 300,
                    'min_height': 300,
                    'per_page': 20,
                    'page': page,
                    'safesearch': 'true',
                    'order': 'popular'
                }
                
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    print(f"⚠️ HTTP {response.status_code} from Pixabay API")
                    break
                
                data = response.json()
                images = data.get('hits', [])
                
                if not images:
                    print(f"📭 No more images found for '{search_term}' on page {page}")
                    page += 1
                    continue
                
                print(f"📄 Processing page {page} ({len(images)} images for '{search_term}')...")
                
                for img_data in images:
                    if collected >= count:
                        break
                    
                    try:
                        image_url = img_data['webformatURL']  # Medium resolution
                        image_id = img_data['id']
                        filename = f"pixabay_cat_{image_id}.jpg"
                        
                        success, reason = self.download_and_validate_image(image_url, filename)
                        
                        if success:
                            collected += 1
                            print(f"✅ {collected}/{count}: {filename} - {reason}")
                        else:
                            print(f"❌ Skipped {image_id}: {reason}")
                        
                        time.sleep(0.5)  # API rate limiting
                        
                    except Exception as e:
                        print(f"⚠️ Error processing image {img_data.get('id', 'unknown')}: {e}")
                        continue
                
                page += 1
                time.sleep(1)  # Rate limiting between pages
                
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break
        
        return collected
    
    def collect_from_pixabay_scraping(self, count: int = 50) -> int:
        """Collect from Pixabay via web scraping (no API key needed)"""
        print(f"📸 Collecting {count} images from Pixabay via scraping...")
        collected = 0
        page = 1
        
        search_terms = [
            "black+cat", "dark+cat", "black+kitten", 
            "black+feline", "domestic+black+cat"
        ]
        
        while collected < count and page <= 5:
            try:
                search_term = random.choice(search_terms)
                
                # Pixabay search URL
                url = f"https://pixabay.com/images/search/{search_term}/"
                params = {
                    'page': page,
                    'min_width': 300,
                    'min_height': 300,
                    'category': 'animals'
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                if response.status_code != 200:
                    print(f"⚠️ HTTP {response.status_code} from Pixabay")
                    break
                
                # Simple HTML parsing to find image URLs
                import re
                # Look for Pixabay CDN image URLs
                img_pattern = r'https://cdn\.pixabay\.com/photo/[^"]+\.jpg'
                image_urls = re.findall(img_pattern, response.text)
                
                # Filter for web format (medium size)
                webformat_urls = [url for url in image_urls if '_640.' in url or '_960.' in url]
                
                if not webformat_urls:
                    print(f"📭 No suitable images found for '{search_term}' on page {page}")
                    page += 1
                    continue
                
                print(f"📄 Processing page {page} ({len(webformat_urls)} images for '{search_term}')...")
                
                for i, image_url in enumerate(webformat_urls[:10]):  # Limit per page
                    if collected >= count:
                        break
                    
                    try:
                        # Extract image ID from URL if possible
                        img_id_match = re.search(r'/(\d+)-', image_url)
                        image_id = img_id_match.group(1) if img_id_match else f"{page}_{i}"
                        
                        filename = f"pixabay_cat_{image_id}.jpg"
                        
                        success, reason = self.download_and_validate_image(image_url, filename)
                        
                        if success:
                            collected += 1
                            print(f"✅ {collected}/{count}: {filename} - {reason}")
                        else:
                            print(f"❌ Skipped image {i+1}: {reason}")
                        
                        time.sleep(1.5)  # Conservative rate limiting for scraping
                        
                    except Exception as e:
                        print(f"⚠️ Error processing image {i+1}: {e}")
                        continue
                
                page += 1
                time.sleep(2)  # Rate limiting between pages
                
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break
        
        return collected
    
    def download_and_validate_image(self, image_url: str, filename: str) -> Tuple[bool, str]:
        """Download image and validate quality"""
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            # Save temporarily
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Validate image
            if not self._is_valid_image(temp_path):
                temp_path.unlink()
                return False, "Invalid image format"
            
            # Check for duplicates
            file_hash = self._get_file_hash(temp_path)
            if not file_hash:
                temp_path.unlink()
                return False, "Could not hash file"
            
            if file_hash in self.file_hashes:
                temp_path.unlink()
                return False, "Duplicate image"
            
            # Check quality
            is_good_quality, quality_reason = self._is_good_quality_image(temp_path)
            if not is_good_quality:
                temp_path.unlink()
                return False, f"Quality check failed: {quality_reason}"
            
            # All checks passed - move to final location
            temp_path.rename(final_path)
            self.file_hashes.add(file_hash)
            
            return True, quality_reason
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Download error: {e}"

def main():
    """Main function to collect high-quality cat images"""
    print("🐱 High-Quality Pixabay Cat Photo Collector")
    print("=" * 50)
    
    collector = PixabayCollector()
    
    # Ask user for collection preference
    print("\nCollection options:")
    print("1. Pixabay API (requires free API key) - Best quality and reliability")
    print("2. Pixabay scraping (no API key) - Good quality, may be slower")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        api_key = input("Enter Pixabay API key (get free at https://pixabay.com/api/docs/): ").strip()
        if api_key:
            collected = collector.collect_from_pixabay_api(count=50, api_key=api_key)
        else:
            print("No API key provided, falling back to scraping...")
            collected = collector.collect_from_pixabay_scraping(count=50)
    else:
        collected = collector.collect_from_pixabay_scraping(count=50)
    
    print(f"\n🎉 Collection complete!")
    print(f"📸 Collected {collected} high-quality cat images from Pixabay")
    print(f"💡 Images saved to: {collector.output_dir}")

if __name__ == "__main__":
    main()
