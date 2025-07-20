#!/usr/bin/env python3
"""
Alternative Black Cat Collector
NEVER TOUCHES SALEM PHOTOS - Only collects other black cat images
Explores multiple sources: DuckDuckGo, Bing, direct URLs, etc.
"""

import os
import hashlib
import requests
import time
import random
from pathlib import Path
from typing import Set, Tuple, List
from PIL import Image
import json
import re
from urllib.parse import urljoin, urlparse

class AlternativeCatCollector:
    def __init__(self, output_dir: str = "data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_hashes: Set[str] = set()
        
        print("⚠️  SAFETY NOTICE: This collector NEVER touches Salem photos!")
        print("    Only collects other black cat photos for balancing dataset")
        
        # Load existing hashes
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing file hashes to avoid duplicates"""
        print("🔍 Loading existing image hashes for duplicate detection...")
        
        # Only check other_cats directories - NEVER touch salem folders
        for data_path in ["data/train/other_cats", "data/test/other_cats", "demo_samples/other_cats"]:
            data_dir = Path(data_path)
            if data_dir.exists():
                for img_file in data_dir.glob("*.jpg"):
                    if self._is_valid_image(img_file):
                        file_hash = self._get_file_hash(img_file)
                        if file_hash:
                            self.file_hashes.add(file_hash)
        
        print(f"✅ Loaded {len(self.file_hashes)} existing other cat image hashes")
    
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
    
    def _is_good_black_cat_image(self, file_path: Path) -> Tuple[bool, str]:
        """Check if image is good quality and likely a black cat"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Size requirements
                if width < 150 or height < 150:
                    return False, "Too small"
                
                # Convert for analysis
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Resize for fast analysis
                img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
                pixels = list(img_small.getdata())
                
                if img.mode == 'RGB':
                    # Calculate average brightness for RGB
                    avg_brightness = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
                else:
                    # Grayscale
                    avg_brightness = sum(pixels) / len(pixels)
                
                # We want darker images (likely black cats) but not solid black
                if avg_brightness < 30:
                    return False, "Too dark/solid black"
                
                if avg_brightness > 120:
                    return False, "Too bright/light"
                
                return True, f"Good black cat candidate {width}x{height}, brightness: {avg_brightness:.1f}"
                
        except Exception as e:
            return False, f"Analysis error: {e}"
    
    def collect_from_duckduckgo(self, count: int = 30) -> int:
        """Collect black cat images from DuckDuckGo image search"""
        print(f"🦆 Collecting {count} images from DuckDuckGo...")
        collected = 0
        
        search_terms = [
            "black cat", "black kitten", "dark cat", "black feline",
            "black domestic cat", "black cat portrait", "black cat face"
        ]
        
        for search_term in search_terms:
            if collected >= count:
                break
                
            try:
                print(f"🔍 Searching DuckDuckGo for: '{search_term}'")
                
                # DuckDuckGo image search URL
                url = "https://duckduckgo.com/"
                params = {
                    'q': search_term,
                    'iax': 'images',
                    'ia': 'images'
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                session = requests.Session()
                
                # Get the search page first
                response = session.get(url, params=params, headers=headers, timeout=15)
                if response.status_code != 200:
                    print(f"⚠️ DuckDuckGo search failed: {response.status_code}")
                    continue
                
                # Look for image URLs in the response
                # DuckDuckGo loads images via JavaScript, so we look for any image URLs in the HTML
                img_urls = re.findall(r'https://[^"]+\.(?:jpg|jpeg|png)', response.text)
                
                # Filter and deduplicate URLs
                unique_urls = []
                seen_domains = set()
                for url in img_urls:
                    domain = urlparse(url).netlify
                    if len(unique_urls) < 10 and domain not in seen_domains:
                        unique_urls.append(url)
                        seen_domains.add(domain)
                
                print(f"📄 Found {len(unique_urls)} potential image URLs")
                
                for i, img_url in enumerate(unique_urls):
                    if collected >= count:
                        break
                    
                    try:
                        filename = f"ddg_cat_{search_term.replace(' ', '_')}_{i:02d}.jpg"
                        success, reason = self.download_and_validate_image(img_url, filename)
                        
                        if success:
                            collected += 1
                            print(f"✅ {collected}/{count}: {filename} - {reason}")
                        else:
                            print(f"❌ Skipped: {reason}")
                        
                        time.sleep(2)  # Conservative rate limiting
                        
                    except Exception as e:
                        print(f"⚠️ Error with image {i+1}: {e}")
                        continue
                
                time.sleep(3)  # Delay between search terms
                
            except Exception as e:
                print(f"❌ Error searching for '{search_term}': {e}")
                continue
        
        return collected
    
    def collect_from_bing(self, count: int = 20) -> int:
        """Collect from Bing image search"""
        print(f"🔍 Collecting {count} images from Bing...")
        collected = 0
        
        search_terms = ["black cat", "black kitten", "dark feline"]
        
        for search_term in search_terms:
            if collected >= count:
                break
            
            try:
                url = "https://www.bing.com/images/search"
                params = {
                    'q': search_term,
                    'form': 'HDRSC2',
                    'first': 1
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue
                
                # Look for image URLs in Bing response
                img_pattern = r'"murl":"([^"]+\.(?:jpg|jpeg|png))"'
                img_urls = re.findall(img_pattern, response.text)
                
                print(f"📄 Found {len(img_urls[:8])} Bing image URLs")
                
                for i, img_url in enumerate(img_urls[:8]):  # Limit per search
                    if collected >= count:
                        break
                    
                    try:
                        # Clean up URL encoding
                        img_url = img_url.replace('\\u002f', '/').replace('\\', '')
                        filename = f"bing_cat_{search_term.replace(' ', '_')}_{i:02d}.jpg"
                        
                        success, reason = self.download_and_validate_image(img_url, filename)
                        
                        if success:
                            collected += 1
                            print(f"✅ {collected}/{count}: {filename} - {reason}")
                        else:
                            print(f"❌ Skipped: {reason}")
                        
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"⚠️ Error with Bing image {i+1}: {e}")
                        continue
                
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Error with Bing search '{search_term}': {e}")
                continue
        
        return collected
    
    def collect_from_direct_sources(self, count: int = 20) -> int:
        """Collect from known cat photo websites"""
        print(f"🌐 Collecting {count} images from direct sources...")
        collected = 0
        
        # Some cat photo websites that often have black cats
        base_urls = [
            "https://placekitten.com/",  # Placeholder cat images
            "https://http.cat/",  # HTTP status cats (some black)
        ]
        
        # Try PlaceKitten with different sizes
        sizes = [(300, 300), (400, 400), (350, 350), (320, 320), (380, 380)]
        
        for i, (width, height) in enumerate(sizes):
            if collected >= count:
                break
            
            try:
                url = f"https://placekitten.com/{width}/{height}"
                filename = f"placekitten_{width}x{height}_{i}.jpg"
                
                success, reason = self.download_and_validate_image(url, filename)
                
                if success:
                    collected += 1
                    print(f"✅ {collected}/{count}: {filename} - {reason}")
                else:
                    print(f"❌ Skipped PlaceKitten {width}x{height}: {reason}")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Error with PlaceKitten {width}x{height}: {e}")
                continue
        
        return collected
    
    def download_and_validate_image(self, image_url: str, filename: str) -> Tuple[bool, str]:
        """Download and validate image - NEVER touches Salem photos"""
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(image_url, headers=headers, timeout=10, stream=True)
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, "Not an image"
            
            # Save temporarily
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate
            if not self._is_valid_image(temp_path):
                temp_path.unlink()
                return False, "Invalid image"
            
            # Check for duplicates
            file_hash = self._get_file_hash(temp_path)
            if not file_hash or file_hash in self.file_hashes:
                temp_path.unlink()
                return False, "Duplicate"
            
            # Quality check
            is_good_quality, quality_reason = self._is_good_black_cat_image(temp_path)
            if not is_good_quality:
                temp_path.unlink()
                return False, quality_reason
            
            # All good - move to final location
            temp_path.rename(final_path)
            self.file_hashes.add(file_hash)
            
            return True, quality_reason
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Error: {e}"

def main():
    """Collect black cat images from multiple alternative sources"""
    print("🐱 Alternative Black Cat Collector")
    print("⚠️  SAFETY: This collector NEVER touches Salem photos!")
    print("=" * 50)
    
    collector = AlternativeCatCollector()
    
    total_collected = 0
    
    # Try multiple sources
    print("\n🦆 Attempting DuckDuckGo collection...")
    ddg_collected = collector.collect_from_duckduckgo(count=25)
    total_collected += ddg_collected
    
    print(f"\n🔍 Attempting Bing collection...")
    bing_collected = collector.collect_from_bing(count=15)  
    total_collected += bing_collected
    
    print(f"\n🌐 Attempting direct sources...")
    direct_collected = collector.collect_from_direct_sources(count=10)
    total_collected += direct_collected
    
    print(f"\n🎉 Collection complete!")
    print(f"📸 Total collected: {total_collected} new black cat images")
    print(f"   - DuckDuckGo: {ddg_collected}")
    print(f"   - Bing: {bing_collected}")  
    print(f"   - Direct: {direct_collected}")
    print(f"💾 Images saved to: {collector.output_dir}")
    print(f"⚠️  Salem photos remain completely untouched!")

if __name__ == "__main__":
    main()
