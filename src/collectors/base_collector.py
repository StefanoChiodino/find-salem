#!/usr/bin/env python3
"""
Modern black cat image collector using requests library.
Requires: pip install requests pillow
"""

import requests
import json
import time
import os
from pathlib import Path
import random
from PIL import Image
import io


class ModernImageCollector:
    """Modern image collector using requests and PIL libraries with duplicate detection."""
    
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Duplicate detection - NEVER touches Salem photos
        self.file_hashes = set()
        self.downloaded_urls = set() 
        print("⚠️  SAFETY: This collector NEVER touches Salem photos!")
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Load existing image hashes for duplicate detection"""
        print("🔍 Loading existing other cat image hashes...")
        
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
    
    def _get_file_hash(self, file_path):
        """Get MD5 hash for duplicate detection"""
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    def _is_valid_image(self, file_path):
        """Check if file is a valid image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
                return True
        except Exception:
            return False
    
    def download_image(self, url, filename):
        """Download image with duplicate detection and validation"""
        # Skip if we've tried this URL before
        if url in self.downloaded_urls:
            return False
        
        self.downloaded_urls.add(url)
        
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            # Download using requests session
            response = self.session.get(url, timeout=15, stream=True)
            
            if response.status_code != 200:
                print(f"❌ Failed to download {filename}: HTTP {response.status_code}")
                return False
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                print(f"⚠️ Not an image: {filename} ({content_type})")
                return False
            
            # Download to temp file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check minimum size
            if temp_path.stat().st_size < 1000:
                temp_path.unlink()
                print(f"⚠️ Image too small: {filename}")
                return False
            
            # Validate image
            if not self._is_valid_image(temp_path):
                temp_path.unlink()
                print(f"⚠️ Invalid image: {filename}")
                return False
            
            # Check for duplicates using file hash
            file_hash = self._get_file_hash(temp_path)
            if not file_hash:
                temp_path.unlink()
                print(f"⚠️ Could not hash: {filename}")
                return False
            
            if file_hash in self.file_hashes:
                temp_path.unlink()
                print(f"🗑️ Duplicate image: {filename}")
                return False
            
            # All checks passed - move to final location
            temp_path.rename(final_path)
            self.file_hashes.add(file_hash)
            
            print(f"✅ Downloaded: {filename} ({final_path.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def search_unsplash(self, query, count=20):
        """Search Unsplash for images using requests session."""
        print(f"🔍 Searching Unsplash for '{query}'...")
        
        try:
            from urllib.parse import quote
            # Unsplash search URL
            search_url = f"https://unsplash.com/napi/search/photos?query={quote(query)}&per_page={count}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Unsplash search failed: HTTP {response.status_code}")
                return []
            
            data = response.json()
            results = data.get('results', [])
            
            image_urls = []
            for photo in results:
                img_url = photo.get('urls', {}).get('regular')
                if img_url and img_url not in self.downloaded_urls:
                    image_urls.append(img_url)
            
            print(f"📋 Found {len(image_urls)} new Unsplash images")
            return image_urls
            
        except Exception as e:
            print(f"❌ Unsplash search error: {e}")
            return []
    
    def search_pixabay(self, query, count=20):
        """Search Pixabay for images using requests session."""
        print(f"🔍 Searching Pixabay for '{query}'...")
        
        try:
            from urllib.parse import quote
            # Pixabay search URL (using demo API key - get your own for production)
            search_url = f"https://pixabay.com/api/?key=9656065-a4094594c34f9ac14c7fc4c39&q={quote(query)}&image_type=photo&per_page={count}&safesearch=true&category=animals"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Pixabay search failed: HTTP {response.status_code}")
                return []
            
            data = response.json()
            hits = data.get('hits', [])
            
            image_urls = []
            for hit in hits:
                img_url = hit.get('webformatURL')
                if img_url and img_url not in self.downloaded_urls:
                    image_urls.append(img_url)
            
            print(f"📋 Found {len(image_urls)} new Pixabay images")
            return image_urls
            
        except Exception as e:
            print(f"❌ Pixabay search error: {e}")
            return []
    
    def get_direct_urls(self):
        """Get a curated list of direct black cat image URLs."""
        # These are example URLs - you would replace with actual image URLs
        direct_urls = [
            # Add actual black cat image URLs here if you find good sources
        ]
        
        print(f"📋 Found {len(direct_urls)} direct URLs")
        return direct_urls
    
    def collect_images(self, count=50):
        """Collect black cat images from various sources."""
        print(f"🐱 Collecting {count} black cat images...")
        
        all_urls = []
        
        # Search terms for black cats - maximum focus on quality
        search_terms = [
            "black cat"  # Only use the most specific term
        ]
        
        # Try Unsplash first
        for term in search_terms:
            urls = self.search_unsplash(term, count=15)
            all_urls.extend(urls)
            time.sleep(1)  # Be respectful
            
            if len(all_urls) >= count:
                break
        
        # Try Pixabay if we need more
        if len(all_urls) < count:
            for term in search_terms:
                urls = self.search_pixabay(term, count=10)
                all_urls.extend(urls)
                time.sleep(1)
                
                if len(all_urls) >= count:
                    break
        
        # Add direct URLs if available
        direct_urls = self.get_direct_urls()
        all_urls.extend(direct_urls)
        
        # Remove duplicates and shuffle
        all_urls = list(set(all_urls))
        random.shuffle(all_urls)
        
        # Limit to requested count
        all_urls = all_urls[:count]
        
        print(f"📊 Total URLs to download: {len(all_urls)}")
        
        # Download images
        successful = 0
        for i, url in enumerate(all_urls):
            filename = f"black_cat_{i+1:03d}.jpg"
            
            if self.download_image(url, filename):
                successful += 1
            
            # Rate limiting
            time.sleep(1)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"📈 Progress: {i+1}/{len(all_urls)} processed, {successful} successful")
        
        return successful


def collect_for_training_and_test():
    """Collect images for both training and test sets."""
    print("🐱 Find Salem - Automated Black Cat Collection")
    print("=" * 50)
    
    # We need 184 training + 46 test = 230 total images
    train_needed = 184
    test_needed = 46
    
    # Collect training images
    print(f"\n📚 Collecting {train_needed} images for training set...")
    train_collector = ModernImageCollector("data/train/other_cats")
    train_collected = train_collector.collect_images(train_needed)
    
    print(f"\n⏱️ Waiting 30 seconds before collecting test images...")
    time.sleep(30)  # Be extra respectful between large batches
    
    # Collect test images
    print(f"\n🧪 Collecting {test_needed} images for test set...")
    test_collector = ModernImageCollector("data/test/other_cats")
    test_collected = test_collector.collect_images(test_needed)
    
    # Summary
    print(f"\n🎉 Collection Complete!")
    print(f"📊 Results:")
    print(f"   Training images collected: {train_collected}/{train_needed}")
    print(f"   Test images collected: {test_collected}/{test_needed}")
    print(f"   Total collected: {train_collected + test_collected}/{train_needed + test_needed}")
    
    # Check dataset balance
    print(f"\n📈 Dataset Status:")
    salem_train = len([f for f in Path("data/train/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    salem_test = len([f for f in Path("data/test/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    
    print(f"   Salem training: {salem_train}")
    print(f"   Salem test: {salem_test}")
    print(f"   Other cats training: {train_collected}")
    print(f"   Other cats test: {test_collected}")
    
    if train_collected >= train_needed * 0.8 and test_collected >= test_needed * 0.8:
        print(f"\n✅ Dataset is ready for training!")
        print(f"💡 Next step: Install ML dependencies and start training")
    else:
        print(f"\n⚠️ May need more images for optimal training")
        print(f"💡 You can run this script again or manually add more images")


if __name__ == "__main__":
    collect_for_training_and_test()
