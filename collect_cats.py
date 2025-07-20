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
    """Modern image collector using requests and PIL libraries."""
    
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_image(self, url, filename):
        """Download image using urllib."""
        try:
            # Create request with headers
            req = urllib.request.Request(url, headers=self.headers)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status != 200:
                    print(f"❌ Failed to download {filename}: HTTP {response.status}")
                    return False
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    print(f"⚠️ Not an image: {filename} ({content_type})")
                    return False
                
                # Read and save image data
                image_data = response.read()
                
                # Check minimum size
                if len(image_data) < 1000:
                    print(f"⚠️ Image too small: {filename}")
                    return False
                
                # Save to file
                filepath = self.output_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                print(f"✅ Downloaded: {filename} ({len(image_data)} bytes)")
                return True
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def search_unsplash(self, query, count=20):
        """Search Unsplash for images."""
        print(f"🔍 Searching Unsplash for '{query}'...")
        
        try:
            # Unsplash search URL
            search_url = f"https://unsplash.com/napi/search/photos?query={urllib.parse.quote(query)}&per_page={count}"
            
            req = urllib.request.Request(search_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    print(f"❌ Unsplash search failed: HTTP {response.status}")
                    return []
                
                data = json.loads(response.read().decode())
                results = data.get('results', [])
                
                image_urls = []
                for photo in results:
                    img_url = photo.get('urls', {}).get('regular')
                    if img_url:
                        image_urls.append(img_url)
                
                print(f"📋 Found {len(image_urls)} Unsplash images")
                return image_urls
                
        except Exception as e:
            print(f"❌ Unsplash search error: {e}")
            return []
    
    def search_pixabay(self, query, count=20):
        """Search Pixabay for images (no API key needed for basic search)."""
        print(f"🔍 Searching Pixabay for '{query}'...")
        
        try:
            # Pixabay search URL (using their public endpoint)
            search_url = f"https://pixabay.com/api/?key=9656065-a4094594c34f9ac14c7fc4c39&q={urllib.parse.quote(query)}&image_type=photo&per_page={count}&safesearch=true"
            
            # Note: This uses a demo API key - in production you'd want your own
            req = urllib.request.Request(search_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    print(f"❌ Pixabay search failed: HTTP {response.status}")
                    return []
                
                data = json.loads(response.read().decode())
                hits = data.get('hits', [])
                
                image_urls = []
                for hit in hits:
                    img_url = hit.get('webformatURL')
                    if img_url:
                        image_urls.append(img_url)
                
                print(f"📋 Found {len(image_urls)} Pixabay images")
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
        
        # Search terms for black cats
        search_terms = [
            "black cat",
            "black kitten", 
            "black domestic cat",
            "dark cat"
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
    train_collector = SimpleImageCollector("data/train/other_cats")
    train_collected = train_collector.collect_images(train_needed)
    
    print(f"\n⏱️ Waiting 30 seconds before collecting test images...")
    time.sleep(30)  # Be extra respectful between large batches
    
    # Collect test images
    print(f"\n🧪 Collecting {test_needed} images for test set...")
    test_collector = SimpleImageCollector("data/test/other_cats")
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
