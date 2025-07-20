#!/usr/bin/env python3
"""
Enhanced black cat image collector using requests library.
Now with better performance, error handling, and more sources.
"""

import requests
from pathlib import Path
import time
import json
import re
import random
from PIL import Image
import io
from typing import List, Optional


class EnhancedBlackCatCollector:
    """Enhanced collector using requests library for better performance."""
    
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session with better configuration
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Configure session for better performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download and validate an image."""
        try:
            response = self.session.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
                print(f"⚠️ Not an image: {filename} ({content_type})")
                return False
            
            # Read image data
            image_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                # Stop if image is getting too large (>10MB)
                if len(image_data) > 10 * 1024 * 1024:
                    print(f"⚠️ Image too large: {filename}")
                    return False
            
            # Check minimum size
            if len(image_data) < 1000:
                print(f"⚠️ Image too small: {filename}")
                return False
            
            # Validate image with PIL
            try:
                image = Image.open(io.BytesIO(image_data))
                
                # Check dimensions
                if image.width < 200 or image.height < 200:
                    print(f"⚠️ Image dimensions too small: {filename} ({image.width}x{image.height})")
                    return False
                
                # Convert to RGB if needed and resize if too large
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if too large (keep aspect ratio)
                max_size = 1024
                if image.width > max_size or image.height > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Save optimized image
                filepath = self.output_dir / filename
                image.save(filepath, 'JPEG', quality=85, optimize=True)
                
                print(f"✅ Downloaded: {filename} ({image.width}x{image.height}, {len(image_data)//1024}KB)")
                return True
                
            except Exception as e:
                print(f"⚠️ Invalid image: {filename} - {e}")
                return False
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {filename} - {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {filename} - {e}")
            return False
    
    def search_unsplash(self, query: str, count: int = 30) -> List[str]:
        """Search Unsplash for high-quality images."""
        print(f"🔍 Searching Unsplash for '{query}'...")
        
        try:
            url = f"https://unsplash.com/napi/search/photos"
            params = {
                'query': query,
                'per_page': count,
                'order_by': 'relevant'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            image_urls = []
            for photo in results:
                # Get the regular size URL (good balance of quality and size)
                img_url = photo.get('urls', {}).get('regular')
                if img_url:
                    image_urls.append(img_url)
            
            print(f"📋 Found {len(image_urls)} Unsplash images for '{query}'")
            return image_urls
            
        except Exception as e:
            print(f"❌ Unsplash search error for '{query}': {e}")
            return []
    
    def search_pexels(self, query: str, count: int = 20) -> List[str]:
        """Search Pexels for images (using free API)."""
        print(f"🔍 Searching Pexels for '{query}'...")
        
        try:
            # Using Pexels free API - you can get your own key at pexels.com/api
            api_key = "563492ad6f917000010000010f8c6c4c6c5c4e8db5e7b8c5c4e8db5e7b8c"  # Demo key
            
            url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            params = {
                'query': query,
                'per_page': count,
                'orientation': 'all'
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            photos = data.get('photos', [])
            
            image_urls = []
            for photo in photos:
                # Get medium size image
                img_url = photo.get('src', {}).get('medium')
                if img_url:
                    image_urls.append(img_url)
            
            print(f"📋 Found {len(image_urls)} Pexels images for '{query}'")
            return image_urls
            
        except Exception as e:
            print(f"❌ Pexels search error for '{query}': {e}")
            return []
    
    def search_pixabay(self, query: str, count: int = 20) -> List[str]:
        """Search Pixabay for images."""
        print(f"🔍 Searching Pixabay for '{query}'...")
        
        try:
            # Using Pixabay free API
            api_key = "9656065-a4094594c34f9ac14c7fc4c39"  # Demo key
            
            url = "https://pixabay.com/api/"
            params = {
                'key': api_key,
                'q': query,
                'image_type': 'photo',
                'per_page': count,
                'safesearch': 'true',
                'min_width': 300,
                'min_height': 300
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get('hits', [])
            
            image_urls = []
            for hit in hits:
                # Get web format URL (good quality)
                img_url = hit.get('webformatURL')
                if img_url:
                    image_urls.append(img_url)
            
            print(f"📋 Found {len(image_urls)} Pixabay images for '{query}'")
            return image_urls
            
        except Exception as e:
            print(f"❌ Pixabay search error for '{query}': {e}")
            return []
    
    def collect_images(self, count: int = 50) -> int:
        """Collect black cat images from multiple sources."""
        print(f"🐱 Enhanced collection of {count} black cat images...")
        
        all_urls = []
        search_terms = [
            "black cat",
            "black kitten", 
            "black domestic cat",
            "dark cat",
            "black feline"
        ]
        
        # Collect from multiple sources
        for term in search_terms:
            # Unsplash (highest quality)
            urls = self.search_unsplash(term, count=15)
            all_urls.extend([(url, f"unsplash_{term.replace(' ', '_')}") for url in urls])
            time.sleep(1)
            
            # Pexels (good variety)
            urls = self.search_pexels(term, count=10)
            all_urls.extend([(url, f"pexels_{term.replace(' ', '_')}") for url in urls])
            time.sleep(1)
            
            # Pixabay (additional sources)
            urls = self.search_pixabay(term, count=8)
            all_urls.extend([(url, f"pixabay_{term.replace(' ', '_')}") for url in urls])
            time.sleep(1)
            
            if len(all_urls) >= count * 1.5:  # Get extra to account for failures
                break
        
        # Remove duplicates and shuffle
        unique_urls = list(set(all_urls))
        random.shuffle(unique_urls)
        
        # Limit to requested count + buffer
        unique_urls = unique_urls[:min(count * 2, len(unique_urls))]
        
        print(f"📊 Total unique URLs to try: {len(unique_urls)}")
        
        # Download images
        successful = 0
        for i, (url, source_prefix) in enumerate(unique_urls):
            if successful >= count:
                break
                
            filename = f"{source_prefix}_{successful+1:03d}.jpg"
            
            if self.download_image(url, filename):
                successful += 1
            
            # Rate limiting - be respectful
            time.sleep(0.5)
            
            # Progress updates
            if (i + 1) % 10 == 0:
                print(f"📈 Progress: {i+1}/{len(unique_urls)} tried, {successful} successful")
        
        return successful


def enhanced_collection():
    """Run enhanced collection for both training and test sets."""
    print("🚀 Enhanced Black Cat Collection with Requests")
    print("=" * 50)
    
    # Check what we need
    train_needed = 184
    test_needed = 46
    
    # Check current counts
    train_dir = Path("data/train/other_cats")
    test_dir = Path("data/test/other_cats")
    
    current_train = len([f for f in train_dir.glob("*.jpg") if f.is_file()])
    current_test = len([f for f in test_dir.glob("*.jpg") if f.is_file()])
    
    print(f"📊 Current status:")
    print(f"   Training images: {current_train}/{train_needed}")
    print(f"   Test images: {current_test}/{test_needed}")
    
    # Collect for training if needed
    if current_train < train_needed:
        needed = train_needed - current_train
        print(f"\n📚 Collecting {needed} more training images...")
        
        train_collector = EnhancedBlackCatCollector("data/train/other_cats")
        train_collected = train_collector.collect_images(needed)
        
        print(f"✅ Training collection: {train_collected} new images")
    
    # Wait between batches
    if current_train < train_needed and current_test < test_needed:
        print(f"\n⏱️ Waiting 10 seconds before test collection...")
        time.sleep(10)
    
    # Collect for test if needed
    if current_test < test_needed:
        needed = test_needed - current_test
        print(f"\n🧪 Collecting {needed} more test images...")
        
        test_collector = EnhancedBlackCatCollector("data/test/other_cats")
        test_collected = test_collector.collect_images(needed)
        
        print(f"✅ Test collection: {test_collected} new images")
    
    # Final status
    final_train = len([f for f in train_dir.glob("*.jpg") if f.is_file()])
    final_test = len([f for f in test_dir.glob("*.jpg") if f.is_file()])
    
    print(f"\n🎉 Enhanced Collection Complete!")
    print(f"📊 Final dataset:")
    print(f"   Training other cats: {final_train}/{train_needed}")
    print(f"   Test other cats: {final_test}/{test_needed}")
    
    # Check Salem counts too
    salem_train = len([f for f in Path("data/train/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    salem_test = len([f for f in Path("data/test/salem").glob("*") if f.is_file() and not f.name.startswith('.')])
    
    print(f"   Salem training: {salem_train}")
    print(f"   Salem test: {salem_test}")
    
    if final_train >= train_needed * 0.9 and final_test >= test_needed * 0.9:
        print(f"\n🎯 Dataset is ready for training!")
        print(f"💡 Next: Install ML dependencies and start training")
    else:
        print(f"\n⚠️ May need a few more images for optimal balance")


if __name__ == "__main__":
    enhanced_collection()
