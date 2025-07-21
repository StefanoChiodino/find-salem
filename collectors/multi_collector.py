#!/usr/bin/env python3
"""
Multi-Source Black Cat Collector
Explores alternative photo sources beyond exhausted Unsplash/Pixabay APIs
"""

import requests
from pathlib import Path
import time
import hashlib
from PIL import Image
import json
import random
from urllib.parse import urlencode, quote_plus
import re

class MultiSourceCollector:
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Load existing image hashes for duplicate detection
        self.existing_hashes = self._load_existing_hashes()
        self.downloaded_urls = set()
        
        print(f"🐱 Multi-Source Black Cat Collector")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔍 Loaded {len(self.existing_hashes)} existing image hashes")

    def _load_existing_hashes(self):
        """Load MD5 hashes of existing images for duplicate detection"""
        hashes = set()
        
        # Check all data directories for existing images
        data_dirs = [
            "data/train/other_cats",
            "data/test/other_cats", 
            "demo_samples/other_cats"
        ]
        
        for data_dir in data_dirs:
            dir_path = Path(data_dir)
            if dir_path.exists():
                for img_file in dir_path.glob("*.jpg"):
                    img_hash = self._get_file_hash(img_file)
                    if img_hash:
                        hashes.add(img_hash)
        
        return hashes

    def _get_file_hash(self, file_path):
        """Get MD5 hash for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def _is_duplicate(self, file_path):
        """Check if image is a duplicate"""
        file_hash = self._get_file_hash(file_path)
        return file_hash in self.existing_hashes if file_hash else False

    def _is_valid_black_cat_image(self, file_path):
        """Basic validation for black cat images"""
        try:
            with Image.open(file_path) as img:
                # Basic size check
                if img.width < 100 or img.height < 100:
                    return False
                
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Basic darkness check (black cats should be predominantly dark)
                # Sample some pixels and check average brightness
                pixels = list(img.getdata())
                if len(pixels) > 1000:
                    sample_pixels = random.sample(pixels, 1000)
                else:
                    sample_pixels = pixels
                
                avg_brightness = sum(sum(pixel) for pixel in sample_pixels) / (len(sample_pixels) * 3)
                
                # Should be relatively dark (lower threshold than before for more variety)
                return avg_brightness < 150
                
        except Exception:
            return False

    def collect_from_flickr_api(self, count=20):
        """Try Flickr API (requires API key but has free tier)"""
        print("🔍 Searching Flickr API...")
        
        # Note: This would require a Flickr API key
        # For now, return empty list but show the method
        print("⚠️  Flickr API requires API key - implement when available")
        return []

    def collect_from_reddit(self, count=20):
        """Collect from Reddit using public API"""
        print("🔍 Searching Reddit r/blackcats...")
        
        try:
            # Reddit public API (no auth needed for read-only)
            url = "https://www.reddit.com/r/blackcats.json"
            params = {
                'limit': count * 3,  # Get more to filter
                'sort': 'hot'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ Reddit API failed: {response.status_code}")
                return []
            
            data = response.json()
            urls = []
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Look for image URLs
                if 'url' in post_data:
                    url = post_data['url']
                    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        urls.append(url)
                
                # Also check preview images
                if 'preview' in post_data:
                    preview = post_data['preview']
                    if 'images' in preview and preview['images']:
                        img = preview['images'][0]
                        if 'source' in img:
                            urls.append(img['source']['url'])
            
            print(f"📋 Found {len(urls)} Reddit image URLs")
            return urls[:count]
            
        except Exception as e:
            print(f"❌ Reddit collection failed: {e}")
            return []

    def collect_from_imgur(self, count=20):
        """Try to collect from Imgur public galleries"""
        print("🔍 Searching Imgur...")
        
        try:
            # Search Imgur public galleries
            url = "https://api.imgur.com/3/gallery/search/time/all/0"
            params = {'q': 'black cat'}
            
            # Imgur API requires client ID, but we can try without
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 403:
                print("⚠️  Imgur API requires client ID - implement when available")
                return []
            
            print(f"📋 Imgur search response: {response.status_code}")
            return []
            
        except Exception as e:
            print(f"❌ Imgur collection failed: {e}")
            return []

    def collect_from_google_images(self, count=20):
        """Try alternative Google Images search"""
        print("🔍 Trying alternative Google Images approach...")
        
        try:
            # Use Google Custom Search API approach (requires API key)
            print("⚠️  Google Custom Search API requires API key - implement when available")
            
            # Alternative: try direct image search with different approach
            search_terms = [
                "black cat photo",
                "schwarze katze",  # German
                "gato negro",      # Spanish  
                "chat noir",       # French
                "gatto nero",      # Italian
            ]
            
            urls = []
            for term in search_terms[:2]:  # Try first 2 to avoid being blocked
                print(f"   Searching: {term}")
                time.sleep(2)  # Be respectful
                # This would need implementation of image search scraping
                # For now, return empty list
            
            return urls
            
        except Exception as e:
            print(f"❌ Google Images collection failed: {e}")
            return []

    def collect_from_alternative_apis(self, count=20):
        """Try other free image APIs"""
        print("🔍 Trying alternative free APIs...")
        
        urls = []
        
        # Try Lorem Picsum with specific dimensions (though quality may vary)
        # This is just an example - Lorem Picsum doesn't have cat-specific images
        
        # Try other public APIs that might have animal images
        apis_to_try = [
            # Add other free image APIs here
            # "https://api.example.com/images?q=black+cat"
        ]
        
        for api_url in apis_to_try:
            try:
                response = self.session.get(api_url, timeout=10)
                # Process API response for image URLs
                # Implementation would depend on specific API format
                pass
            except Exception:
                continue
        
        print(f"📋 Found {len(urls)} alternative API URLs")
        return urls

    def download_image(self, url, filename):
        """Download and validate image"""
        if url in self.downloaded_urls:
            return False
        
        self.downloaded_urls.add(url)
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            response = self.session.get(url, timeout=15, stream=True)
            if response.status_code != 200:
                return False
            
            # Download image
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate image
            if not self._is_valid_black_cat_image(temp_path):
                temp_path.unlink()
                print(f"🗑️ Invalid image: {filename}")
                return False
            
            # Check for duplicates
            if self._is_duplicate(temp_path):
                temp_path.unlink()
                print(f"🗑️ Duplicate image: {filename}")
                return False
            
            # Move to final location
            temp_path.rename(final_path)
            file_size = final_path.stat().st_size
            
            # Add hash to existing hashes
            file_hash = self._get_file_hash(final_path)
            if file_hash:
                self.existing_hashes.add(file_hash)
            
            print(f"✅ Downloaded: {filename} ({file_size} bytes)")
            return True
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"❌ Failed to download {filename}: {e}")
            return False

    def collect_images(self, target_count=50):
        """Collect black cat images from multiple sources"""
        print(f"🐱 Collecting {target_count} black cat images from multiple sources...")
        print("⚠️  SAFETY: This collector NEVER touches Salem photos!")
        
        all_urls = []
        successful_downloads = 0
        
        # Try different sources
        sources = [
            ("Reddit", lambda: self.collect_from_reddit(target_count // 2)),
            ("Flickr", lambda: self.collect_from_flickr_api(target_count // 4)),
            ("Imgur", lambda: self.collect_from_imgur(target_count // 4)),
            ("Google Images", lambda: self.collect_from_google_images(target_count // 4)),
            ("Alternative APIs", lambda: self.collect_from_alternative_apis(target_count // 4)),
        ]
        
        for source_name, source_func in sources:
            print(f"\n🔍 Trying source: {source_name}")
            try:
                urls = source_func()
                all_urls.extend(urls)
                print(f"📋 {source_name} contributed {len(urls)} URLs")
                time.sleep(3)  # Be respectful between sources
            except Exception as e:
                print(f"❌ {source_name} failed: {e}")
        
        if not all_urls:
            print("❌ No URLs found from any source")
            return 0
        
        print(f"\n📊 Total URLs to download: {len(all_urls)}")
        
        # Download images
        for i, url in enumerate(all_urls[:target_count], 1):
            filename = f"multi_source_{i:03d}.jpg"
            
            if self.download_image(url, filename):
                successful_downloads += 1
            
            if i % 10 == 0:
                print(f"📈 Progress: {i}/{len(all_urls)} processed, {successful_downloads} successful")
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Successfully downloaded: {successful_downloads}/{len(all_urls)} images")
        
        return successful_downloads

def main():
    """Main collection function"""
    collector = MultiSourceCollector()
    
    print("🚀 Starting multi-source black cat collection...")
    result = collector.collect_images(target_count=50)
    
    if result > 0:
        print(f"\n✅ Successfully collected {result} new black cat images!")
        print("💡 You can now retrain the models with more balanced data")
    else:
        print("\n⚠️  No new images collected - may need API keys or manual collection")
        print("💡 Consider implementing API keys for Flickr, Google Custom Search, etc.")

if __name__ == "__main__":
    main()
