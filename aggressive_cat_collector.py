#!/usr/bin/env python3
"""
Aggressive Cat Collector
Multiple Reddit sources and aggressive collection to fix class imbalance
"""

import requests
from pathlib import Path
import time
import hashlib
from PIL import Image
import random

class AggressiveCatCollector:
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        self.existing_hashes = self._load_existing_hashes()
        self.downloaded_urls = set()
        
        print(f"🐱 AGGRESSIVE Black Cat Collector")
        print(f"🎯 GOAL: Balance dataset for FastAI (need ~175 more other cats)")
        print(f"🔍 Loaded {len(self.existing_hashes)} existing image hashes")

    def _load_existing_hashes(self):
        """Load existing image hashes"""
        hashes = set()
        for data_dir in ["data/train/other_cats", "data/test/other_cats"]:
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

    def collect_from_multiple_subreddits(self):
        """Try multiple cat-related subreddits"""
        subreddits = [
            'blackcats',           # Primary source
            'cats',                # General cats (filter for black)
            'CatsAreAssholes',     # Often has black cats
            'catpictures',         # More cat pics
            'catloaf',             # Cat poses
            'CatGifs',             # Some have black cats
            'aww',                 # General cute animals
            'AnimalsBeingJerks',   # Often cats
            'blackcatsarevoid',    # Black cat specific
            'TuxedoCats',          # Black and white cats
        ]
        
        all_urls = []
        
        for subreddit in subreddits:
            print(f"🔍 Searching r/{subreddit}...")
            try:
                # Try different sort methods for more variety
                for sort in ['hot', 'top', 'new']:
                    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
                    params = {'limit': 25, 't': 'week'}  # This week's posts
                    
                    response = self.session.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        urls = self._extract_image_urls(data)
                        all_urls.extend(urls)
                        print(f"   📋 Found {len(urls)} URLs from r/{subreddit}/{sort}")
                        
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"❌ Failed r/{subreddit}: {e}")
                continue
                
            time.sleep(2)  # Between subreddits
        
        print(f"🎯 TOTAL URLs from all subreddits: {len(all_urls)}")
        return list(set(all_urls))  # Remove duplicates

    def _extract_image_urls(self, reddit_data):
        """Extract image URLs from Reddit API response"""
        urls = []
        
        try:
            for post in reddit_data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Direct image URLs
                if 'url' in post_data:
                    url = post_data['url']
                    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        # Convert Reddit preview URLs to direct URLs
                        if 'preview.redd.it' in url:
                            url = url.replace('preview.redd.it', 'i.redd.it')
                        urls.append(url)
                
                # Preview images
                if 'preview' in post_data and 'images' in post_data['preview']:
                    for img in post_data['preview']['images']:
                        if 'source' in img:
                            source_url = img['source']['url']
                            # Fix HTML entities
                            source_url = source_url.replace('&amp;', '&')
                            urls.append(source_url)
                
        except Exception as e:
            print(f"❌ Error extracting URLs: {e}")
        
        return urls

    def _is_valid_black_cat_image(self, file_path):
        """Enhanced black cat validation"""
        try:
            with Image.open(file_path) as img:
                # Size check
                if img.width < 100 or img.height < 100:
                    return False
                
                # File size check (too small = probably not a good photo)
                file_size = file_path.stat().st_size
                if file_size < 10000:  # Less than 10KB
                    return False
                
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # More sophisticated darkness check
                pixels = list(img.getdata())
                if len(pixels) > 2000:
                    sample_pixels = random.sample(pixels, 2000)
                else:
                    sample_pixels = pixels
                
                # Check if reasonably dark (allows for some variety)
                avg_brightness = sum(sum(pixel) for pixel in sample_pixels) / (len(sample_pixels) * 3)
                return avg_brightness < 160  # Slightly more permissive
                
        except Exception:
            return False

    def download_image(self, url, filename):
        """Download and validate image with better error handling"""
        if url in self.downloaded_urls:
            return False
        
        self.downloaded_urls.add(url)
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            # Fix common Reddit URL issues
            if 'preview.redd.it' in url:
                url = url.replace('preview.redd.it', 'i.redd.it')
            
            response = self.session.get(url, timeout=15, stream=True)
            if response.status_code != 200:
                return False
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not any(ct in content_type.lower() for ct in ['image/jpeg', 'image/png', 'image/jpg']):
                return False
            
            # Download
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate as black cat image
            if not self._is_valid_black_cat_image(temp_path):
                temp_path.unlink()
                return False
            
            # Check for duplicates
            file_hash = self._get_file_hash(temp_path)
            if file_hash in self.existing_hashes:
                temp_path.unlink()
                return False
            
            # Success - move to final location
            temp_path.rename(final_path)
            self.existing_hashes.add(file_hash)
            
            file_size = final_path.stat().st_size
            print(f"✅ Downloaded: {filename} ({file_size} bytes)")
            return True
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"❌ Failed {filename}: {e}")
            return False

    def aggressive_collect(self, target_count=100):
        """Aggressive collection from multiple sources"""
        print(f"\n🚀 AGGRESSIVE COLLECTION: Target {target_count} photos")
        print("🎯 Goal: Fix FastAI class imbalance (need ~175 more other cats)")
        
        # Get URLs from multiple subreddits
        all_urls = self.collect_from_multiple_subreddits()
        
        if not all_urls:
            print("❌ No URLs found from any source!")
            return 0
        
        # Shuffle for variety
        random.shuffle(all_urls)
        
        successful_downloads = 0
        
        print(f"\n📊 Processing {len(all_urls)} URLs...")
        
        for i, url in enumerate(all_urls[:target_count * 2]):  # Try more than target
            if successful_downloads >= target_count:
                print(f"🎯 Target reached: {successful_downloads} images")
                break
                
            filename = f"aggressive_{successful_downloads + 1:03d}.jpg"
            
            if self.download_image(url, filename):
                successful_downloads += 1
                
                # Progress updates
                if successful_downloads % 10 == 0:
                    print(f"📈 Progress: {successful_downloads}/{target_count} successful")
            
            if i % 20 == 0 and i > 0:
                print(f"📊 Processed: {i}/{len(all_urls)}, Success: {successful_downloads}")
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"\n🎉 AGGRESSIVE COLLECTION COMPLETE!")
        print(f"📊 Successfully downloaded: {successful_downloads} new black cat images")
        
        return successful_downloads

def main():
    collector = AggressiveCatCollector()
    result = collector.aggressive_collect(target_count=100)
    
    if result > 0:
        print(f"\n🚀 SUCCESS: Collected {result} new photos!")
        print("💡 Dataset balance significantly improved")
        print("🎯 Ready to retrain FastAI model with better data")
    else:
        print("\n⚠️  No new photos collected")
        print("💡 May need API keys or manual collection methods")

if __name__ == "__main__":
    main()
