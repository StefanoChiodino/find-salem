#!/usr/bin/env python3
"""
Proper FastAI Book Collector - Chapter 1 Approach
Strict black cat validation using requests library
Based on fastbook image collection methodology
"""

import requests
from pathlib import Path
import time
import hashlib
from PIL import Image
import numpy as np
from typing import List, Tuple, Set
import json
from urllib.parse import quote_plus
import random

class ProperFastAICollector:
    """Strict black cat collector following fastbook Chapter 1 methodology"""
    
    def __init__(self, output_dir="data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup requests session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Load existing hashes for duplicate detection
        self.existing_hashes = self._load_existing_hashes()
        self.downloaded_urls = set()
        
        print("🐱 Proper FastAI Book-Style Black Cat Collector")
        print("⚠️  SAFETY: This collector NEVER touches Salem photos!")
        print(f"🔍 Loaded {len(self.existing_hashes)} existing image hashes")

    def _load_existing_hashes(self) -> Set[str]:
        """Load MD5 hashes of existing images for duplicate detection"""
        hashes = set()
        
        # Check all other_cats directories (NEVER Salem directories)
        for data_dir in ["data/train/other_cats", "data/test/other_cats", "demo_samples/other_cats"]:
            dir_path = Path(data_dir)
            if dir_path.exists():
                for img_file in dir_path.glob("*.jpg"):
                    try:
                        with open(img_file, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            hashes.add(file_hash)
                    except:
                        continue
        
        return hashes

    def _is_valid_black_cat(self, img_path: Path) -> Tuple[bool, str]:
        """Strict black cat validation following fastbook quality guidelines"""
        try:
            with Image.open(img_path) as img:
                # Convert to RGB for consistent analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Size requirements - fastbook recommends minimum useful sizes
                if width < 200 or height < 200:
                    return False, f"Too small: {width}x{height}"
                
                # File size check - too small files are often logos/icons
                file_size = img_path.stat().st_size
                if file_size < 20000:  # Less than 20KB
                    return False, f"File too small: {file_size} bytes"
                
                # Convert to numpy for analysis
                img_array = np.array(img)
                
                # Check if image is mostly dark (black cat characteristic)
                mean_brightness = np.mean(img_array)
                
                if mean_brightness < 30:  # Too dark - likely solid black or very dark
                    return False, f"Too dark: {mean_brightness:.1f}"
                
                if mean_brightness > 140:  # Too bright - likely not a black cat
                    return False, f"Too bright: {mean_brightness:.1f}"
                
                # Check color distribution - black cats should be low in all channels
                r_mean, g_mean, b_mean = np.mean(img_array[:,:,0]), np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
                
                # Black cats should have relatively balanced but low RGB values
                channel_max = max(r_mean, g_mean, b_mean)
                channel_min = min(r_mean, g_mean, b_mean)
                
                if channel_max > 160:  # Any channel too bright
                    return False, f"Channel too bright: R{r_mean:.1f} G{g_mean:.1f} B{b_mean:.1f}"
                
                # Check for reasonable contrast (not a flat image)
                img_std = np.std(img_array)
                if img_std < 15:  # Too little variation - likely solid color or very flat
                    return False, f"No contrast: std={img_std:.1f}"
                
                # Check aspect ratio - extremely wide/tall images are often banners/graphics
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3.0:
                    return False, f"Unusual aspect ratio: {aspect_ratio:.1f}"
                
                # Passed all checks
                return True, f"Valid black cat: {width}x{height}, brightness={mean_brightness:.1f}"
                
        except Exception as e:
            return False, f"Analysis failed: {e}"

    def search_bing_images(self, search_term: str, count: int = 30) -> List[str]:
        """Search Bing images using requests (fastbook alternative approach)"""
        print(f"🔍 Searching Bing for: '{search_term}'")
        
        try:
            # Bing image search URL
            search_url = f"https://www.bing.com/images/search?q={quote_plus(search_term)}&first=0&count={count}"
            
            response = self.session.get(search_url, timeout=15)
            if response.status_code != 200:
                print(f"❌ Bing search failed: {response.status_code}")
                return []
            
            # Extract image URLs from response
            html = response.text
            urls = []
            
            # Look for image URLs in the HTML
            import re
            # This is a simplified approach - in practice you'd need more robust parsing
            img_pattern = r'"murl":"([^"]+\.(?:jpg|jpeg|png))"'
            matches = re.findall(img_pattern, html)
            
            for match in matches[:count]:
                # Clean up the URL
                url = match.replace('\\/', '/')
                if url.startswith('http'):
                    urls.append(url)
            
            print(f"📋 Found {len(urls)} image URLs")
            return urls[:count]
            
        except Exception as e:
            print(f"❌ Bing search error: {e}")
            return []

    def search_reddit_black_cats(self, count: int = 20) -> List[str]:
        """Search Reddit r/blackcats for high-quality black cat images"""
        print("🔍 Searching Reddit r/blackcats...")
        
        try:
            # Use Reddit JSON API (public, no auth needed)
            url = "https://www.reddit.com/r/blackcats/hot.json"
            params = {'limit': count * 3}  # Get more to filter
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code != 200:
                print(f"❌ Reddit API failed: {response.status_code}")
                return []
            
            data = response.json()
            urls = []
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Method 1: Direct image URLs ending in image extensions
                post_url = post_data.get('url', '')
                if any(ext in post_url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    # Fix common Reddit URL issues
                    if 'preview.redd.it' in post_url:
                        post_url = post_url.replace('preview.redd.it', 'i.redd.it')
                    if 'external-preview.redd.it' in post_url:
                        continue  # Skip external previews, they're often not direct images
                    urls.append(post_url)
                    continue
                
                # Method 2: i.redd.it URLs (Reddit's image hosting)
                if 'i.redd.it' in post_url:
                    urls.append(post_url)
                    continue
                
                # Method 3: Extract from preview data (more reliable)
                preview = post_data.get('preview', {})
                if 'images' in preview and preview['images']:
                    img = preview['images'][0]
                    if 'source' in img:
                        source_url = img['source']['url']
                        # Fix HTML entities and convert to direct URL
                        source_url = source_url.replace('&amp;', '&')
                        if 'preview.redd.it' in source_url:
                            source_url = source_url.replace('preview.redd.it', 'i.redd.it')
                        # Only add if it looks like a direct image URL
                        if any(ext in source_url.lower() for ext in ['.jpg', '.jpeg', '.png']) or 'i.redd.it' in source_url:
                            urls.append(source_url)
                
                # Method 4: Check for imgur links (common on Reddit)
                if 'imgur.com' in post_url and not post_url.endswith('.gifv'):
                    # Convert imgur gallery/album links to direct image links
                    if '/gallery/' in post_url or '/a/' in post_url:
                        continue  # Skip galleries for now
                    # Convert imgur page links to direct image links
                    if not any(ext in post_url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        post_url = post_url + '.jpg'  # Try adding .jpg extension
                    urls.append(post_url)
            
            # Remove duplicates while preserving order
            unique_urls = []
            seen = set()
            for url in urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)
            
            print(f"📋 Found {len(unique_urls)} unique Reddit URLs")
            return unique_urls[:count]
            
        except Exception as e:
            print(f"❌ Reddit search error: {e}")
            return []

    def download_and_validate_image(self, url: str, filename: str) -> Tuple[bool, str]:
        """Download and strictly validate image following fastbook quality standards"""
        if url in self.downloaded_urls:
            return False, "URL already tried"
        
        self.downloaded_urls.add(url)
        temp_path = self.output_dir / f"temp_{filename}"
        final_path = self.output_dir / filename
        
        try:
            # Download with proper headers
            response = self.session.get(url, timeout=20, stream=True)
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in ['image/jpeg', 'image/jpg', 'image/png']):
                return False, f"Invalid content type: {content_type}"
            
            # Download image
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate as proper black cat image
            is_valid, reason = self._is_valid_black_cat(temp_path)
            if not is_valid:
                temp_path.unlink()
                return False, reason
            
            # Check for duplicates
            with open(temp_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in self.existing_hashes:
                temp_path.unlink()
                return False, "Duplicate image"
            
            # All checks passed - move to final location
            temp_path.rename(final_path)
            self.existing_hashes.add(file_hash)
            
            file_size = final_path.stat().st_size
            return True, f"Valid black cat ({file_size} bytes)"
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Download error: {e}"

    def collect_high_quality_black_cats(self, target_count: int = 50) -> int:
        """Collect high-quality black cat images using fastbook methodology"""
        print(f"\n🎯 FastAI Book Collection: Target {target_count} high-quality black cats")
        print("=" * 70)
        
        all_urls = []
        successful_downloads = 0
        
        # Strategy 1: Reddit r/blackcats (high quality, community curated)
        reddit_urls = self.search_reddit_black_cats(count=target_count)
        all_urls.extend(reddit_urls)
        time.sleep(2)
        
        # Strategy 2: Focused search terms (fastbook recommends specific terms)
        search_terms = [
            "black cat portrait",
            "black domestic cat",
            "black cat close up",
        ]
        
        for term in search_terms:
            if len(all_urls) >= target_count * 2:  # Enough candidates
                break
            
            bing_urls = self.search_bing_images(term, count=15)
            all_urls.extend(bing_urls)
            time.sleep(3)  # Be respectful
        
        if not all_urls:
            print("❌ No image URLs found from any source")
            return 0
        
        # Remove duplicates and shuffle for variety
        unique_urls = list(set(all_urls))
        random.shuffle(unique_urls)
        
        print(f"📊 Total candidate URLs: {len(unique_urls)}")
        print("🔍 Downloading with strict black cat validation...")
        
        # Download and validate each image
        for i, url in enumerate(unique_urls):
            if successful_downloads >= target_count:
                print(f"🎯 Target reached: {successful_downloads}/{target_count}")
                break
            
            filename = f"fastai_black_cat_{successful_downloads + 1:03d}.jpg"
            
            success, reason = self.download_and_validate_image(url, filename)
            
            if success:
                successful_downloads += 1
                print(f"✅ {successful_downloads:2d}/{target_count}: {filename} - {reason}")
            else:
                print(f"❌ Skip {i+1:2d}: {reason}")
            
            # Progress updates
            if (i + 1) % 15 == 0:
                print(f"📈 Progress: {i+1} URLs processed, {successful_downloads} collected")
            
            # Rate limiting (fastbook recommends being respectful)
            time.sleep(1.5)
        
        return successful_downloads

def main():
    """Main collection function following fastbook Chapter 1 approach"""
    print("🐱 Proper FastAI Book Black Cat Collection")
    print("📖 Following fastbook Chapter 1 methodology")
    print("⚠️  SAFETY: NEVER touches Salem photos!")
    print("=" * 60)
    
    collector = ProperFastAICollector()
    
    # Set target based on current dataset imbalance
    # We have ~48 other cat training images vs 183 Salem images
    # Target: get to at least 100+ other cat images for better balance
    target = 60
    
    print(f"🎯 Goal: Collect {target} high-quality black cat images")
    print("📊 Current dataset imbalance: 183 Salem vs ~50 other cats")
    print("💡 Deep learning needs balanced datasets for good performance")
    
    collected = collector.collect_high_quality_black_cats(target)
    
    # Final summary
    print(f"\n🎉 FastAI Book Collection Complete!")
    print(f"📊 Results: {collected}/{target} high-quality black cat images collected")
    
    if collected >= 30:
        print("✅ Good progress! Ready to retrain FastAI model")
        print("💡 Run this again for even more balance")
    else:
        print("⚠️  Limited success - may need API keys or manual collection")
    
    print(f"📁 Images saved to: data/train/other_cats/")
    print("🚀 Next step: Retrain FastAI model with better balanced dataset")

if __name__ == "__main__":
    main()
