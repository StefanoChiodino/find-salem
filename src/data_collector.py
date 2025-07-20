"""Data collection utilities for gathering black cat images from the internet."""

import requests
from pathlib import Path
import time
from typing import List, Optional
import json
from PIL import Image
import io
import random
import re
from urllib.parse import urlparse, parse_qs
import base64


class BlackCatImageCollector:
    """Collector for black cat images from various sources."""
    
    def __init__(self, output_dir: str = "data/train/other_cats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download a single image from URL."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Verify it's an image and has reasonable size
            if len(response.content) < 1000:  # Too small
                print(f"⚠️ Image too small: {filename}")
                return False
                
            image = Image.open(io.BytesIO(response.content))
            
            # Check image dimensions
            if image.width < 200 or image.height < 200:
                print(f"⚠️ Image dimensions too small: {filename} ({image.width}x{image.height})")
                return False
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (keep aspect ratio)
            max_size = 800
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save image
            filepath = self.output_dir / filename
            image.save(filepath, 'JPEG', quality=85, optimize=True)
            
            print(f"✅ Downloaded: {filename} ({image.width}x{image.height})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {str(e)}")
            return False
    
    def collect_from_unsplash(self, count: int = 30) -> int:
        """Collect images from Unsplash (no API key required for basic usage)."""
        print(f"🔍 Searching Unsplash for {count} black cat images...")
        
        search_terms = [
            "black cat",
            "black kitten", 
            "black feline",
            "dark cat",
            "black domestic cat"
        ]
        
        successful = 0
        
        for term in search_terms:
            if successful >= count:
                break
                
            try:
                # Unsplash search URL (using their public API endpoint)
                url = f"https://unsplash.com/napi/search/photos?query={term.replace(' ', '%20')}&per_page=20"
                
                response = self.session.get(url)
                if response.status_code != 200:
                    print(f"⚠️ Failed to search Unsplash for '{term}'")
                    continue
                    
                data = response.json()
                
                if 'results' not in data:
                    continue
                    
                for i, photo in enumerate(data['results']):
                    if successful >= count:
                        break
                        
                    try:
                        # Get the regular size image URL
                        img_url = photo['urls']['regular']
                        filename = f"unsplash_{term.replace(' ', '_')}_{i+1:02d}_{photo['id'][:8]}.jpg"
                        
                        if self.download_image(img_url, filename):
                            successful += 1
                            
                        # Be respectful with rate limiting
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing Unsplash photo: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ Error searching Unsplash for '{term}': {e}")
                continue
                
            # Delay between search terms
            time.sleep(3)
        
        return successful
    
    def collect_from_duckduckgo(self, count: int = 50) -> int:
        """Collect images from DuckDuckGo image search."""
        print(f"🔍 Searching DuckDuckGo for {count} black cat images...")
        
        search_terms = [
            "black cat",
            "black kitten", 
            "dark cat",
            "black domestic cat",
            "black feline"
        ]
        
        successful = 0
        
        for term in search_terms:
            if successful >= count:
                break
                
            try:
                # DuckDuckGo image search
                search_url = f"https://duckduckgo.com/?q={term.replace(' ', '+')}&t=h_&iax=images&ia=images"
                
                # Get the search page
                response = self.session.get(search_url)
                if response.status_code != 200:
                    print(f"⚠️ Failed to search DuckDuckGo for '{term}'")
                    continue
                
                # Get the vqd token needed for image requests
                vqd_match = re.search(r'vqd=([^&]+)', response.text)
                if not vqd_match:
                    print(f"⚠️ Could not find vqd token for '{term}'")
                    continue
                    
                vqd = vqd_match.group(1)
                
                # Request actual image data
                images_url = "https://duckduckgo.com/i.js"
                params = {
                    'l': 'us-en',
                    'o': 'json',
                    'q': term,
                    'vqd': vqd,
                    'f': ',,,',
                    'p': '1'
                }
                
                img_response = self.session.get(images_url, params=params)
                if img_response.status_code != 200:
                    print(f"⚠️ Failed to get images for '{term}'")
                    continue
                    
                try:
                    img_data = img_response.json()
                    results = img_data.get('results', [])
                except:
                    print(f"⚠️ Could not parse image data for '{term}'")
                    continue
                
                for i, result in enumerate(results[:20]):  # Limit per search term
                    if successful >= count:
                        break
                        
                    try:
                        img_url = result.get('image')
                        if not img_url:
                            continue
                            
                        filename = f"ddg_{term.replace(' ', '_')}_{i+1:02d}_{successful+1:03d}.jpg"
                        
                        if self.download_image(img_url, filename):
                            successful += 1
                            
                        # Be respectful with rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing DuckDuckGo image: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ Error searching DuckDuckGo for '{term}': {e}")
                continue
                
            # Delay between search terms
            time.sleep(2)
        
        return successful
    
    def collect_from_bing(self, count: int = 30) -> int:
        """Collect images from Bing image search."""
        print(f"🔍 Searching Bing for {count} black cat images...")
        
        search_terms = [
            "black cat",
            "black kitten",
            "black house cat",
            "dark cat"
        ]
        
        successful = 0
        
        for term in search_terms:
            if successful >= count:
                break
                
            try:
                # Bing image search URL
                search_url = "https://www.bing.com/images/search"
                params = {
                    'q': term,
                    'form': 'HDRSC2',
                    'first': '1',
                    'count': '35'
                }
                
                response = self.session.get(search_url, params=params)
                if response.status_code != 200:
                    print(f"⚠️ Failed to search Bing for '{term}'")
                    continue
                
                # Extract image URLs from the page
                # Look for image URLs in the page content
                img_pattern = r'"murl":"([^"]+)"'
                img_matches = re.findall(img_pattern, response.text)
                
                if not img_matches:
                    print(f"⚠️ No images found on Bing for '{term}'")
                    continue
                
                for i, img_url in enumerate(img_matches[:15]):  # Limit per search term
                    if successful >= count:
                        break
                        
                    try:
                        # Clean up the URL
                        img_url = img_url.replace('\\u0026', '&')
                        
                        filename = f"bing_{term.replace(' ', '_')}_{i+1:02d}_{successful+1:03d}.jpg"
                        
                        if self.download_image(img_url, filename):
                            successful += 1
                            
                        # Be respectful with rate limiting
                        time.sleep(1.5)
                        
                    except Exception as e:
                        print(f"⚠️ Error processing Bing image: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ Error searching Bing for '{term}': {e}")
                continue
                
            # Delay between search terms
            time.sleep(3)
        
        return successful
    
    def collect_from_pexels(self, count: int = 20) -> int:
        """Collect images from Pexels (requires API key but has free tier)."""
        print(f"🔍 Attempting to collect from Pexels...")
        
        # Note: This would require a Pexels API key
        # For now, we'll skip this and focus on other sources
        print("⚠️ Pexels collection requires API key - skipping for now")
        return 0
    
    def collect_from_direct_urls(self) -> int:
        """Collect from a curated list of direct image URLs."""
        print("🔍 Collecting from direct URLs...")
        
        # Curated list of black cat image URLs (these are examples - replace with real URLs)
        direct_urls = [
            # Add actual image URLs here
            # "https://example.com/black_cat_1.jpg",
        ]
        
        if not direct_urls:
            print("⚠️ No direct URLs configured")
            return 0
            
        return self.collect_from_urls(direct_urls, "direct")
    
    def collect_from_lorem_picsum(self, count: int = 10) -> int:
        """Collect random images from Lorem Picsum (for testing purposes)."""
        print(f"🔍 Collecting {count} test images from Lorem Picsum...")
        
        successful = 0
        
        for i in range(count):
            try:
                # Lorem Picsum provides random images
                # We'll use different seeds to get variety
                seed = random.randint(1, 1000)
                url = f"https://picsum.photos/seed/{seed}/400/400"
                filename = f"test_cat_{i+1:03d}.jpg"
                
                if self.download_image(url, filename):
                    successful += 1
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Error downloading test image {i+1}: {e}")
                
        return successful


    def collect_from_urls(self, urls: List[str], prefix: str = "black_cat") -> int:
        """Download images from a list of URLs."""
        successful = 0
        
        for i, url in enumerate(urls):
            filename = f"{prefix}_{i+1:03d}.jpg"
            
            if self.download_image(url, filename):
                successful += 1
            
            # Be respectful with delays
            time.sleep(1)
        
        return successful


def collect_sample_images(count: int = 50, output_dir: str = "data/train/other_cats"):
    """
    Collect black cat images from various sources.
    
    Args:
        count: Number of images to collect
        output_dir: Directory to save images
    """
    collector = BlackCatImageCollector(output_dir)
    
    print(f"🐱 Starting collection of {count} black cat images...")
    print(f"📁 Output directory: {output_dir}")
    
    total_collected = 0
    
    # Try DuckDuckGo first (free and reliable)
    print("\n=== Collecting from DuckDuckGo ===")
    ddg_count = collector.collect_from_duckduckgo(min(count, 100))
    total_collected += ddg_count
    
    # If we need more images, try Bing
    remaining = count - total_collected
    
    if remaining > 0:
        print(f"\n=== Collecting {remaining} more from Bing ===")
        bing_count = collector.collect_from_bing(remaining)
        total_collected += bing_count
        remaining = count - total_collected
    
    # Try Unsplash if still need more
    if remaining > 0:
        print(f"\n=== Collecting {remaining} more from Unsplash ===")
        unsplash_count = collector.collect_from_unsplash(remaining)
        total_collected += unsplash_count
        remaining = count - total_collected
    
    # Try direct URLs if still need more
    if remaining > 0:
        print("\n=== Collecting from direct URLs ===")
        direct_count = collector.collect_from_direct_urls()
        total_collected += direct_count
        remaining = count - total_collected
    
    print(f"\n🎉 Collection complete!")
    print(f"✅ Successfully collected {total_collected} images")
    print(f"📁 Saved to: {Path(output_dir).absolute()}")
    
    return total_collected


def create_data_readme_files():
    """Create README files in data directories with instructions."""
    directories = {
        "data/train/salem": "Place images of Salem (your specific black cat) here for training.",
        "data/train/other_cats": "Place images of other black cats here for training.",
        "data/test/salem": "Place test images of Salem here for evaluation.",
        "data/test/other_cats": "Place test images of other black cats here for evaluation."
    }
    
    for dir_path, description in directories.items():
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        readme_path = dir_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# {dir_path.name.replace('_', ' ').title()} Images\n\n")
            f.write(f"{description}\n\n")
            f.write("## Guidelines:\n")
            f.write("- Use high-quality images (preferably > 224x224 pixels)\n")
            f.write("- Ensure good lighting and clear visibility of the cat\n")
            f.write("- Include various poses and angles\n")
            f.write("- Remove any images that are blurry or unclear\n")
            f.write("- Aim for at least 50-100 images per category for good results\n")


if __name__ == "__main__":
    print("🐱 Black Cat Image Collector")
    print("=" * 40)
    
    # Create README files and data structure
    create_data_readme_files()
    print("✅ Created data directory README files")
    
    # Collect black cat images
    print("\n🚀 Starting image collection...")
    
    try:
        collected = collect_sample_images(count=30)
        
        if collected > 0:
            print(f"\n🎉 Success! Collected {collected} black cat images")
            print("📝 Next steps:")
            print("1. Add your Salem photos to data/train/salem/")
            print("2. Review and clean the collected images")
            print("3. Run the training script when ready")
        else:
            print("\n⚠️ No images were collected")
            print("💡 You can manually add images to the data folders")
            
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("💡 You can manually add images to the data folders")
    
    print(f"\n📁 Data structure at: {Path('data').absolute()}")
