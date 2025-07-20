#!/usr/bin/env python3
"""
Create lightweight demo samples for Streamlit Cloud deployment.
This script creates a small set of properly sized images that work on cloud without Git LFS.
"""

from PIL import Image
from pathlib import Path
import random

def create_demo_samples():
    """Create lightweight demo samples for cloud deployment."""
    
    # Create demo directories
    demo_dir = Path("demo_samples")
    demo_salem = demo_dir / "salem"
    demo_other = demo_dir / "other_cats"
    
    demo_salem.mkdir(parents=True, exist_ok=True)
    demo_other.mkdir(parents=True, exist_ok=True)
    
    # Get source files
    salem_files = list(Path("data/test/salem").glob("*.jpg"))
    other_files = [f for f in Path("data/test/other_cats").glob("*.jpg") if "black_cat" in f.name.lower()]
    
    # Select random samples
    if salem_files:
        salem_samples = random.sample(salem_files, min(5, len(salem_files)))
        for i, src in enumerate(salem_samples):
            try:
                img = Image.open(src).convert('RGB')
                # Keep good quality but reasonable size for cloud
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                dest = demo_salem / f"salem_{i+1}.jpg"
                img.save(dest, 'JPEG', quality=90, optimize=True)
                print(f"✓ Created {dest} ({dest.stat().st_size // 1024}KB)")
            except Exception as e:
                print(f"✗ Error with {src}: {e}")
    
    if other_files:
        other_samples = random.sample(other_files, min(5, len(other_files)))
        for i, src in enumerate(other_samples):
            try:
                img = Image.open(src).convert('RGB')
                # Keep good quality but reasonable size for cloud
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                dest = demo_other / f"other_{i+1}.jpg"
                img.save(dest, 'JPEG', quality=90, optimize=True)
                print(f"✓ Created {dest} ({dest.stat().st_size // 1024}KB)")
            except Exception as e:
                print(f"✗ Error with {src}: {e}")
    
    print(f"\n✅ Demo samples created in {demo_dir}/")

if __name__ == "__main__":
    create_demo_samples()
