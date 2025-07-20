# Manual Data Collection Guide

Since we're experiencing network connectivity issues, here's how to manually collect black cat images for training:

## 1. Online Sources for Black Cat Images

### Free Stock Photo Sites:
- **Unsplash**: https://unsplash.com/s/photos/black-cat
- **Pexels**: https://www.pexels.com/search/black%20cat/
- **Pixabay**: https://pixabay.com/images/search/black%20cat/
- **Freepik**: https://www.freepik.com/search?format=search&query=black%20cat

### Search Tips:
- Use terms like: "black cat", "black kitten", "dark cat", "black feline"
- Look for high-resolution images (at least 400x400 pixels)
- Choose images with good lighting and clear cat visibility
- Avoid images with multiple cats or unclear subjects

## 2. Directory Structure

Save images to these directories:

```
data/
├── train/
│   ├── salem/          <- Your Salem photos go here
│   └── other_cats/     <- Downloaded black cat photos go here
└── test/
    ├── salem/          <- Test photos of Salem
    └── other_cats/     <- Test photos of other black cats
```

## 3. Image Guidelines

### For Salem photos:
- Take 50-100 photos of Salem in different:
  - Poses (sitting, lying, standing, playing)
  - Lighting conditions (bright, dim, natural light)
  - Angles (front, side, back, close-up, full body)
  - Backgrounds (different rooms, outdoor, etc.)

### For other black cat photos:
- Collect 50-100 photos of different black cats
- Ensure variety in breeds, sizes, and poses
- Include both kittens and adult cats
- Mix of indoor and outdoor photos

## 4. Image Processing

- Save as JPG format
- Resize very large images to max 800x800 pixels
- Remove blurry or unclear images
- Name files descriptively (e.g., salem_sitting_01.jpg, other_cat_playing_05.jpg)

## 5. Quick Collection Script

Once you have network connectivity, run:
```bash
python3 src/data_collector.py
```

This will attempt to automatically download images from Unsplash and other sources.

## 6. Manual Download Steps

1. Visit the stock photo sites listed above
2. Search for "black cat" images
3. Download 10-15 images from each site
4. Save them to `data/train/other_cats/`
5. Add your Salem photos to `data/train/salem/`
6. Keep some images aside for testing in the respective test folders

## 7. Verification

After collecting images, you can verify your dataset by running:
```bash
ls -la data/train/salem/ | wc -l
ls -la data/train/other_cats/ | wc -l
```

Aim for at least 30-50 images in each category for basic training.
