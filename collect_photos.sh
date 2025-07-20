#!/bin/bash
# Quick collection helper script

echo "🐱 Black Cat Photo Collection Helper"
echo "=================================="

echo "📊 Current dataset status:"
echo "Salem training photos: $(ls -1 data/train/salem/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Salem test photos: $(ls -1 data/test/salem/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Other cats training: $(ls -1 data/train/other_cats/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"
echo "Other cats test: $(ls -1 data/test/other_cats/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)"

echo ""
echo "🌐 Quick download commands (replace URLs with actual image URLs):"
echo "cd data/train/other_cats"
echo "curl -o black_cat_01.jpg 'https://example.com/black_cat_1.jpg'"
echo "curl -o black_cat_02.jpg 'https://example.com/black_cat_2.jpg'"
echo "# ... add more URLs as needed"
