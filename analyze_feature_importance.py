#!/usr/bin/env python3
"""
Feature Importance Analysis for Salem Classifier
Visualizes which pixel regions are most important for classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from simple_trainer import SimpleSalemClassifier

def analyze_feature_importance(model_path="models/simple_salem_classifier.pkl"):
    """Analyze and visualize feature importance from trained Random Forest model."""
    
    print("🔍 Analyzing Salem Classifier Feature Importance")
    print("=" * 50)
    
    # Load the trained model
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Train the model first with: python simple_trainer.py")
        return
    
    try:
        classifier = SimpleSalemClassifier()
        classifier.load_model(model_path)
        
        # Check if it's a Random Forest (feature importance available)
        if not hasattr(classifier.model, 'feature_importances_'):
            print(f"❌ Feature importance not available for {classifier.model_type} model")
            print("💡 Feature importance is only available for Random Forest and Gradient Boosting models")
            return
            
        print(f"✅ Loaded {classifier.model_type} model")
        print(f"📊 Image size: {classifier.img_size}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get feature importances
    feature_importances = classifier.model.feature_importances_
    print(f"📈 Total features: {len(feature_importances)}")
    
    # Calculate image dimensions (RGB channels)
    img_height, img_width = classifier.img_size
    total_pixels = img_height * img_width * 3  # RGB channels
    
    if len(feature_importances) != total_pixels:
        print(f"❌ Feature count mismatch: expected {total_pixels}, got {len(feature_importances)}")
        return
    
    # Reshape feature importances back to image format (H, W, C)
    importance_image = feature_importances.reshape(img_height, img_width, 3)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Overall importance (average across RGB channels)
    plt.subplot(2, 3, 1)
    overall_importance = np.mean(importance_image, axis=2)
    plt.imshow(overall_importance, cmap='hot', interpolation='nearest')
    plt.title('Overall Feature Importance\n(Average across RGB channels)')
    plt.colorbar()
    plt.axis('off')
    
    # 2. Red channel importance
    plt.subplot(2, 3, 2)
    plt.imshow(importance_image[:, :, 0], cmap='Reds', interpolation='nearest')
    plt.title('Red Channel Importance')
    plt.colorbar()
    plt.axis('off')
    
    # 3. Green channel importance
    plt.subplot(2, 3, 3)
    plt.imshow(importance_image[:, :, 1], cmap='Greens', interpolation='nearest')
    plt.title('Green Channel Importance')
    plt.colorbar()
    plt.axis('off')
    
    # 4. Blue channel importance
    plt.subplot(2, 3, 4)
    plt.imshow(importance_image[:, :, 2], cmap='Blues', interpolation='nearest')
    plt.title('Blue Channel Importance')
    plt.colorbar()
    plt.axis('off')
    
    # 5. Top 10% most important features highlighted
    plt.subplot(2, 3, 5)
    threshold = np.percentile(overall_importance, 90)  # Top 10%
    high_importance = overall_importance > threshold
    plt.imshow(high_importance, cmap='hot')
    plt.title('Top 10% Most Important Regions')
    plt.colorbar()
    plt.axis('off')
    
    # 6. Feature importance distribution histogram
    plt.subplot(2, 3, 6)
    plt.hist(feature_importances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Feature Importance Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Number of Features')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = "feature_importance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Feature importance visualization saved to: {output_path}")
    
    # Print statistics
    print(f"\n📈 Feature Importance Statistics:")
    print(f"   Max importance: {np.max(feature_importances):.6f}")
    print(f"   Mean importance: {np.mean(feature_importances):.6f}")
    print(f"   Min importance: {np.min(feature_importances):.6f}")
    print(f"   Std deviation: {np.std(feature_importances):.6f}")
    
    # Find most important regions
    top_indices = np.argsort(feature_importances)[-10:]  # Top 10 features
    print(f"\n🎯 Top 10 Most Important Features:")
    for i, idx in enumerate(reversed(top_indices), 1):
        # Convert flat index back to (y, x, channel)
        pixel_idx = idx // 3
        channel = idx % 3
        y = pixel_idx // img_width
        x = pixel_idx % img_width
        channel_name = ['Red', 'Green', 'Blue'][channel]
        
        print(f"   {i:2d}. Pixel ({y:3d}, {x:3d}) {channel_name:5s} channel: {feature_importances[idx]:.6f}")
    
    plt.show()

def compare_sample_predictions():
    """Compare model predictions with feature importance on sample images."""
    
    print("\n🔍 Comparing Predictions with Feature Importance")
    print("=" * 50)
    
    # Load model
    classifier = SimpleSalemClassifier()
    model_path = "models/simple_salem_classifier.pkl"
    
    if not Path(model_path).exists():
        print("❌ Model not found. Run the main analysis first.")
        return
    
    classifier.load_model(model_path)
    
    # Check for sample images
    demo_salem_dir = Path("demo_samples/salem")
    demo_other_dir = Path("demo_samples/other_cats")
    
    sample_images = []
    
    if demo_salem_dir.exists():
        salem_samples = list(demo_salem_dir.glob("*.jpg"))[:2]  # First 2 Salem
        sample_images.extend([(img, "Salem") for img in salem_samples])
    
    if demo_other_dir.exists():
        other_samples = list(demo_other_dir.glob("*.jpg"))[:2]  # First 2 other cats
        sample_images.extend([(img, "Other Cat") for img in other_samples])
    
    if not sample_images:
        print("❌ No demo sample images found")
        return
    
    print(f"📸 Analyzing {len(sample_images)} sample images...")
    
    for img_path, true_label in sample_images:
        try:
            # Make prediction
            pred_label, confidence = classifier.predict(str(img_path))
            
            print(f"\n🖼️ {img_path.name}")
            print(f"   True label: {true_label}")
            print(f"   Predicted: {pred_label} (confidence: {confidence:.1%})")
            
            # Show if correct
            if pred_label == true_label:
                print("   ✅ Correct prediction")
            else:
                print("   ❌ Incorrect prediction")
                
        except Exception as e:
            print(f"❌ Error analyzing {img_path.name}: {e}")

if __name__ == "__main__":
    analyze_feature_importance()
    compare_sample_predictions()
