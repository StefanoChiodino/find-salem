#!/usr/bin/env python3
"""
FastAI Salem Classifier for Web Interface
Production-ready inference class based on fastbook Chapter 1 approach.
"""

from fastai.vision.all import *
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, List

class FastAISalemClassifier:
    """
    FastAI-based Salem classifier for production inference.
    Uses pre-trained CNN model with transfer learning.
    """
    
    def __init__(self, model_path: str = "models/salem_fastai_model.pkl"):
        """Initialize the FastAI classifier"""
        self.model_path = model_path
        self.learn = None
        self.classes = None
        
    def load_model(self, model_path: str = None) -> bool:
        """Load the trained FastAI model"""
        if model_path:
            self.model_path = model_path
            
        if not Path(self.model_path).exists():
            print(f"❌ FastAI model file not found: {self.model_path}")
            return False
        
        try:
            # Load the exported model (fastbook production approach)
            self.learn = load_learner(self.model_path)
            self.classes = self.learn.dls.vocab
            
            print(f"✅ FastAI model loaded from {self.model_path}")
            print(f"📋 Classes: {self.classes}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading FastAI model: {e}")
            return False
    
    def predict_single(self, image_path: str) -> Tuple[str, float, dict]:
        """
        Make prediction on a single image (fastbook inference approach)
        
        Returns:
            - predicted_class: str
            - confidence: float (0-1)
            - all_probabilities: dict mapping class names to probabilities
        """
        if self.learn is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        try:
            # Make prediction using fastbook inference pattern
            pred_class, pred_idx, probs = self.learn.predict(image_path)
            
            # Extract confidence and probabilities
            confidence = probs.max().item()
            
            # Create probability dictionary for all classes
            all_probs = {}
            for i, class_name in enumerate(self.classes):
                all_probs[class_name] = probs[i].item()
            
            return str(pred_class), confidence, all_probs
            
        except Exception as e:
            print(f"❌ Error making prediction on {image_path}: {e}")
            return "Error", 0.0, {}
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float, dict]]:
        """
        Make predictions on multiple images efficiently
        
        Returns:
            List of (predicted_class, confidence, all_probabilities) tuples
        """
        if self.learn is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        results = []
        
        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)
        
        return results
    
    def predict_with_interpretation(self, image_path: str) -> dict:
        """
        Make prediction with detailed interpretation (fastbook debugging approach)
        
        Returns comprehensive analysis including confidence, probabilities, and interpretation
        """
        if self.learn is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        try:
            # Make prediction
            pred_class, confidence, all_probs = self.predict_single(image_path)
            
            # Create interpretation
            interpretation = {
                'image_path': image_path,
                'predicted_class': pred_class,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'model_type': 'FastAI CNN (ResNet34)',
                'is_salem': pred_class.lower() == 'salem',
                'certainty_level': self._get_certainty_level(confidence)
            }
            
            return interpretation
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'predicted_class': 'Error',
                'confidence': 0.0
            }
    
    def _get_certainty_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable certainty level"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High" 
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.learn is None:
            return {"error": "Model not loaded"}
        
        return {
            'model_type': 'FastAI CNN Classifier',
            'architecture': 'Pre-trained ResNet34',
            'classes': list(self.classes),
            'num_classes': len(self.classes),
            'input_size': '224x224',
            'training_approach': 'Transfer Learning + Fine-tuning',
            'data_augmentation': 'Yes (FastAI aug_transforms)',
            'model_path': self.model_path
        }
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Visualize prediction with the original image (fastbook visualization)
        """
        if self.learn is None:
            raise ValueError("Model not loaded! Call load_model() first.")
        
        try:
            # Load and display image with prediction
            img = PILImage.create(image_path)
            pred_class, pred_idx, probs = self.learn.predict(image_path)
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            img.show(ax=ax)
            
            # Add prediction text
            confidence = probs.max().item()
            title = f"Predicted: {pred_class} ({confidence:.1%} confidence)"
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"📊 Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")

def test_fastai_classifier():
    """Test the FastAI classifier on demo samples"""
    print("🧪 Testing FastAI Salem Classifier")
    print("=" * 40)
    
    # Initialize classifier
    classifier = FastAISalemClassifier()
    
    # Try to load model
    if not classifier.load_model():
        print("❌ Could not load FastAI model. Train it first with: python fastai_trainer.py")
        return
    
    # Get model info
    info = classifier.get_model_info()
    print(f"\n📋 Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test on demo samples
    demo_salem_dir = Path("demo_samples/salem")
    demo_other_dir = Path("demo_samples/other_cats")
    
    test_images = []
    if demo_salem_dir.exists():
        test_images.extend([(f, "Salem") for f in demo_salem_dir.glob("*.jpg")[:2]])
    if demo_other_dir.exists():
        test_images.extend([(f, "Other Cat") for f in demo_other_dir.glob("*.jpg")[:2]])
    
    if not test_images:
        print("❌ No demo images found for testing")
        return
    
    print(f"\n🖼️ Testing on {len(test_images)} images:")
    
    correct_predictions = 0
    total_predictions = len(test_images)
    
    for img_path, true_label in test_images:
        # Get detailed interpretation
        result = classifier.predict_with_interpretation(str(img_path))
        
        if 'error' not in result:
            pred_class = result['predicted_class']
            confidence = result['confidence']
            certainty = result['certainty_level']
            
            print(f"\n📸 {img_path.name}")
            print(f"   True: {true_label}")
            print(f"   Predicted: {pred_class} ({confidence:.1%}, {certainty} certainty)")
            
            if pred_class == true_label:
                print("   ✅ Correct")
                correct_predictions += 1
            else:
                print("   ❌ Incorrect")
        else:
            print(f"\n❌ Error with {img_path.name}: {result['error']}")
    
    # Summary
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    print(f"   Model: {info['architecture']}")

if __name__ == "__main__":
    test_fastai_classifier()
