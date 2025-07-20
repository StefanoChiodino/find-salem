#!/usr/bin/env python3
"""
Simple Salem classifier using scikit-learn.
Works without FastAI for immediate training capability.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pickle
from pathlib import Path
import time
from typing import Tuple, List


class SimpleSalemClassifier:
    """Simple image classifier using scikit-learn."""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.model = None
        self.model_type = None
        
    def load_and_preprocess_images(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images from directory structure."""
        print(f"📂 Loading images from {data_dir}...")
        
        images = []
        labels = []
        
        # Load Salem images (label = 1)
        salem_dir = Path(data_dir) / "salem"
        if salem_dir.exists():
            salem_files = [f for f in salem_dir.glob("*") 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']]
            
            print(f"📸 Loading {len(salem_files)} Salem images...")
            for img_path in salem_files:
                try:
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Salem = 1
                except Exception as e:
                    print(f"⚠️ Error loading {img_path.name}: {e}")
        
        # Load other cat images (label = 0)
        other_dir = Path(data_dir) / "other_cats"
        if other_dir.exists():
            other_files = [f for f in other_dir.glob("*.jpg")]
            
            print(f"🐱 Loading {len(other_files)} other cat images...")
            for img_path in other_files:
                try:
                    img = self.preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Other cats = 0
                except Exception as e:
                    print(f"⚠️ Error loading {img_path.name}: {e}")
        
        if not images:
            raise ValueError("No images loaded! Check your data directory structure.")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"✅ Loaded {len(X)} images total")
        print(f"   Salem images: {np.sum(y == 1)}")
        print(f"   Other cat images: {np.sum(y == 0)}")
        
        return X, y
    
    def preprocess_image(self, img_path: Path) -> np.ndarray:
        """Preprocess a single image."""
        try:
            # Load and convert image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Flatten for traditional ML algorithms
            img_flat = img_array.flatten()
            
            return img_flat
            
        except Exception as e:
            print(f"⚠️ Error preprocessing {img_path}: {e}")
            return None
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
        """Train the classification model."""
        print(f"🚀 Training {model_type} model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} images")
        print(f"📊 Validation set: {len(X_val)} images")
        
        # Choose model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"📈 Validation Accuracy: {accuracy:.3f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['Other Cat', 'Salem']))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Other  Salem")
        print(f"Actual Other    {cm[0,0]:3d}    {cm[0,1]:3d}")
        print(f"       Salem    {cm[1,0]:3d}    {cm[1,1]:3d}")
        
        return accuracy
    
    def predict(self, img_path: str) -> Tuple[str, float]:
        """Make prediction on a single image."""
        if self.model is None:
            raise ValueError("Model not trained! Call train_model first.")
        
        # Preprocess image
        img = self.preprocess_image(Path(img_path))
        if img is None:
            return "Error", 0.0
        
        # Make prediction
        img_reshaped = img.reshape(1, -1)
        prediction = self.model.predict(img_reshaped)[0]
        probabilities = self.model.predict_proba(img_reshaped)[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        # Convert prediction to label
        label = "Salem" if prediction == 1 else "Other Cat"
        
        return label, confidence
    
    def save_model(self, filepath: str = "models/simple_salem_classifier.pkl"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        # Create models directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'img_size': self.img_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/simple_salem_classifier.pkl"):
        """Load a saved model."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.img_size = model_data['img_size']
        
        print(f"📥 Model loaded from {filepath}")


def train_salem_classifier():
    """Main training function."""
    print("🐱 Simple Salem Classifier Training")
    print("=" * 40)
    
    # Initialize classifier
    classifier = SimpleSalemClassifier(img_size=(64, 64))
    
    # Load training data
    try:
        X_train, y_train = classifier.load_and_preprocess_images("data/train")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return
    
    # Train model
    try:
        accuracy = classifier.train_model(X_train, y_train, model_type="random_forest")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Save model
    try:
        classifier.save_model()
        print(f"✅ Training complete! Model saved with {accuracy:.1%} accuracy")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    # Test on a few images if test set exists
    test_dir = Path("data/test")
    if test_dir.exists():
        print(f"\n🧪 Testing on test set...")
        try:
            X_test, y_test = classifier.load_and_preprocess_images("data/test")
            y_pred = classifier.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"🎯 Test Accuracy: {test_accuracy:.1%}")
        except Exception as e:
            print(f"⚠️ Error testing: {e}")
    
    print(f"\n🎉 Salem classifier is ready!")
    print(f"💡 You can now use the web interface to make predictions")


if __name__ == "__main__":
    train_salem_classifier()
