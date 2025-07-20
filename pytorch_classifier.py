#!/usr/bin/env python3
"""
PyTorch-based Salem classifier inference module.
This module handles loading the trained PyTorch model and making predictions.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path


class PyTorchSalemClassifier:
    """PyTorch-based Salem classifier for inference."""
    
    def __init__(self, model_path="models/pytorch_salem_classifier.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['Other Cat', 'Salem']
        
        # Define the same transforms used during validation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _get_model_architecture(self):
        """Create the same model architecture used during training."""
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),  # 2 classes: Other Cat, Salem
            nn.LogSoftmax(dim=1)
        )
        
        return model
    
    def load_model(self, model_path=None):
        """Load the trained PyTorch model."""
        if model_path:
            self.model_path = model_path
            
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model architecture
        self.model = self._get_model_architecture()
        
        # Load the trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        print(f"PyTorch model loaded from {self.model_path}")
        return True
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image):
        """Make a prediction on a single image."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess the image
        input_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Convert log probabilities to probabilities
            probabilities = torch.exp(outputs)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Get probabilities for both classes
        probs = probabilities[0].cpu().numpy()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence_score,
            'probabilities': {
                'Other Cat': probs[0],
                'Salem': probs[1]
            }
        }
    
    def predict_batch(self, images):
        """Make predictions on a batch of images."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_type": "PyTorch ResNet50",
            "model_path": self.model_path,
            "device": str(self.device),
            "classes": self.classes,
            "input_size": "224x224",
            "status": "Loaded and ready"
        }


# Helper function for easy import
def load_pytorch_classifier(model_path="models/pytorch_salem_classifier.pth"):
    """Convenience function to load and return a PyTorch classifier."""
    classifier = PyTorchSalemClassifier(model_path)
    classifier.load_model()
    return classifier


if __name__ == "__main__":
    # Test the classifier if run directly
    classifier = PyTorchSalemClassifier()
    
    if Path("models/pytorch_salem_classifier.pth").exists():
        classifier.load_model()
        print("✅ PyTorch classifier loaded successfully!")
        print(classifier.get_model_info())
    else:
        print("❌ PyTorch model not found. Please train the model first using:")
        print("python pytorch_trainer.py")
