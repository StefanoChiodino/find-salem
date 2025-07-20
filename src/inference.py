"""
Inference utilities for the Find Salem project.
Handles loading trained models and making predictions on new images.
"""

from fastai.vision.all import *
from pathlib import Path
import torch
from typing import Union, Tuple, List
import matplotlib.pyplot as plt


class SalemPredictor:
    """Predictor class for identifying Salem in new images."""
    
    def __init__(self, model_path: str = "models/salem_classifier.pkl"):
        self.model_path = model_path
        self.learner = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.learner = load_learner(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")
    
    def predict_single(self, img_path: Union[str, Path]) -> Tuple[str, float]:
        """
        Predict if an image contains Salem.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.learner is None:
            raise ValueError("Model not loaded")
        
        pred, pred_idx, probs = self.learner.predict(img_path)
        confidence = float(probs.max())
        
        return str(pred), confidence
    
    def predict_batch(self, img_paths: List[Union[str, Path]]) -> List[Tuple[str, float]]:
        """
        Predict on multiple images.
        
        Args:
            img_paths: List of paths to image files
            
        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        for img_path in img_paths:
            try:
                pred, conf = self.predict_single(img_path)
                results.append((pred, conf))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append(("error", 0.0))
        
        return results
    
    def predict_with_visualization(self, img_path: Union[str, Path], figsize=(8, 6)):
        """
        Predict and visualize the result.
        
        Args:
            img_path: Path to the image file
            figsize: Figure size for matplotlib
        """
        pred, confidence = self.predict_single(img_path)
        
        # Load and display the image
        img = PILImage.create(img_path)
        
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Prediction: {pred} (Confidence: {confidence:.2%})')
        plt.tight_layout()
        plt.show()
        
        return pred, confidence
    
    def batch_predict_directory(self, directory: Union[str, Path], 
                               extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> dict:
        """
        Predict on all images in a directory.
        
        Args:
            directory: Path to directory containing images
            extensions: List of file extensions to process
            
        Returns:
            Dictionary mapping filename to (prediction, confidence)
        """
        directory = Path(directory)
        results = {}
        
        for ext in extensions:
            for img_path in directory.glob(f'*{ext}'):
                try:
                    pred, conf = self.predict_single(img_path)
                    results[img_path.name] = (pred, conf)
                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    results[img_path.name] = ("error", 0.0)
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.learner is None:
            return "No model loaded"
        
        info = {
            "classes": self.learner.dls.vocab,
            "architecture": str(self.learner.model.__class__.__name__),
            "model_path": self.model_path
        }
        
        return info


def quick_predict(img_path: str, model_path: str = "models/salem_classifier.pkl") -> Tuple[str, float]:
    """
    Quick prediction function for single images.
    
    Args:
        img_path: Path to image file
        model_path: Path to trained model
        
    Returns:
        Tuple of (prediction, confidence)
    """
    predictor = SalemPredictor(model_path)
    return predictor.predict_single(img_path)


if __name__ == "__main__":
    # Example usage
    print("Salem Predictor module loaded successfully!")
    print("Use SalemPredictor() to create a predictor instance.")
    print("Use quick_predict('path/to/image.jpg') for quick predictions.")
