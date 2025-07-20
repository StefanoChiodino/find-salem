"""
Model definition and training utilities for the Find Salem project.
"""

from fastai.vision.all import *
import torch
from pathlib import Path
from typing import Optional, Tuple


class SalemClassifier:
    """Salem identification classifier using FastAI."""
    
    def __init__(self, arch=resnet34, pretrained=True):
        self.arch = arch
        self.pretrained = pretrained
        self.learner = None
        
    def create_learner(self, dls: DataLoaders, metrics=None):
        """Create a FastAI learner for the classification task."""
        if metrics is None:
            metrics = [accuracy, error_rate]
            
        self.learner = vision_learner(
            dls, 
            self.arch, 
            pretrained=self.pretrained,
            metrics=metrics
        )
        return self.learner
    
    def find_lr(self, start_lr=1e-7, end_lr=10):
        """Find optimal learning rate using FastAI's lr_find."""
        if self.learner is None:
            raise ValueError("Learner not created. Call create_learner first.")
        
        lr_finder = self.learner.lr_find(start_lr=start_lr, end_lr=end_lr)
        return lr_finder
    
    def train(self, epochs: int, lr: float = 1e-3):
        """Train the model for specified epochs."""
        if self.learner is None:
            raise ValueError("Learner not created. Call create_learner first.")
            
        # Fine-tune the pre-trained model
        self.learner.fine_tune(epochs, base_lr=lr)
        
    def evaluate(self):
        """Evaluate the model and show results."""
        if self.learner is None:
            raise ValueError("Learner not created. Call create_learner first.")
            
        # Show results
        self.learner.show_results()
        
        # Get validation metrics
        return self.learner.validate()
    
    def save_model(self, path: str = "models/salem_classifier"):
        """Save the trained model."""
        if self.learner is None:
            raise ValueError("Learner not created. Call create_learner first.")
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.learner.export(f"{path}.pkl")
        
    def load_model(self, path: str = "models/salem_classifier.pkl"):
        """Load a saved model."""
        self.learner = load_learner(path)
        return self.learner
    
    def predict(self, img_path: str):
        """Make prediction on a single image."""
        if self.learner is None:
            raise ValueError("Learner not created or loaded. Call create_learner or load_model first.")
            
        pred, pred_idx, probs = self.learner.predict(img_path)
        return pred, pred_idx, probs
    
    def interpret_results(self):
        """Show interpretation results including confusion matrix."""
        if self.learner is None:
            raise ValueError("Learner not created. Call create_learner first.")
            
        interp = ClassificationInterpretation.from_learner(self.learner)
        interp.plot_confusion_matrix()
        return interp


def train_salem_model(data_path: str, epochs: int = 5, lr: float = 1e-3):
    """Complete training pipeline for Salem classifier."""
    from .data_utils import SalemDataLoader
    
    # Load data
    data_loader = SalemDataLoader(data_path)
    dls = data_loader.create_dataloaders()
    
    # Create and train model
    classifier = SalemClassifier()
    learner = classifier.create_learner(dls)
    
    # Find optimal learning rate
    lr_finder = classifier.find_lr()
    print(f"Suggested learning rate: {lr_finder.valley}")
    
    # Train the model
    classifier.train(epochs, lr)
    
    # Evaluate
    results = classifier.evaluate()
    
    # Save model
    classifier.save_model()
    
    return classifier, results


if __name__ == "__main__":
    # Example usage
    print("Salem Classifier module loaded successfully!")
    print("Use train_salem_model('data') to start training once you have data.")
