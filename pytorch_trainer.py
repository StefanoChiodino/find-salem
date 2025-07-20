#!/usr/bin/env python3
"""
PyTorch-based Salem classifier using a pre-trained ResNet model.
This script handles dataset loading, model training, and saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from PIL import Image
from pathlib import Path
import time

# --- 1. Custom Dataset for Cat Images ---

class CatDataset(Dataset):
    """A custom PyTorch Dataset to load Salem and other cat images."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'data/train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['other_cats', 'salem']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = self._find_images()

    def _find_images(self):
        """Find all image paths and assign them the correct label."""
        image_files = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.heic']

        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            if not class_dir.is_dir():
                print(f"[Warning] Directory not found, skipping: {class_dir}")
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in supported_formats:
                    image_files.append((img_path, self.class_to_idx[cls_name]))
        
        return image_files

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an image and its label by index."""
        img_path, label = self.image_paths[idx]
        
        try:
            # Open image and ensure it's in RGB format
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Return a placeholder tensor if image is corrupt
            return torch.randn(3, 224, 224), torch.tensor(0)

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
            
        return image, label


# --- 2. Model Definition ---

def get_model(num_classes=2, freeze_layers=True):
    """Load a pre-trained ResNet50 model and adapt it for our cat classification task."""
    # Load a model pre-trained on ImageNet
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Freeze all the parameters in the pre-trained model if requested
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # Get the number of input features for the classifier layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer with our custom classifier
    # The new layers' parameters will be trainable by default
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), # A new hidden layer
        nn.ReLU(),               # Activation function
        nn.Dropout(0.4),         # Dropout for regularization
        nn.Linear(256, num_classes), # The final output layer
        nn.LogSoftmax(dim=1)     # Use LogSoftmax for NLLLoss
    )
    
    return model
