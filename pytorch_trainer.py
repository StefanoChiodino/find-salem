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


# --- 3. Training and Validation Loop ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """Train the model and track the best version."""
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
        
        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects.double() / total_train
        train_losses.append(epoch_loss)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_corrects = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                val_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
        
        val_acc = val_corrects.double() / total_val
        val_accuracies.append(val_acc)
        print(f'Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/pytorch_salem_classifier.pth')
            print(f'✨ New best model saved! Accuracy: {best_accuracy:.4f}')
    
    print(f'\nTraining completed! Best validation accuracy: {best_accuracy:.4f}')
    return model, train_losses, val_accuracies


# --- 4. Main Execution ---

if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CatDataset('data/train', transform=train_transform)
    
    # Split for validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transforms to validation subset
    val_dataset = CatDataset('data/train', transform=val_transform)
    val_subset.dataset = val_dataset
    
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2)
    
    # Model, optimizer, and criterion
    print("Initializing model...")
    model = get_model(num_classes=2)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    # Train the model
    print("Starting training...")
    model, losses, accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10
    )
    
    print("\n🎉 Training complete! Model saved to models/pytorch_salem_classifier.pth")
