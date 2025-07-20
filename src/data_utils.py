"""
Data utilities for the Find Salem project.
Handles data loading, preprocessing, and augmentation.
"""

from pathlib import Path
from fastai.vision.all import *
import torch
from typing import Tuple, Optional


class SalemDataLoader:
    """Data loader for Salem identification dataset."""
    
    def __init__(self, data_path: str, batch_size: int = 32, img_size: int = 224):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.img_size = img_size
        
    def create_dataloaders(self, valid_pct: float = 0.2) -> DataLoaders:
        """Create FastAI DataLoaders for training and validation."""
        dls = ImageDataLoaders.from_folder(
            self.data_path,
            train="train",
            valid_pct=valid_pct,
            bs=self.batch_size,
            item_tfms=Resize(self.img_size),
            batch_tfms=aug_transforms(
                mult=2.0,
                do_flip=True,
                flip_vert=False,
                max_rotate=15.0,
                max_zoom=1.1,
                max_lighting=0.2,
                max_warp=0.2,
                p_affine=0.75,
                p_lighting=0.75
            )
        )
        return dls
    
    def show_batch(self, dls: DataLoaders, nrows: int = 3, ncols: int = 3):
        """Display a batch of images for inspection."""
        dls.show_batch(nrows=nrows, ncols=ncols)


def prepare_data_structure():
    """Create placeholder files to maintain directory structure in git."""
    base_path = Path("data")
    
    directories = [
        "train/salem",
        "train/other_cats", 
        "test/salem",
        "test/other_cats"
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create a README in each directory
        readme_path = full_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                if "salem" in dir_path:
                    f.write("# Salem Images\n\nPlace images of Salem (the target cat) in this directory.\n")
                else:
                    f.write("# Other Cats Images\n\nPlace images of other black cats in this directory.\n")


if __name__ == "__main__":
    prepare_data_structure()
    print("Data structure prepared!")
