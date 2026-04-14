import os
from monai.data import Dataset, DataLoader
from src.data.brats_2d_dataset import get_brats_manifest

class BraTS3DDataset(Dataset):
    """
    A PyTorch/MONAI Dataset that takes a list of 3D volume paths and 
    applies the provided 3D transforms (e.g. RandCropByPosNegLabeld).
    """
    def __init__(self, manifest, transform=None):
        self.manifest = manifest
        # MONAI Dataset expects a list of dictionaries and a transform
        super().__init__(data=self.manifest, transform=transform)

def get_3d_dataloaders(data_dir, train_transforms, val_transforms, batch_size=1, num_workers=0):
    """
    Utility function to build BraTS dataset and dataloaders for 3D patched training.
    """
    train_manifest, val_manifest = get_brats_manifest(data_dir)
    
    if len(train_manifest) == 0:
        raise ValueError(f"No training data found in {data_dir}")

    # Use a small subset for dev testing if necessary, but here we assume full set
    train_ds = BraTS3DDataset(train_manifest, transform=train_transforms)
    val_ds = BraTS3DDataset(val_manifest, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_manifest, val_manifest
