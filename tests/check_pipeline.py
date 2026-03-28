import os
import sys
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.brats_2d_dataset import get_brats_manifest, BraTS2DSliceDataset
from src.data.transforms import get_brats_transforms

def run_check():
    # Looking for a data directory:
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(workspace_dir, 'BraTS2021_Training_Data')
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please ensure BraTS2021 dataset is located here.")
        return

    train_manifest, val_manifest = get_brats_manifest(data_dir)
    print(f"Found {len(train_manifest)} training cases and {len(val_manifest)} validation cases.")
    
    if len(train_manifest) == 0:
        print("No valid cases found in data_dir. Verify file naming conventions.")
        return

    transforms = get_brats_transforms()
    # Create a dataset using the first case
    test_manifest = [train_manifest[0]]
    dataset = BraTS2DSliceDataset(test_manifest, transform=transforms, num_slices_per_volume=155)
    
    # Typically, center slices contain the tumor. Around slice index 75 for BraTS.
    sample_idx = 75
    img, seg = dataset[sample_idx]
    
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {seg.shape}")
    
    assert img.shape == (4, 240, 240), f"Expected input shape (4, 240, 240), got {img.shape}"
    assert seg.shape == (1, 240, 240), f"Expected mask shape (1, 240, 240), got {seg.shape}"
    
    # Save a PNG to verify alignment
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    for i in range(4):
        # The tensor format is (C, W, H), so we transpose back to (H, W) for plotting usually,
        # but standard plotting handles it. Using numpy().T might be necessary if visually rotated.
        # We will plot directly to verify spatial alignment.
        im_slice = img[i].cpu().numpy().T
        axes[i].imshow(im_slice, cmap='gray')
        axes[i].set_title(modality_names[i])
        axes[i].axis('off')
        
    mask_slice = seg[0].cpu().numpy().T
    axes[4].imshow(mask_slice, cmap='nipy_spectral')
    axes[4].set_title('Segmentation Mask')
    axes[4].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'pipeline_check.png')
    plt.savefig(out_path)
    print(f"Successfully saved test output image to {out_path}")
    print("Pipeline check passed.")

if __name__ == "__main__":
    run_check()
