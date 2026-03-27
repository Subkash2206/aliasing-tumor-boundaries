import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.brats_2d_dataset import get_brats_manifest, BraTS2DSliceDataset
from src.data.transforms import get_brats_transforms
from src.models.baseline_unet import get_baseline_unet
from src.models.blurpool_unet import get_blurpool_unet

def extract_boundary(mask, target_class=4):
    binary_mask = (mask == target_class).astype(np.uint8)
    eroded = scipy.ndimage.binary_erosion(binary_mask, iterations=1).astype(np.uint8)
    return binary_mask ^ eroded

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    val_metrics_path = os.path.join(results_dir, 'val_metrics.csv')
    blurpool_metrics_path = os.path.join(results_dir, 'val_metrics_blurpool.csv')
    
    df_base = pd.read_csv(val_metrics_path)
    df_blur = pd.read_csv(blurpool_metrics_path)
    
    gain = df_blur['BF1_ET'] - df_base['BF1_ET']
    # Select 3 samples with high baseline AVR & highest gain
    # Find ones with high AVR (above median)
    median_avr = df_base['Avg_AVR'].median()
    valid_indices = df_base[df_base['Avg_AVR'] >= median_avr].index
    
    # From those, get top 3 highest BF1_ET gain
    sorted_gain_indices = gain.loc[valid_indices].sort_values(ascending=False).index
    top_3_indices = sorted_gain_indices[:3].tolist()
    
    # Load dataset
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')
    _, val_manifest = get_brats_manifest(data_dir)
    transforms = get_brats_transforms()
    val_ds = BraTS2DSliceDataset(val_manifest[:5], transform=transforms, num_slices_per_volume=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_model = get_baseline_unet(4, 4).to(device)
    blurpool_model = get_blurpool_unet(4, 4).to(device)
    
    base_model_path = os.path.join(results_dir, "best_baseline.pth")
    blur_model_path = os.path.join(results_dir, "best_blurpool.pth")
    
    if os.path.exists(base_model_path):
        baseline_model.load_state_dict(torch.load(base_model_path, map_location=device))
    if os.path.exists(blur_model_path):
        blurpool_model.load_state_dict(torch.load(blur_model_path, map_location=device))
        
    baseline_model.eval()
    blurpool_model.eval()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_labels = ["Patient A", "Patient B", "Patient C"]
    col_labels = ["MRI Slice", "GT (Red)", "Baseline (Blue)", "BlurPool (Green)"]
    
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=14)
        
    for i, row in enumerate(axes):
        row[0].set_ylabel(row_labels[i], fontsize=14)
        
    with torch.no_grad():
        for i, idx in enumerate(top_3_indices):
            if idx >= len(val_ds): idx = 0
            img, label_orig = val_ds[idx]
            while label_orig.sum() == 0 and idx < len(val_ds)-1:
                idx += 1
                img, label_orig = val_ds[idx]
            
            # Predict
            img_tensor = img.unsqueeze(0).to(device)
            base_out = baseline_model(img_tensor)
            blur_out = blurpool_model(img_tensor)
            
            base_pred = torch.argmax(base_out, dim=1).squeeze(0).cpu().numpy()
            blur_pred = torch.argmax(blur_out, dim=1).squeeze(0).cpu().numpy()
            
            # test_metrics formatting matches test scripts
            # The test scripts set map label 3 back to 4 for evaluation.
            base_pred[base_pred == 3] = 4
            blur_pred[blur_pred == 3] = 4
            
            gt_mask = label_orig.squeeze(0).numpy()
            # If during training 4 was 3, maybe gt is still 4 here?
            # From train_baseline: `labels_cpu = labels_orig.squeeze(1).numpy()`
            # We will just look at target_class=4.
            
            bnd_gt = extract_boundary(gt_mask, target_class=4)
            bnd_base = extract_boundary(base_pred, target_class=4)
            bnd_blur = extract_boundary(blur_pred, target_class=4)
            
            # T1ce channel is commonly index 1 or 2 (FLAIR is 0). 
            # We'll use channel 2 for MRI slice (T1Gd/T1ce).
            mri_slice = img[2].cpu().numpy() 
            
            # Normalize MRI for display
            mri_disp = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
            
            # Helper to create colored overlay
            def overlay_boundary(mri_grayscale, boundary, color):
                # color is e.g. [1, 0, 0] for red
                rgb = np.stack([mri_grayscale]*3, axis=-1)
                for c in range(3):
                    rgb[:, :, c] = np.where(boundary > 0, color[c], rgb[:, :, c])
                return rgb
            
            mri_gt = overlay_boundary(mri_disp, bnd_gt, [1, 0, 0])      # Red
            mri_base = overlay_boundary(mri_disp, bnd_base, [0, 0, 1])  # Blue
            mri_blur = overlay_boundary(mri_disp, bnd_blur, [0, 1, 0])  # Green
            
            axes[i, 0].imshow(mri_disp, cmap='gray')
            axes[i, 1].imshow(mri_gt)
            axes[i, 2].imshow(mri_base)
            axes[i, 3].imshow(mri_blur)
            
            for ax in axes[i]:
                ax.axis('off')
                
    plt.tight_layout()
    out_path = os.path.join(results_dir, "figure2_qualitative_boundaries.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved qualitative visualization to: {out_path}")

if __name__ == "__main__":
    main()
