import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage
from matplotlib.patches import Rectangle
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from src.data.transforms_3d import get_brats_3d_val_transforms
from src.data.brats_3d_dataset import get_3d_dataloaders
from src.models.blurpool3d import replace_stride_with_blurpool3d

def create_stats_charts(results_dir, atlas_dir):
    print("Generating Figure 3: Regression Sensitivity...")
    df_base = pd.read_csv(os.path.join(results_dir, 'val_metrics.csv'))
    df_blur = pd.read_csv(os.path.join(results_dir, 'val_metrics_blurpool.csv'))
    
    df_base['Hue'] = 'Baseline'
    df_blur['Hue'] = 'BlurPool'
    df_combined = pd.concat([df_base, df_blur], ignore_index=True)
    
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df_combined, x='Alias_Violation_Ratio', y='BF1_ET', hue='Hue', alpha=0.6, s=50)
    plt.title('Spectral Aliasing Variance vs Boundary F1 (3D)')
    plt.xlabel('Alias Violation Ratio (AVR)')
    plt.ylabel('Boundary F1 — Enhancing Tumor (ET)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(atlas_dir, 'fig3_regression_sensitivity.png'), dpi=200)
    plt.close()
    
    print("Generating Figure 4: Cross Arch Performance...")
    df_cross = pd.read_csv(os.path.join(results_dir, 'final_summary_table.csv'))
    try:
        df_cross['Dice_ET'] = df_cross['Dice_ET'].str.rstrip('%').astype('float') / 100.0
        df_cross['BF1_ET'] = df_cross['BF1_ET'].str.rstrip('%').astype('float') / 100.0
    except:
        pass # Already floats
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=df_cross, x='Architecture', y='BF1_ET', hue='Intervention', ax=axes[0])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title('Boundary F1 (ET)')
    sns.barplot(data=df_cross, x='Architecture', y='Dice_ET', hue='Intervention', ax=axes[1])
    axes[1].set_title('Volume Dice (ET)')
    axes[1].set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(atlas_dir, 'fig4_cross_arch_performance.png'))
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    atlas_dir = os.path.join(results_dir, 'atlas')
    os.makedirs(atlas_dir, exist_ok=True)
    
    create_stats_charts(results_dir, atlas_dir)
    
    print("Mounting SOTA models for Heavy GPU Diagnostics...")
    val_transforms = get_brats_3d_val_transforms()
    _, val_loader, _, _ = get_3d_dataloaders(os.path.join(base_dir, 'BraTS2021_Training_Data'), val_transforms, val_transforms, 1, 0)

    base_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    base_model.load_state_dict(torch.load(os.path.join(results_dir, 'latest_segresnet_bp_False.pth'), map_location=device))
    base_model.eval()

    blur_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    replace_stride_with_blurpool3d(blur_model)
    blur_model.to(device)
    blur_model.load_state_dict(torch.load(os.path.join(results_dir, 'latest_segresnet_bp_True.pth'), map_location=device))
    blur_model.eval()
    
    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.25)

    print("Generating Figure 6: Shift Consistency...")
    base_ious, blur_ious = [], []
    shifts = [0, 1, 2, 3, 4, 5]
    computed_shift = False

    # Find a patient with sufficient tumor
    target_img, target_lbl = None, None
    for batch in val_loader:
        if "BraTS2021_00045" in batch["id"][0] or batch["seg"].sum() > 5000:
            target_img = batch["image"].to(device)
            target_lbl = batch["seg"].squeeze(1)[0].numpy()
            target_lbl[target_lbl == 3] = 4
            break
            
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Base reference (no shift)
            base_ref = torch.argmax(inferer(inputs=target_img, network=base_model), dim=1)[0].cpu()
            blur_ref = torch.argmax(inferer(inputs=target_img, network=blur_model), dim=1)[0].cpu()
            
            for s in shifts:
                # 4D images: (1, 4, D, H, W)
                shifted_img = torch.roll(target_img, shifts=s, dims=4)
                
                base_shift = torch.argmax(inferer(inputs=shifted_img, network=base_model), dim=1)[0].cpu()
                blur_shift = torch.argmax(inferer(inputs=shifted_img, network=blur_model), dim=1)[0].cpu()
                
                # Unshift predictions to compare native voxel overlap
                # dims=3 because output is (D, H, W)
                base_shift_rev = torch.roll(base_shift, shifts=-s, dims=2)
                blur_shift_rev = torch.roll(blur_shift, shifts=-s, dims=2)
                
                def calc_iou(m1, m2):
                    v1, v2 = (m1 > 0).numpy(), (m2 > 0).numpy()
                    inter = np.logical_and(v1, v2).sum()
                    union = np.logical_or(v1, v2).sum()
                    return float(inter)/union if union > 0 else 1.0
                    
                base_ious.append(calc_iou(base_ref, base_shift_rev))
                blur_ious.append(calc_iou(blur_ref, blur_shift_rev))
                print(f"   Shift {s} -> Base IoU: {base_ious[-1]:.3f} | Blur IoU: {blur_ious[-1]:.3f}")

    plt.figure(figsize=(7, 5))
    plt.plot(shifts, base_ious, label='Baseline SegResNet', marker='o', markersize=10, linewidth=3, color='blue')
    plt.plot(shifts, blur_ious, label='BlurPool SegResNet', marker='s', markersize=10, linewidth=2, alpha=0.7, color='orange')
    plt.title('Robustness Test: 3D Volumetric Shift Consistency')
    plt.xlabel('Horizontal Shift (pixels)')
    plt.ylabel('Consistency (IoU)')
    plt.legend()
    plt.ylim(max(0, min(*base_ious, *blur_ious) - 0.05), 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(atlas_dir, 'fig6_shift_consistency.png'))
    plt.close()

    print("Generating Figure 7: Error Heatmaps...")
    center_z = np.where(target_lbl > 0)[0]
    cz = int(np.mean(center_z)) if len(center_z) > 0 else target_lbl.shape[0] // 2
    
    gt_slice = target_lbl[cz, :, :]
    base_slice = base_ref.numpy()[cz, :, :]
    blur_slice = blur_ref.numpy()[cz, :, :]
    t1ce = target_img[0, 1, cz, :, :].cpu().numpy()
    
    fp_base = np.logical_and(base_slice > 0, gt_slice == 0).astype(np.uint8)
    fn_base = np.logical_and(base_slice == 0, gt_slice > 0).astype(np.uint8)
    fp_blur = np.logical_and(blur_slice > 0, gt_slice == 0).astype(np.uint8)
    fn_blur = np.logical_and(blur_slice == 0, gt_slice > 0).astype(np.uint8)
    
    # Exaggerate boundaries for visibility
    fp_base = scipy.ndimage.binary_dilation(fp_base, iterations=2).astype(np.uint8)
    fn_base = scipy.ndimage.binary_dilation(fn_base, iterations=2).astype(np.uint8)
    fp_blur = scipy.ndimage.binary_dilation(fp_blur, iterations=2).astype(np.uint8)
    fn_blur = scipy.ndimage.binary_dilation(fn_blur, iterations=2).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    for ax in axes: ax.imshow(t1ce, cmap='gray'); ax.axis('off')
    
    def draw_overlay(ax, fp, fn, title):
        canvas = np.zeros((*fp.shape, 4))
        canvas[fp > 0] = [1.0, 0.5, 0.0, 1.0] # Orange False Positives
        canvas[fn > 0] = [0.0, 1.0, 1.0, 1.0] # Cyan False Negatives
        ax.imshow(canvas)
        ax.set_title(title, color='white', pad=10)
        
    draw_overlay(axes[0], fp_base, fn_base, "Baseline Errors (Orange=FP, Cyan=FN)")
    draw_overlay(axes[1], fp_blur, fn_blur, "BlurPool Errors (Orange=FP, Cyan=FN)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(atlas_dir, 'fig7_error_heatmaps.png'), dpi=200, facecolor='black')
    plt.close()

if __name__ == "__main__":
    main()
