import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from src.data.transforms_3d import get_brats_3d_val_transforms
from src.data.brats_3d_dataset import get_3d_dataloaders
from src.models.blurpool3d import replace_stride_with_blurpool3d

def map_braTS_colors(mask_array):
    """
    BraTS Labels:
    1: NCR / NET (Core) -> Red
    2: ED (Edema) -> Green
    4: ET (Enhancing Tumor) -> Yellow
    Background is transparent.
    Creates an RGB image from a categorical map.
    """
    rgb = np.zeros((*mask_array.shape, 4))
    rgb[mask_array == 1] = [1.0, 0.0, 0.0, 1.0] # Red Core
    rgb[mask_array == 2] = [0.0, 1.0, 0.0, 0.5] # Green Edema (transparent)
    rgb[mask_array == 4] = [1.0, 1.0, 0.0, 1.0] # Yellow Enhancing
    return rgb

def extract_center_slice(tensor_3d):
    """ Extract middle axial slice of the volume bounding box of the tumor. """
    z_coords = np.where(tensor_3d > 0)[0]
    if len(z_coords) == 0:
        return tensor_3d.shape[0] // 2
    return int(np.mean(z_coords))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')
    
    val_transforms = get_brats_3d_val_transforms()
    # Dummy train transform since we only need validation
    _, val_loader, _, _ = get_3d_dataloaders(
        data_dir=data_dir,
        train_transforms=val_transforms,
        val_transforms=val_transforms,
        batch_size=1,
        num_workers=0
    )
    
    print("Mounting SOTA models for Visual Render...")
    base_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    base_model.load_state_dict(torch.load(os.path.join(base_dir, 'results', 'latest_segresnet_bp_False.pth'), map_location=device))
    base_model.eval()

    blur_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    replace_stride_with_blurpool3d(blur_model)
    blur_model.to(device)
    blur_model.load_state_dict(torch.load(os.path.join(base_dir, 'results', 'latest_segresnet_bp_True.pth'), map_location=device))
    blur_model.eval()

    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.25)
    
    targets = ["BraTS2021_00045", "BraTS2021_00048"] # Specific geometrically complex subjects
    collected = []

    with torch.no_grad():
        for batch in val_loader:
            case_id = batch["id"][0]
            if case_id not in targets:
                continue

            images = batch["image"].to(device)
            labels_orig = batch["seg"].squeeze(1)[0].numpy()
            labels_orig[labels_orig == 3] = 4

            with torch.amp.autocast('cuda'):
                base_out = torch.argmax(inferer(inputs=images, network=base_model), dim=1)[0].cpu().numpy()
                base_out[base_out == 3] = 4
                
                blur_out = torch.argmax(inferer(inputs=images, network=blur_model), dim=1)[0].cpu().numpy()
                blur_out[blur_out == 3] = 4

            center_z = extract_center_slice(labels_orig)
            
            flair = images[0, 0, center_z, :, :].cpu().numpy()
            t1ce = images[0, 1, center_z, :, :].cpu().numpy() # We primarily show T1ce slice for context
            
            gt_slice = labels_orig[center_z, :, :]
            base_slice = base_out[center_z, :, :]
            blur_slice = blur_out[center_z, :, :]

            collected.append({
                'id': case_id,
                't1ce': t1ce,
                'gt': gt_slice,
                'base': base_slice,
                'blur': blur_slice
            })
            
            if len(collected) >= len(targets):
                break

    # Build qualitative figure
    fig, axes = plt.subplots(len(collected), 4, figsize=(16, 4 * len(collected)))
    for idx, c in enumerate(collected):
        # T1ce Background
        axes[idx, 0].imshow(c['t1ce'], cmap='gray')
        axes[idx, 0].set_title(f"T1ce Array ({c['id']})", color='white', pad=10)
        axes[idx, 0].axis('off')
        
        # Ground Truth Overlay
        axes[idx, 1].imshow(c['t1ce'], cmap='gray')
        axes[idx, 1].imshow(map_braTS_colors(c['gt']))
        axes[idx, 1].set_title("Clinical Ground Truth", color='white', pad=10)
        axes[idx, 1].axis('off')
        
        # Baseline Segment
        axes[idx, 2].imshow(c['t1ce'], cmap='gray')
        axes[idx, 2].imshow(map_braTS_colors(c['base']))
        axes[idx, 2].set_title("SegResNet SOTA (Baseline)", color='white', pad=10)
        axes[idx, 2].axis('off')
        
        # BlurPool Segment
        axes[idx, 3].imshow(c['t1ce'], cmap='gray')
        axes[idx, 3].imshow(map_braTS_colors(c['blur']))
        axes[idx, 3].set_title("SegResNet Anti-Aliased (BlurPool)", color='white', pad=10)
        axes[idx, 3].axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'results', 'figure2_qualitative_boundaries.png')
    plt.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Qualitative Rendering complete -> {out_path}")

if __name__ == "__main__":
    main()
