import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from monai.networks.nets import SegResNet
from src.data.transforms_3d import get_brats_3d_val_transforms
from src.data.brats_3d_dataset import get_3d_dataloaders
from src.models.blurpool3d import replace_stride_with_blurpool3d
from scipy.signal.windows import hamming

def apply_fft_windowing_3d(tensor_3d):
    """ Apply Hamming window to prevent edge-effect frequency artifacts """
    D, H, W = tensor_3d.shape
    win_d = hamming(D)
    win_h = hamming(H)
    win_w = hamming(W)
    window = win_d[:, None, None] * win_h[None, :, None] * win_w[None, None, :]
    return tensor_3d * torch.tensor(window, dtype=tensor_3d.dtype, device=tensor_3d.device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    atlas_dir = os.path.join(results_dir, 'atlas')
    
    val_transforms = get_brats_3d_val_transforms()
    _, val_loader, _, _ = get_3d_dataloaders(os.path.join(base_dir, 'BraTS2021_Training_Data'), val_transforms, val_transforms, 1, 0)
    
    base_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    base_model.load_state_dict(torch.load(os.path.join(results_dir, 'latest_segresnet_bp_False.pth'), map_location=device))
    
    blur_model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2).to(device)
    replace_stride_with_blurpool3d(blur_model)
    blur_model.to(device)
    blur_model.load_state_dict(torch.load(os.path.join(results_dir, 'latest_segresnet_bp_True.pth'), map_location=device))

    # Pull a valid volume
    for batch in val_loader:
        if "BraTS2021_00045" in batch["id"][0]:
            img_tensor = batch["image"].to(device)
            # Crop to 96x96x96 to ensure perfect divisibility by 16 for SegResNet forward pass
            img_tensor = img_tensor[:, :, 40:136, 60:156, 60:156]
            break
            
    # Hook the second downsampling Block (where stride occurs)
    feat_base, feat_blur = [], []
    def hook_base(m, i, o): feat_base.append(o.detach().cpu())
    def hook_blur(m, i, o): feat_blur.append(o.detach().cpu())
    
    # In SegResNet, down_layers is a ModuleList of ResBlock downsampling
    h1 = base_model.down_layers[1].register_forward_hook(hook_base)
    h2 = blur_model.down_layers[1].register_forward_hook(hook_blur)
    
    with torch.no_grad():
        try:
            base_model.encode(img_tensor)
        except: pass
        try:
            blur_model.encode(img_tensor)
        except: pass
        
    h1.remove()
    h2.remove()
    
    # Feature Map is shape (B, C, D, H, W). We analyze Channel 0 of the downsampled block
    F_base = feat_base[0][0, 0]
    F_blur = feat_blur[0][0, 0]
    
    # FFT on central slice for visual 2D display
    cz_base, cz_blur = F_base.shape[0] // 2, F_blur.shape[0] // 2
    f_b_2d = F_base[cz_base, :, :]
    f_p_2d = F_blur[cz_blur, :, :]
    
    # 2D RFFT calculation 
    fft_base = torch.fft.fftshift(torch.fft.fft2(f_b_2d, norm="forward"))
    fft_blur = torch.fft.fftshift(torch.fft.fft2(f_p_2d, norm="forward"))
    
    P_base = torch.abs(fft_base)**2
    P_blur = torch.abs(fft_blur)**2
    
    mag_base = 10 * torch.log10(P_base + 1e-9).numpy()
    mag_blur = 10 * torch.log10(P_blur + 1e-9).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    
    vmax = max(np.percentile(mag_base, 99), np.percentile(mag_blur, 99))
    vmin = min(np.percentile(mag_base, 5), np.percentile(mag_blur, 5))
    
    axes[0].imshow(mag_base, cmap='magma', origin='lower', vmax=vmax, vmin=vmin)
    axes[0].set_title("Baseline Frequency Spectrum", color='white', pad=10)
    axes[1].imshow(mag_blur, cmap='magma', origin='lower', vmax=vmax, vmin=vmin)
    axes[1].set_title("BlurPool Frequency Spectrum", color='white', pad=10)
    
    for ax, mag in zip(axes, [mag_base, mag_blur]):
        H, W = mag.shape
        # Draw central low-pass box to show aliased zones (edges)
        rect = patches.Rectangle((W//4, H//4), W//2, H//2, linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.axis('off')
        
    plt.tight_layout()
    outpath = os.path.join(atlas_dir, 'fig1_spectral_leakage.png')
    plt.savefig(outpath, dpi=200, facecolor='black')
    plt.close()
    print(f"Generated {outpath}")

if __name__ == "__main__":
    main()
