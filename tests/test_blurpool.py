import torch
import torch.nn as nn
from src.models.blurpool import BlurPool2d

def test_blurpool_avr():
    # 64x64 synthetic checkerboard tensor
    B, C, H, W = 1, 1, 64, 64
    x = torch.zeros((B, C, H, W))
    for i in range(H):
        for j in range(W):
            if (i + j) % 2 == 0:
                x[:, :, i, j] = 1.0
                
    baseline_conv = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
    
    bp_conv = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        BlurPool2d(channels=1, stride=2)
    )
    
    with torch.no_grad():
        # Copy weights for fair comparison
        bp_conv[0].weight.data = baseline_conv.weight.data
        if baseline_conv.bias is not None:
            bp_conv[0].bias.data = baseline_conv.bias.data
            
        out_base = baseline_conv(x)
        out_bp = bp_conv(x)
        
        # Power Spectrum
        def get_avr(out):
            freq = torch.fft.rfft2(out, norm="ortho")
            power = torch.abs(freq)**2
            h, w = power.shape[-2], power.shape[-1]
            nyquist_h, nyquist_w = h // 4, w // 4
            
            mask = torch.zeros_like(power)
            mask[..., nyquist_h:, :] = 1
            mask[..., :, nyquist_w:] = 1
            
            p_out = (power * mask).sum()
            p_tot = power.sum()
            return (p_out / (p_tot + 1e-8)).item()
            
        avr_base = get_avr(out_base)
        avr_bp = get_avr(out_bp)
        
        print(f"Baseline AVR: {avr_base:.4f}")
        print(f"BlurPool AVR: {avr_bp:.4f}")
        
        # In a checkerboard, high frequencies dominate. Anti-aliasing must reduce it.
        assert avr_bp < avr_base, "BlurPool must reduce AVR compared to baseline!"
        
if __name__ == "__main__":
    test_blurpool_avr()
    print("BlurPool pathological test passed!")
