import torch
import torch.fft

class AVRHook3D:
    def __init__(self, name, avr_dict):
        self.name = name
        self.avr_dict = avr_dict

    def __call__(self, module, inputs):
        # Extract F and forcefully cast to fp32 to prevent half-precision cuFFT restrictions
        F = inputs[0].float()
        # shape: (B, C, D, H, W)
        B, C, D, H, W = F.shape
        
        # Determine valid compute region
        if D < 4 or H < 4 or W < 4:
            return
            
        # Compute 3D Real FFT over spatial dimensions
        F_freq = torch.fft.rfftn(F, dim=(-3, -2, -1), norm="forward")
        
        # Compute Power Spectrum: P = |F_freq|^2
        P = torch.abs(F_freq) ** 2
        
        # Get frequency bins
        freqs_z = torch.fft.fftfreq(D, d=1.0).to(F.device)
        freqs_y = torch.fft.fftfreq(H, d=1.0).to(F.device) 
        freqs_x = torch.fft.rfftfreq(W, d=1.0).to(F.device)
        
        # Nyquist limit for stride 2 is 1/4
        nyq_limit = 0.25
        
        # Outside nyquist mask
        mask_z = torch.abs(freqs_z) > nyq_limit
        mask_y = torch.abs(freqs_y) > nyq_limit
        mask_x = torch.abs(freqs_x) > nyq_limit
        
        # Create 3D broadcast masks
        mask_z_3d = mask_z.view(-1, 1, 1).expand(D, H, freqs_x.shape[0])
        mask_y_3d = mask_y.view(1, -1, 1).expand(D, H, freqs_x.shape[0])
        mask_x_3d = mask_x.view(1, 1, -1).expand(D, H, freqs_x.shape[0])
        
        mask_outside = mask_z_3d | mask_y_3d | mask_x_3d 
        
        # Expand over Batch and Channels
        mask_expanded = mask_outside.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
        
        # Compute AVR (Alias Violation Ratio)
        P_outside = P[mask_expanded].sum()
        P_total = P.sum()
        
        avr = (P_outside / P_total).item() if P_total > 0 else 0.0
        
        if self.name not in self.avr_dict:
            self.avr_dict[self.name] = []
        self.avr_dict[self.name].append(avr)

def attach_avr_hooks_3d(model, avr_dict):
    """
    Attaches the AVRHook3D to the stride-2 bottleneck layers dynamically.
    """
    hooks = []
    
    count = 1
    for name, module in model.named_modules():
        is_stride2_conv = isinstance(module, torch.nn.Conv3d) and (module.stride == (2, 2, 2) or module.stride == 2)
        is_blurpool = module.__class__.__name__ == 'BlurPool3d'
        
        if is_stride2_conv or is_blurpool:
            hook = module.register_forward_pre_hook(AVRHook3D(f"layer{count}", avr_dict))
            hooks.append(hook)
            count += 1
            
    return hooks
