import torch
import torch.fft

class AVRHook:
    def __init__(self, name, avr_dict):
        self.name = name
        self.avr_dict = avr_dict

    def __call__(self, module, inputs):
        # inputs is a tuple for pre-hooks. The feature map F is inputs[0]
        F = inputs[0]
        # shape: (B, C, H, W)
        B, C, H, W = F.shape
        
        # Determine valid compute region
        if H < 4 or W < 4:
            return
            
        # Compute 2D Real FFT
        F_freq = torch.fft.rfft2(F, norm="forward")
        
        # Compute Power Spectrum: P = |F_freq|^2
        P = torch.abs(F_freq) ** 2
        
        # Get frequency bins
        # freqs_y is typically in [-0.5, 0.5] if d=1.0
        freqs_y = torch.fft.fftfreq(H, d=1.0).to(F.device) 
        freqs_x = torch.fft.rfftfreq(W, d=1.0).to(F.device)
        
        # Nyquist limit for stride 2 is 1/4
        nyq_limit = 0.25
        
        # Outside nyquist mask
        mask_y = torch.abs(freqs_y) > nyq_limit
        mask_x = torch.abs(freqs_x) > nyq_limit
        
        mask_y_2d = mask_y.unsqueeze(1).expand(H, freqs_x.shape[0])
        mask_x_2d = mask_x.unsqueeze(0).expand(H, freqs_x.shape[0])
        mask_outside = mask_y_2d | mask_x_2d 
        
        mask_expanded = mask_outside.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Compute AVR
        P_outside = P[mask_expanded].sum()
        P_total = P.sum()
        
        avr = (P_outside / P_total).item() if P_total > 0 else 0.0
        
        if self.name not in self.avr_dict:
            self.avr_dict[self.name] = []
        self.avr_dict[self.name].append(avr)

def attach_avr_hooks(model, avr_dict):
    """
    Attaches the AVRHook to the stride-2 bottleneck layers dynamically.
    """
    hooks = []
    
    count = 1
    for name, module in model.named_modules():
        is_stride2_conv = isinstance(module, torch.nn.Conv2d) and (module.stride == (2, 2) or module.stride == 2)
        is_blurpool = module.__class__.__name__ == 'BlurPool2d'
        
        if is_stride2_conv or is_blurpool:
            hook = module.register_forward_pre_hook(AVRHook(f"layer{count}", avr_dict))
            hooks.append(hook)
            count += 1
            
    return hooks
