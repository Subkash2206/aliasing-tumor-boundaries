import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.avr_hooks import AVRHook

def test_hooks():
    avr_dict = {}
    hook = AVRHook("test_layer", avr_dict)
    
    dummy = nn.Conv2d(1, 1, 1)
    
    B, C, H, W = 1, 1, 64, 64
    x = torch.linspace(0, W-1, W).view(1, 1, 1, W).expand(B, C, H, W)
    y = torch.linspace(0, H-1, H).view(1, 1, H, 1).expand(B, C, H, W)
    
    # 1. Sine wave at Nyquist limit: frequency = 1/4 cycles per pixel.
    sine_wave = torch.sin(np.pi / 2 * x) + torch.cos(np.pi / 2 * y)
    
    hook(dummy, (sine_wave,))
    
    avr_nyq = avr_dict["test_layer"][-1]
    print(f"AVR for Nyquist limit Sine Wave (<= 0.25 freq): {avr_nyq:.4f}")
    assert avr_nyq < 0.05, f"Expected AVR near 0, got {avr_nyq}"
    
    # 2. High frequency noise signal
    noise = torch.randn(B, C, H, W)
    
    hook(dummy, (noise,))
    
    avr_noise = avr_dict["test_layer"][-1]
    print(f"AVR for Noise: {avr_noise:.4f}")
    assert avr_noise > 0.5, f"Expected AVR > 0.5 for high freq noise, got {avr_noise}"
    
    print("[SUCCESS] Hook Tests Passed!")

if __name__ == "__main__":
    test_hooks()
