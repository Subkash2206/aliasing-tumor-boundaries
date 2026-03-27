import matplotlib.pyplot as plt
import torch

def apply_fft_windowing(tensor):
    if len(tensor.shape) == 2:
        H, W = tensor.shape
        window_h = torch.hann_window(H, device=tensor.device)
        window_w = torch.hann_window(W, device=tensor.device)
        window_2d = window_h.unsqueeze(1) * window_w.unsqueeze(0)
        return tensor * window_2d
    elif len(tensor.shape) == 3:
        C, H, W = tensor.shape
        window_h = torch.hann_window(H, device=tensor.device)
        window_w = torch.hann_window(W, device=tensor.device)
        window_2d = window_h.unsqueeze(1) * window_w.unsqueeze(0)
        return tensor * window_2d.unsqueeze(0)
    return tensor

def get_standard_colors():
    return {
        'FP': 'orange',
        'FN': 'cyan',
        'GT': 'red',
        'Baseline': 'blue',
        'BlurPool': 'green'
    }

def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "mathtext.fontset": "cm",
        "font.family": "serif"
    })
