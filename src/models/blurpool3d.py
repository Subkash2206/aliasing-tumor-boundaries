import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BlurPool3d(nn.Module):
    def __init__(self, channels, stride=2):
        super(BlurPool3d, self).__init__()
        self.channels = channels
        self.stride = stride
        
        # 3x3x3 binomial filter
        a = np.array([1., 2., 1.])
        filt2d = a[:, None] * a[None, :]
        filt3d = filt2d[:, :, None] * a[None, None, :]
        filt3d = filt3d / np.sum(filt3d)
        
        filt = torch.tensor(filt3d, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 3, 3, 3)
        self.register_buffer('filt', filt)
        
    def forward(self, x):
        return F.conv3d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)

def replace_stride_with_blurpool3d(module):
    """
    Recursively replace stride-2 Conv3d layers with BlurPool3d sequences.
    """
    if not isinstance(module, nn.Module):
        return
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d) and (child.stride == (2, 2, 2) or child.stride == 2):
            stride_1_conv = nn.Conv3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=1,
                padding=child.padding,
                bias=child.bias is not None
            )
            with torch.no_grad():
                stride_1_conv.weight.data = child.weight.data
                if child.bias is not None:
                    stride_1_conv.bias.data = child.bias.data
            blurpool = BlurPool3d(channels=child.out_channels, stride=2)
            setattr(module, name, nn.Sequential(stride_1_conv, blurpool))
        else:
            replace_stride_with_blurpool3d(child)
