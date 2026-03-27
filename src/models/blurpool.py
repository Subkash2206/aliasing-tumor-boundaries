import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=2):
        super(BlurPool2d, self).__init__()
        self.channels = channels
        self.stride = stride
        
        # 3x3 binomial filter
        a = np.array([1., 2., 1.])
        filt = a[:, None] * a[None, :]
        filt = filt / np.sum(filt)
        
        filt = torch.tensor(filt, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 3, 3)
        self.register_buffer('filt', filt)
        
    def forward(self, x):
        # Uses padding=1 to avoid coordinate shifts
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)

class BlurMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(BlurMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
        a = np.array([1., 2., 1.])
        filt = a[:, None] * a[None, :]
        self.base_filt = filt / np.sum(filt)
        self.filt = None

    def forward(self, x):
        c = x.shape[1]
        if self.filt is None or self.filt.shape[0] != c:
            filt = torch.tensor(self.base_filt, dtype=torch.float32, device=x.device)
            self.filt = filt.unsqueeze(0).expand(c, 1, 3, 3)
        blur_x = F.conv2d(x, self.filt, stride=1, padding=1, groups=c)
        return self.pool(blur_x)
