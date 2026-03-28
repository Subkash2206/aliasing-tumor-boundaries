"""
Cross-Architecture Factory Module
Provides VGG16-UNet and EfficientNet-B0-UNet variants for cross-validation.
Since MONAI FlexibleUNet only natively supports ResNet backbones, we build
a lightweight encoder+decoder wrapper from torchvision feature extractors.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from src.models.blurpool_unet import replace_stride_with_blurpool


class ConvDecoder(nn.Module):
    """Simple UNet-style decoder that upsamples and merges a flat bottleneck feature."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)


class EncoderDecoderUNet(nn.Module):
    def __init__(self, encoder, encoder_out_channels, out_channels):
        super().__init__()
        self.encoder = encoder
        self.decoder = ConvDecoder(encoder_out_channels, out_channels)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


def _do_4ch_surgery(conv_layer, in_channels):
    """Replace a Conv2d with a new one that accepts `in_channels` input channels."""
    old = conv_layer
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        if in_channels > 3:
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.data = old.bias.data
    return new


def get_cross_arch_unet(arch="vgg16", in_channels=4, out_channels=4, apply_blurpool=False):
    """
    Returns a model with:
      - vgg16 encoder  (MaxPool downsampling => worst aliasing)
      - efficientnet-b0 encoder (depthwise-sep strided conv)
    with 4-channel weight surgery + optional BlurPool intervention.
    """
    if arch == "vgg16":
        backbone = tv_models.vgg16(pretrained=True)
        # Surgery on features[0] (first Conv2d)
        backbone.features[0] = _do_4ch_surgery(backbone.features[0], in_channels)
        encoder = backbone.features               # (B, 512, H/32, W/32)
        encoder_out_ch = 512

    elif arch == "efficientnet-b0":
        backbone = tv_models.efficientnet_b0(pretrained=True)
        # Surgery on features[0][0] (first Conv2d inside ConvNormActivation)
        backbone.features[0][0] = _do_4ch_surgery(backbone.features[0][0], in_channels)
        encoder = backbone.features               # (B, 1280, H/32, W/32)
        encoder_out_ch = 1280
    else:
        raise ValueError(f"Unsupported arch: {arch}. Choose 'vgg16' or 'efficientnet-b0'.")

    if apply_blurpool:
        replace_stride_with_blurpool(encoder)
        print(f"Applied BlurPool to {arch} encoder.")

    model = EncoderDecoderUNet(encoder, encoder_out_ch, out_channels)
    return model
