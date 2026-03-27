import torch
import torch.nn as nn
import torchvision.models as models
from monai.networks.nets import FlexibleUNet
from src.models.blurpool import BlurPool2d, BlurMaxPool2d
from src.models.avr_hooks import AVRHook, attach_avr_hooks

def replace_stride_with_blurpool(module, target_stages=None):
    """
    Recursively replace stride-2 Conv2d/MaxPool2d layers with BlurPool equivalents.
    
    Args:
        module: The nn.Module to modify in-place.
        target_stages: Optional list of int stage indices (0-based) to apply BlurPool.
                       If None, all stride-2 layers are replaced (original behaviour).
    """
    _replace_stride_with_blurpool_impl(module, target_stages=target_stages, counter=[0])

def _replace_stride_with_blurpool_impl(module, target_stages, counter):
    if not isinstance(module, nn.Module):
        return
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and (child.stride == (2, 2) or child.stride == 2):
            stage_idx = counter[0]
            counter[0] += 1
            if target_stages is None or stage_idx in target_stages:
                stride_1_conv = nn.Conv2d(
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
                blurpool = BlurPool2d(channels=child.out_channels, stride=2)
                setattr(module, name, nn.Sequential(stride_1_conv, blurpool))
        elif isinstance(child, nn.MaxPool2d):
            stride = child.stride if child.stride is not None else child.kernel_size
            if stride == (2, 2) or stride == 2:
                stage_idx = counter[0]
                counter[0] += 1
                if target_stages is None or stage_idx in target_stages:
                    setattr(module, name, BlurMaxPool2d(
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        return_indices=child.return_indices,
                        ceil_mode=child.ceil_mode
                    ))
                else:
                    _replace_stride_with_blurpool_impl(child, target_stages, counter)
            else:
                _replace_stride_with_blurpool_impl(child, target_stages, counter)
        else:
            _replace_stride_with_blurpool_impl(child, target_stages, counter)


def get_blurpool_unet(in_channels=4, out_channels=4):
    model = FlexibleUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        backbone="resnet50",
        pretrained=False,
        is_pad=True,
        spatial_dims=2
    )
    
    resnet = models.resnet50(pretrained=True)
    
    if hasattr(model.backbone, 'conv1'):
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        
        resnet.conv1 = new_conv
        model.backbone.load_state_dict(resnet.state_dict(), strict=False)
        print(f"Weight surgery complete for in_channels={in_channels} from torchvision.")
    
    replace_stride_with_blurpool(model.backbone)
    print("Replaced stride-2 convolutions with BlurPool2d sequence.")
        
    return model
