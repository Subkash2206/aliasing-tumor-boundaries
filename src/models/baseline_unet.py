import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet

import torchvision.models as models

def get_baseline_unet(in_channels=4, out_channels=4):
    """
    Creates the baseline FlexibleUNet with a ResNet50 encoder.
    """
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
    else:
        print("Warning: Could not find conv1 for weight surgery.")
            
    return model
