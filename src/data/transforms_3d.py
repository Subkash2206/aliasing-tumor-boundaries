import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    MapTransform
)

class StackModalitiesd(MapTransform):
    """
    Custom transform to stack T1, T1ce, T2, FLAIR into a 4-channel input.
    """
    def __init__(self, keys, output_key="image"):
        super().__init__(keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        tensors = [d[k] for k in self.keys]
        stacked = torch.cat(tensors, dim=0)
        d[self.output_key] = stacked
        
        for k in self.keys:
            if k != self.output_key:
                del d[k]
        return d

class RemapBraTSLabelsd(MapTransform):
    """
    Remap ET class 4 to 3 for contiguous indices.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key][d[key] == 4] = 3
        return d

def get_brats_3d_train_transforms(patch_size=(96, 96, 96)):
    """
    Defines the MONAI Compose transform chain for the 3D BraTS training pipeline.
    """
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        StackModalitiesd(keys=["t1", "t1ce", "t2", "flair"], output_key="image"),
        RemapBraTSLabelsd(keys=["seg"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "seg"],
            label_key="seg",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=1, # Number of patches generated per volume
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
    ])

def get_brats_3d_val_transforms():
    """
    Validation transforms (No cropping, full volume, or sliding window later).
    """
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        StackModalitiesd(keys=["t1", "t1ce", "t2", "flair"], output_key="image"),
        RemapBraTSLabelsd(keys=["seg"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
