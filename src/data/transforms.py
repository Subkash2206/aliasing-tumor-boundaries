import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    SpatialPadd,
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
        # EnsureChannelFirstd makes each modality (1, W, H, D). 
        # Stack them along the channel dimension (dim=0).
        tensors = [d[k] for k in self.keys]
        stacked = torch.cat(tensors, dim=0)
        d[self.output_key] = stacked
        
        # Remove original separated keys to save memory
        for k in self.keys:
            if k != self.output_key:
                del d[k]
        return d

class SelectSliceD(MapTransform):
    """
    Custom transform to extract a 2D axial slice.
    """
    def __init__(self, keys, axis=-1):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        slice_idx = d.get('slice_idx')
        if slice_idx is None:
            raise ValueError("SelectSliceD requires 'slice_idx' in the data dictionary.")
        
        for key in self.keys:
            if key in d:
                # The shape is typically (C, W, H, D). Extract along the D axis.
                # The resulting shape becomes (C, W, H).
                if self.axis == -1 or self.axis == 3:
                    d[key] = d[key][..., slice_idx]
                elif self.axis == 2:
                    d[key] = d[key][:, :, slice_idx, :]
                elif self.axis == 1:
                    d[key] = d[key][:, slice_idx, :, :]
        return d

def get_brats_transforms():
    """
    Defines the MONAI Compose transform chain for the 2D BraTS pipeline.
    """
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        StackModalitiesd(keys=["t1", "t1ce", "t2", "flair"], output_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SelectSliceD(keys=["image", "seg"], axis=-1),
        SpatialPadd(keys=["image", "seg"], spatial_size=[256, 256])
    ])
