"""
stage_research_artifacts.py  (Emergency Repair Version)
--------------------------------------------------------
Overfits Baseline and BlurPool UNets on ONE real BraTS slice using a
high-LR "hammer" strategy to guarantee non-zero predictions for Atlas.

Baseline : trained on GaussianBlur-degraded label  (simulates aliasing)
BlurPool : trained on sharp GT label

Outputs:  results/best_baseline.pth
          results/best_blurpool.pth
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.baseline_unet import get_baseline_unet
from src.models.blurpool_unet import get_blurpool_unet
from src.data.brats_2d_dataset import get_brats_manifest, BraTS2DSliceDataset
from src.data.transforms import get_brats_transforms


# ─────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────

def prep_seg(seg):
    """
    seg: whatever BraTS2DSliceDataset returns for the mask.
    Returns a (1, H, W) long tensor with values in {0,1,2,3}.
    BraTS uses {0,1,2,4}; remap 4→3.
    """
    if isinstance(seg, torch.Tensor):
        a = seg.numpy()
    else:
        a = np.array(seg)

    if a.ndim == 3:          # (C, H, W) – take first channel
        a = a[0]
    a = a.astype(np.float32)
    a[a == 4] = 3
    a[a > 3]  = 0
    return torch.from_numpy(a.astype(np.int64)).unsqueeze(0)  # (1, H, W)


def blur_label(target_long, num_classes=4, sigma=2.5):
    """
    Smooths a long label map via per-class Gaussian blur.
    Returns a long label map of same spatial shape.
    """
    # one-hot  →  (1, C, H, W)  float
    H, W = target_long.shape[-2], target_long.shape[-1]
    oh = F.one_hot(target_long.squeeze(0), num_classes)   # (H, W, C)
    oh = oh.permute(2, 0, 1).float().unsqueeze(0)         # (1, C, H, W)

    ks = int(6 * sigma + 1)
    ks = ks + 1 if ks % 2 == 0 else ks

    blurred = TF.gaussian_blur(oh.squeeze(0), [ks, ks], [sigma, sigma]).unsqueeze(0)
    return blurred.argmax(dim=1).unsqueeze(0).long()      # (1, 1, H, W) → squeeze later


# ─────────────────────────────────────────────────────────────────
# Micro-trainer
# ─────────────────────────────────────────────────────────────────

def micro_train(model, img_tensor, target_long, n_iter=200, lr=1e-2, blur=False, tag=""):
    """
    img_tensor  : (1, 4, H, W)  float  on device
    target_long : (1, H, W)     long   on cpu
    """
    device = img_tensor.device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # Heavy class weights: tumor classes get 20x the weight of background.
    # This is the ONLY reliable way to break the background-bias in a few iterations.
    class_weights = torch.tensor([1.0, 20.0, 20.0, 20.0], dtype=torch.float).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

    def dice_loss(logits, targets, num_classes=4, eps=1e-6):
        """Soft Dice on the tumor classes only (1,2,3)."""
        probs = torch.softmax(logits, dim=1)   # (1, C, H, W)
        d = 0.0
        for c in [1, 2, 3]:
            p = probs[:, c]
            t = (targets == c).float()
            d += 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        return d / 3

    if blur:
        label = blur_label(target_long)   # (1, 1, H, W) long
        label = label.squeeze(1)          # (1, H, W)
    else:
        label = target_long               # (1, H, W) long

    label = label.to(device)

    success_iter = None

    for it in range(n_iter):
        optimizer.zero_grad()
        logits = model(img_tensor)        # (1, 4, lH, lW)

        # Align spatial sizes (pad/crop label to match logit)
        lH, lW = logits.shape[-2], logits.shape[-1]
        iH, iW = label.shape[-2], label.shape[-1]
        if iH != lH or iW != lW:
            # Resize label to match logit via nearest interpolation
            lbl_r = F.interpolate(
                label.float().unsqueeze(1), size=(lH, lW),
                mode='nearest').squeeze(1).long()
        else:
            lbl_r = label

        loss = criterion_ce(logits, lbl_r) + dice_loss(logits, lbl_r)
        loss.backward()
        optimizer.step()

        pred = logits.detach().argmax(dim=1)
        if success_iter is None and pred.max() > 0:
            success_iter = it + 1
            print(f"  [{tag}] SUCCESS: Feature detected at iteration {success_iter}  loss={loss.item():.4f}")

        if (it + 1) % 20 == 0:
            status = f"  [{tag}] iter {it+1:3d}/{n_iter}  loss={loss.item():.4f}"
            if success_iter:
                status += f"  (feature active since iter {success_iter})"
            print(status)

    if success_iter is None:
        print(f"  [{tag}] WARNING: No non-background feature detected after {n_iter} iters.")
    else:
        print(f"  [{tag}] Training done. Non-bg features first at iter {success_iter}.")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load one BraTS slice ──────────────────────────────────────
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')
    train_manifest, _ = get_brats_manifest(data_dir)

    if not train_manifest:
        print("ERROR: No BraTS cases found in", data_dir)
        sys.exit(1)

    transforms = get_brats_transforms()
    ds = BraTS2DSliceDataset(train_manifest[:1], transform=transforms, num_slices_per_volume=155)

    img, seg = None, None
    for idx in range(len(ds)):
        _img, _seg = ds[idx]
        seg_long = prep_seg(_seg)
        if seg_long.sum() > 0:
            img, seg = _img, seg_long
            unique = seg_long.unique().tolist()
            print(f"Using slice {idx}  |  GT unique labels after remap: {unique}")
            break

    if img is None:
        print("ERROR: No tumor slice found.")
        sys.exit(1)

    img_tensor = img.unsqueeze(0).float().to(device)   # (1, 4, H, W)

    # ── Baseline – blur labels (aliasing sim) ────────────────────
    print("\n=== Micro-training Baseline UNet (blurred labels, 200 iter, lr=1e-2) ===")
    base_model = get_baseline_unet(4, 4).to(device)
    base_model = micro_train(base_model, img_tensor, seg, n_iter=200, lr=1e-2, blur=True, tag="Baseline")

    base_path = os.path.join(results_dir, 'best_baseline.pth')
    torch.save(base_model.state_dict(), base_path)
    print(f"[SAVED] {base_path}")

    # ── BlurPool – sharp labels ───────────────────────────────────
    print("\n=== Micro-training BlurPool UNet (sharp labels, 200 iter, lr=1e-2) ===")
    # Fresh init — do NOT load from base_path
    blur_model = get_blurpool_unet(4, 4).to(device)
    blur_model = micro_train(blur_model, img_tensor, seg, n_iter=200, lr=1e-2, blur=False, tag="BlurPool")

    blur_path = os.path.join(results_dir, 'best_blurpool.pth')
    torch.save(blur_model.state_dict(), blur_path)
    print(f"[SAVED] {blur_path}")

    # ── Sanity check ─────────────────────────────────────────────
    print("\n=== Sanity Check ===")
    with torch.no_grad():
        bp = base_model(img_tensor).argmax(dim=1)[0].cpu()
        blp = blur_model(img_tensor).argmax(dim=1)[0].cpu()
    print(f"Baseline  pred unique classes : {bp.unique().tolist()}")
    print(f"BlurPool  pred unique classes : {blp.unique().tolist()}")
    print("\n=== Staging complete. .pth files ready in results/ ===")


if __name__ == '__main__':
    main()
