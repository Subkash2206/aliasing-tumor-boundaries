"""
train_baseline.py — Baseline UNet training with all class-collapse fixes applied.

Fixes applied:
  #1  Class weights [0.1, 1.0, 2.0, 4.0] on both Dice and CE losses
  #3  Filter empty-prediction rows before Pearson r
  #4  TumorAwareSliceDataset for validation (top-3 tumor slices per patient)
  #7  Print full training-set class distribution before training
  #8  Combined loss: 0.5 * DiceLoss + 0.5 * CrossEntropyLoss
  #9  Collapse detection at epoch 5 and epoch 10 — early-stop if >=99% BG
  #10 Best checkpoint saved by BF1_ET, not Dice or total loss
  #11 best_baseline.pth is overwritten by this run
  #12 Training curve, post-training sanity check on BraTS2021_00016 sl.80,
      ET-only subset correlation (GT >= 100 ET voxels)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.brats_2d_dataset import (BraTS2DSliceDataset,
                                        TumorAwareSliceDataset,
                                        get_brats_manifest)
from src.data.transforms import get_brats_transforms
from src.metrics.boundary_f1 import (compute_boundary_f1, compute_dice,
                                      extract_brats_subregions)
from src.models.avr_hooks import attach_avr_hooks
from src.models.baseline_unet import get_baseline_unet
from src.utils.logger import init_wandb_logger


# ---------------------------------------------------------------------------
# Helper: Hausdorff Distance 95
# ---------------------------------------------------------------------------
def compute_hd95(pred_mask, true_mask):
    pred_bin = pred_mask > 0
    true_bin = true_mask > 0
    if not pred_bin.any() and not true_bin.any():
        return 0.0
    if not pred_bin.any() or not true_bin.any():
        return 100.0
    dist_pred_to_true = distance_transform_edt(~true_bin)
    dist_true_to_pred = distance_transform_edt(~pred_bin)
    pred_boundary = pred_bin & ~binary_erosion(pred_bin, iterations=1)
    true_boundary = true_bin & ~binary_erosion(true_bin, iterations=1)
    fwd = dist_pred_to_true[pred_boundary] if pred_boundary.any() else np.array([0.0])
    bwd = dist_true_to_pred[true_boundary] if true_boundary.any() else np.array([0.0])
    return float(np.percentile(np.concatenate([fwd, bwd]), 95))


# ---------------------------------------------------------------------------
# Fix #7: Class distribution of full training set
# ---------------------------------------------------------------------------
def print_class_distribution(train_manifest):
    import nibabel as nib
    print("\n" + "=" * 60)
    print("TRAINING SET CLASS DISTRIBUTION (all volumes)")
    print("=" * 60)
    counts = {0: 0, 1: 0, 2: 0, 4: 0}
    total  = 0
    for entry in train_manifest:
        seg = nib.load(entry['seg']).get_fdata().astype(np.int32)
        for label in [0, 1, 2, 4]:
            counts[label] += int((seg == label).sum())
        total += seg.size
    label_names = {
        0: 'Background  (label 0)',
        1: 'NCR/WT-core (label 1)',
        2: 'Edema       (label 2)',
        4: 'ET          (label 4)',
    }
    print(f"Total voxels scanned: {total:,}")
    for label in [0, 1, 2, 4]:
        pct = 100.0 * counts[label] / total
        print(f"  {label_names[label]}: {counts[label]:>12,}  ({pct:.3f}%)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Fix #8: Combined loss — 0.5 * DiceLoss + 0.5 * CrossEntropyLoss
# ---------------------------------------------------------------------------
def make_loss_fns(class_weights, device):
    dice_fn = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        weight=class_weights.to(device),
    )
    ce_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return dice_fn, ce_fn


def combined_loss(outputs, labels, dice_fn, ce_fn):
    # labels: (B, 1, H, W) long tensor arriving from DataLoader
    # DiceLoss with to_onehot_y=True expects (B, 1, H, W) — pass directly
    dice_l = dice_fn(outputs, labels)
    # CrossEntropyLoss expects (B, H, W) long — squeeze the channel dim
    ce_l   = ce_fn(outputs, labels.squeeze(1))
    return 0.5 * dice_l + 0.5 * ce_l


# ---------------------------------------------------------------------------
# Helper: argmax distribution printer (post-remap: classes 0,1,2,4)
# ---------------------------------------------------------------------------
def print_argmax_dist(pred_arr, label=""):
    total  = pred_arr.size
    header = f"  Argmax distribution{' -- '+label if label else ''}:"
    print(header)
    for cls in [0, 1, 2, 4]:
        cnt = int((pred_arr == cls).sum())
        pct = 100.0 * cnt / total
        print(f"    Class {cls}: {cnt:7d} px  ({pct:6.2f}%)")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    init_wandb_logger(project_name="spectral-aliasing-brats")

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'BraTS2021_Training_Data'))
    if not os.path.exists(data_dir):
        print("Data dir missing — aborting.")
        return

    train_manifest, val_manifest = get_brats_manifest(data_dir)
    if len(train_manifest) == 0:
        print("No training data found — aborting.")
        return

    val_manifest_use = val_manifest[:5]
    print(f"Train volumes: {len(train_manifest)}  |  Val volumes: {len(val_manifest_use)}")

    # Fix #7 — class distribution (loads NIfTI volumes; takes ~1–2 min on CPU)
    print_class_distribution(train_manifest)

    transforms = get_brats_transforms()

    # Training dataset: random slice selection (existing behaviour)
    train_ds = BraTS2DSliceDataset(train_manifest, transform=transforms,
                                   num_slices_per_volume=3)

    # Fix #4 — validation dataset: top-3 tumor slices per patient
    print("\nBuilding TumorAwareSliceDataset for validation...")
    val_ds = TumorAwareSliceDataset(val_manifest_use, transform=transforms,
                                    top_k_slices=3)
    print(f"Validation samples: {len(val_ds)} total slices\n")

    # Loaders
    steps_per_epoch    = args.steps_per_epoch
    samples_per_epoch  = steps_per_epoch * args.batch_size
    train_sampler = RandomSampler(train_ds, replacement=True,
                                  num_samples=samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = get_baseline_unet(in_channels=4, out_channels=4).to(device)
    avr_dict = {}
    attach_avr_hooks(model, avr_dict)

    # Fix #1 — class weights [BG, label1, label2, ET] with enhanced ET weight
    class_weights = torch.tensor([0.1, 1.0, 2.0, 8.0], device=device)
    dice_fn, ce_fn = make_loss_fns(class_weights, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Tracking
    best_bf1_et       = -1.0
    epoch_train_losses = []
    epoch_val_bf1_et  = []
    final_rows        = []
    stopped_early     = False

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in range(args.max_epochs):
        # Linear warmup for the first 5 epochs (epoch 0 to 4) from 1e-6 to args.lr
        target_lr = args.lr
        warmup_lr = 1e-6 + (target_lr - 1e-6) * min(1.0, epoch / 4.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}  (lr={current_lr:.2e})")
        print(f"{'='*60}")

        # ------ Training pass ------------------------------------------
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            # labels: (B, 1, H, W) — remap ET label 4 → internal class 3
            labels[labels == 4] = 3
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, labels, dice_fn, ce_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step {step+1:3d}/{steps_per_epoch}  Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / steps_per_epoch
        epoch_train_losses.append(avg_train_loss)
        print(f"Avg Train Loss: {avg_train_loss:.4f}")

        # ------ Validation pass ----------------------------------------
        model.eval()
        avr_dict.clear()

        all_dices = {'WT': [], 'TC': [], 'ET': []}
        all_bf1   = {'WT': [], 'TC': [], 'ET': []}
        val_sample_avrs = []
        val_sample_bf1s = []
        epoch_rows = []
        first_val_preds = None         # for collapse check

        with torch.no_grad():
            for batch in val_loader:
                images, labels_orig = batch
                images = images.to(device)

                pre_fwd = {k: len(v) for k, v in avr_dict.items()}
                outputs = model(images)

                avrs = [float(v[-1]) for k, v in avr_dict.items()
                        if len(v) > pre_fwd.get(k, 0)]
                batch_avr = sum(avrs) / len(avrs) if avrs else 0.0

                preds      = torch.argmax(outputs, dim=1).cpu().numpy()
                preds[preds == 3] = 4   # remap internal 3 → BraTS label 4
                labels_cpu = labels_orig.squeeze(1).numpy()

                if first_val_preds is None:
                    first_val_preds = preds[0].copy()

                for b in range(preds.shape[0]):
                    pred_b = preds[b]
                    true_b = labels_cpu[b]

                    d    = compute_dice(pred_b, true_b)
                    b_f1 = compute_boundary_f1(pred_b, true_b, tolerance=2)

                    pred_regions = extract_brats_subregions(pred_b)
                    true_regions = extract_brats_subregions(true_b)

                    hd95_wt = compute_hd95(pred_regions['WT'], true_regions['WT'])
                    hd95_tc = compute_hd95(pred_regions['TC'], true_regions['TC'])
                    hd95_et = compute_hd95(pred_regions['ET'], true_regions['ET'])

                    et_gt_voxels = int(true_regions['ET'].sum())

                    for k in ['WT', 'TC', 'ET']:
                        all_dices[k].append(d[k])
                        all_bf1[k].append(b_f1[k])

                    val_sample_avrs.append(batch_avr)
                    val_sample_bf1s.append(b_f1['WT'])

                    epoch_rows.append({
                        'Avg_AVR':      batch_avr,
                        'BF1_WT':       b_f1['WT'],
                        'BF1_TC':       b_f1['TC'],
                        'BF1_ET':       b_f1['ET'],
                        'HD95_WT':      hd95_wt,
                        'HD95_TC':      hd95_tc,
                        'HD95_ET':      hd95_et,
                        'ET_GT_voxels': et_gt_voxels,
                    })

        final_rows = epoch_rows   # keep last epoch's rows

        # ------ Epoch summary ------------------------------------------
        mean_bf1_et = (sum(all_bf1['ET']) / len(all_bf1['ET'])
                       if all_bf1['ET'] else 0.0)
        epoch_val_bf1_et.append(mean_bf1_et)

        print(f"\nValidation Epoch {epoch + 1}:")
        for k in ['WT', 'TC', 'ET']:
            md  = sum(all_dices[k]) / len(all_dices[k]) if all_dices[k] else 0.0
            mbf = sum(all_bf1[k])   / len(all_bf1[k])   if all_bf1[k]   else 0.0
            print(f"  Dice {k}: {md:.4f}  |  BF1 {k}: {mbf:.4f}")

        if epoch_rows:
            print(f"  HD95  WT:{np.mean([r['HD95_WT'] for r in epoch_rows]):.2f}  "
                  f"TC:{np.mean([r['HD95_TC'] for r in epoch_rows]):.2f}  "
                  f"ET:{np.mean([r['HD95_ET'] for r in epoch_rows]):.2f}")

        if (len(set(val_sample_avrs)) > 1 and len(set(val_sample_bf1s)) > 1):
            p_corr, _ = pearsonr(val_sample_avrs, val_sample_bf1s)
            s_corr, _ = spearmanr(val_sample_avrs, val_sample_bf1s)
            print(f"  AVR vs BF1(WT): Pearson r={p_corr:.4f}, Spearman rho={s_corr:.4f}")

        # Fix #9 — Collapse detection at epochs 5 and 10
        if (epoch + 1) in [5, 10] and first_val_preds is not None:
            total_px = first_val_preds.size
            bg_pct   = 100.0 * int((first_val_preds == 0).sum()) / total_px
            print(f"\n[COLLAPSE CHECK @ epoch {epoch+1}]")
            print_argmax_dist(first_val_preds, f"first val patient (epoch {epoch+1})")
            if bg_pct >= 99.0:
                print(f"\n!!! MODEL COLLAPSE DETECTED AT EPOCH {epoch+1} !!!")
                print(f"    {bg_pct:.1f}% of predictions are background.")
                print("    Stopping training — review class weights and loss function.")
                stopped_early = True
                break

        # Fix #10 — Save best checkpoint by BF1_ET
        if mean_bf1_et > best_bf1_et:
            best_bf1_et = mean_bf1_et
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/best_baseline.pth')
            print(f"  >>> NEW BEST  BF1_ET={best_bf1_et:.4f} --"
                  f"saved results/best_baseline.pth <<<")

    # ===================================================================
    # Post-training: save val_metrics.csv
    # ===================================================================
    os.makedirs('results', exist_ok=True)

    if final_rows:
        df = pd.DataFrame(final_rows)
        df.to_csv('results/val_metrics.csv', index=False)
        print(f"Saved {len(df)} validation rows -> results/val_metrics.csv")
        print("\nFirst 5 rows of val_metrics.csv:")
        print(df.head().to_string(index=False))
    else:
        print("\nNo validation rows collected - val_metrics.csv NOT written.")

    # Fix #12 — Training curve
    if epoch_train_losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses,
                 marker='o', markersize=3, linewidth=1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss (combined)')
        ax1.set_title('Baseline — Training Loss')
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(1, len(epoch_val_bf1_et) + 1), epoch_val_bf1_et,
                 marker='o', markersize=3, linewidth=1, color='darkorange')
        ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1,
                    label='Target BF1_ET = 0.3')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean BF1_ET (val)')
        ax2.set_title('Baseline — Val BF1_ET per Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training_curve.png', dpi=150)
        plt.close()
        print("Saved results/training_curve.png")

    if stopped_early:
        print("\n[EARLY STOPPED] Model collapsed — skip post-training sanity check.")
        return

    # ===================================================================
    # Fix #12 — Post-training sanity check: BraTS2021_00016 slice 80
    # ===================================================================
    print("\n" + "=" * 60)
    print("POST-TRAINING SANITY CHECK: BraTS2021_00016  slice 80")
    print("=" * 60)

    target_dir = os.path.join(data_dir, 'BraTS2021_00016')
    if not os.path.exists(target_dir):
        print(f"  WARNING: {target_dir} not found — skipping sanity check.")
    else:
        case_id      = 'BraTS2021_00016'
        sanity_entry = {
            'id':    case_id,
            't1':    os.path.join(target_dir, f'{case_id}_t1.nii.gz'),
            't1ce':  os.path.join(target_dir, f'{case_id}_t1ce.nii.gz'),
            't2':    os.path.join(target_dir, f'{case_id}_t2.nii.gz'),
            'flair': os.path.join(target_dir, f'{case_id}_flair.nii.gz'),
            'seg':   os.path.join(target_dir, f'{case_id}_seg.nii.gz'),
            'slice_idx': 80,
        }

        sanity_model = get_baseline_unet(in_channels=4, out_channels=4).to(device)
        sanity_model.load_state_dict(
            torch.load('results/best_baseline.pth', map_location=device))
        sanity_model.eval()

        with torch.no_grad():
            data_dict = transforms(sanity_entry)
            img_t = data_dict['image'].unsqueeze(0).to(device)
            out   = sanity_model(img_t)
            pred  = torch.argmax(out, dim=1).cpu().numpy()[0]
            pred[pred == 3] = 4

            total  = pred.size
            bg_pct = 100.0 * int((pred == 0).sum()) / total

            print(f"Argmax distribution — {case_id} slice 80 (best_baseline.pth):")
            for cls in [0, 1, 2, 4]:
                cnt = int((pred == cls).sum())
                print(f"  Class {cls}: {cnt:7d} px  ({100*cnt/total:.2f}%)")

            if bg_pct >= 99.0:
                print("\n!!! SANITY CHECK FAILED — 99%+ background predictions !!!")
                print("    Do NOT proceed to correlation. Report this issue.")
            else:
                print("\n  Sanity check PASSED — non-background classes present.")

    # ===================================================================
    # Fix #3 / #12 — Filtered correlation
    # ===================================================================
    if final_rows:
        df = pd.DataFrame(final_rows)

        # Filter 1: remove empty-prediction slices (BF1_ET==0 AND HD95_ET==100)
        df_filtered = df[~((df['BF1_ET'] == 0.0) & (df['HD95_ET'] == 100.0))]
        n_dropped   = len(df) - len(df_filtered)
        print(f"\nFiltered correlation (dropped {n_dropped} empty-pred rows, BF1_ET==0 & HD95_ET==100):")
        if len(df_filtered) >= 3 and df_filtered['Avg_AVR'].nunique() > 1:
            r, p = pearsonr(df_filtered['Avg_AVR'], df_filtered['BF1_ET'])
            rho, ps = spearmanr(df_filtered['Avg_AVR'], df_filtered['BF1_ET'])
            print(f"  n={len(df_filtered)}  "
                  f"Pearson r={r:.4f} (p={p:.3e})  "
                  f"Spearman rho={rho:.4f} (p={ps:.3e})")
        else:
            print(f"  Not enough variance after filtering (n={len(df_filtered)})")

        # Filter 2: ET voxel subset (GT >= 100 ET voxels)
        if 'ET_GT_voxels' in df.columns:
            df_et = df[df['ET_GT_voxels'] >= 100]  # noqa: ascii-safe
            print(f"\nET-only subset (GT >=100 ET voxels): n={len(df_et)}")
            if len(df_et) >= 3 and df_et['Avg_AVR'].nunique() > 1:
                r_et, p_et = pearsonr(df_et['Avg_AVR'], df_et['BF1_ET'])
                rho_et, ps_et = spearmanr(df_et['Avg_AVR'], df_et['BF1_ET'])
                print(f"  ET-only Pearson r={r_et:.4f} (p={p_et:.3e})  "
                      f"Spearman rho={rho_et:.4f} (p={ps_et:.3e})")
            else:
                print(f"  Not enough variance in ET-only subset (n={len(df_et)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--steps_per_epoch", type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=3e-5)
    args = parser.parse_args()
    train(args)
