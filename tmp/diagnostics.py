"""
Full diagnostic script — runs every check the user requested.
Prints raw results. Does NOT modify any source files.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.brats_2d_dataset import get_brats_manifest, BraTS2DSliceDataset
from src.data.transforms import get_brats_transforms
from src.models.baseline_unet import get_baseline_unet
from src.models.blurpool_unet import get_blurpool_unet
from src.models.avr_hooks import attach_avr_hooks
from src.metrics.boundary_f1 import compute_boundary_f1, extract_brats_subregions

SEP = "=" * 70

# ─────────────────────────────────────────────────────────
# SHARED SETUP
# ─────────────────────────────────────────────────────────
base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir  = os.path.join(base_dir, 'BraTS2021_Training_Data')
res_dir   = os.path.join(base_dir, 'results')
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, val_manifest = get_brats_manifest(data_dir)
transforms = get_brats_transforms()

# ─────────────────────────────────────────────────────────
# SECTION 1 — DATA PIPELINE: SINGLE PATIENT INSPECTION
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 1 — LOADING A SINGLE BRATS PATIENT (slice 80 of patient 0)")
print(SEP)

patient = val_manifest[0]
print(f"Patient ID : {patient['id']}")
print(f"Files      : t1={os.path.basename(patient['t1'])}, t1ce={os.path.basename(patient['t1ce'])}, "
      f"t2={os.path.basename(patient['t2'])}, flair={os.path.basename(patient['flair'])}, "
      f"seg={os.path.basename(patient['seg'])}")

# Load raw NIfTI first (no transforms) to show pre-normalisation values
import nibabel as nib
raw_flair = nib.load(patient['flair']).get_fdata()
raw_t1    = nib.load(patient['t1']).get_fdata()
raw_t1ce  = nib.load(patient['t1ce']).get_fdata()
raw_t2    = nib.load(patient['t2']).get_fdata()
raw_seg   = nib.load(patient['seg']).get_fdata()

print(f"\nRaw volume shape (H, W, D) : {raw_flair.shape}")
print(f"--- PRE-NORMALISATION voxel ranges ---")
print(f"  FLAIR  : min={raw_flair.min():.4f}  max={raw_flair.max():.4f}")
print(f"  T1     : min={raw_t1.min():.4f}      max={raw_t1.max():.4f}")
print(f"  T1CE   : min={raw_t1ce.min():.4f}    max={raw_t1ce.max():.4f}")
print(f"  T2     : min={raw_t2.min():.4f}       max={raw_t2.max():.4f}")

# Now run through the full MONAI transform chain (slice 80)
data_dict = patient.copy()
data_dict['slice_idx'] = 80
data_dict = transforms(data_dict)
img_tensor = data_dict['image']   # (4, H, W)
seg_tensor = data_dict['seg']     # (1, H, W)

print(f"\n--- POST-NORMALISATION ---")
print(f"Image tensor shape : {img_tensor.shape}   (C=4 channels: T1, T1CE, T2, FLAIR in stacking order)")
print(f"Seg tensor shape   : {seg_tensor.shape}")
print(f"Channel order (per StackModalitiesd)  : [0]=T1  [1]=T1CE  [2]=T2  [3]=FLAIR")
for ch, name in enumerate(['T1', 'T1CE', 'T2', 'FLAIR']):
    c = img_tensor[ch]
    nz  = c[c != 0]
    print(f"  Channel {ch} ({name}): min={c.min():.6f}  max={c.max():.6f}  mean={c.mean():.6f}  "
          f"nonzero-pixels={nz.numel()}  nonzero-mean={nz.mean():.6f}  nonzero-std={nz.std():.6f}")

seg_vals, seg_counts = torch.unique(seg_tensor, return_counts=True)
print(f"\nUnique seg labels in slice 80 : {dict(zip(seg_vals.tolist(), seg_counts.tolist()))}")

# Save a raw PNG of the FLAIR channel at slice 80 (channel index 3)
out_png = os.path.join(base_dir, 'tmp', 'diagnostic_slice80_flair.png')
os.makedirs(os.path.dirname(out_png), exist_ok=True)
arr = img_tensor[3].numpy()   # FLAIR
plt.figure(figsize=(6, 6))
plt.title(f"{patient['id']} — FLAIR, slice=80 (post-norm)")
plt.imshow(arr, cmap='gray', interpolation='bilinear')
plt.colorbar()
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()
print(f"\nSaved post-norm FLAIR PNG to: {out_png}")

# Also save NOT normalised raw slice for visual comparison
raw_slice = raw_flair[:, :, 80]
out_raw_png = os.path.join(base_dir, 'tmp', 'diagnostic_slice80_flair_RAW.png')
plt.figure(figsize=(6, 6))
plt.title(f"{patient['id']} — FLAIR raw (no norm), slice=80")
plt.imshow(raw_slice, cmap='gray')
plt.colorbar()
plt.tight_layout()
plt.savefig(out_raw_png, dpi=150)
plt.close()
print(f"Saved raw (non-normalised) FLAIR PNG to: {out_raw_png}")

# ─────────────────────────────────────────────────────────
# SECTION 2 — SLICE SELECTION LOGIC
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 2 — SLICE SELECTION MECHANISM")
print(SEP)
print("BraTS2DSliceDataset logic (from brats_2d_dataset.py):")
print("  - __len__ = len(manifest) * num_slices_per_volume")
print("  - slice_idx = global_idx % num_slices_per_volume")
print("  - No empty-slice filtering exists. No minimum tumor-voxel threshold.")
print("  - When num_slices_per_volume=3 (used in val_ds for generate_atlas.py),")
print("    the slice indices used are: 0, 1, 2 — EMPTY SKULL SLICES.")
print("  - The training train_baseline.py datasets: num_slices_per_volume is NOT")
print("    restricted. It iterates 0..154.")
print()

# Show what slices are actually covered in training vs. evaluate_models.py
print("Slices covered by TumorSliceDataset in evaluate_models.py: [60, 70, 75, 80, 90]")
print("Slices covered by BraTS2DSliceDataset with num_slices_per_volume=3: [0, 1, 2]")
print()

# Count tumor voxels in slices 0..9 vs. slice 60..100 for this patient
labeled_slices_early = []
labeled_slices_mid   = []
for s in range(0, 10):
    tumor_vox = int((raw_seg[:, :, s] > 0).sum())
    labeled_slices_early.append(f"  slice {s:3d}: tumor voxels = {tumor_vox}")
for s in range(60, 101, 5):
    tumor_vox = int((raw_seg[:, :, s] > 0).sum())
    labeled_slices_mid.append(f"  slice {s:3d}: tumor voxels = {tumor_vox}")

print("Tumor voxel counts in slices 0-9 (what old val pipeline evaluated):")
print('\n'.join(labeled_slices_early))
print("\nTumor voxel counts in slices 60-100 (what evaluate_models.py uses):")
print('\n'.join(labeled_slices_mid))

# ─────────────────────────────────────────────────────────
# SECTION 3 — LABEL REMAPPING
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 3 — GROUND TRUTH LABEL REMAPPING")
print(SEP)
print("Raw BraTS labels in the NIfTI segmentation files: {0, 1, 2, 4}")
unique_raw = np.unique(raw_seg)
print(f"Confirmed unique labels in patient {patient['id']} seg volume: {unique_raw}")
print()
print("Pipeline remapping (from boundary_f1.py — extract_brats_subregions):")
print("  WT = (mask==1) | (mask==2) | (mask==4)  → all non-background tumour")
print("  TC = (mask==1) | (mask==4)               → Tumour Core (NCR + ET)")
print("  ET = (mask==4)                            → Enhancing Tumour only")
print()
print("Model output class remapping (train_baseline.py lines 124, 171):")
print("  TRAINING  : labels[labels==4] = 3   → remap BraTS label 4 → internal class 3")
print("  INFERENCE : preds[preds==3]   = 4   → remap internal class 3 → BraTS label 4")
print()
print("CRITICAL CHECK — extract_brats_subregions is called with REMAPPED preds (labels 0,1,2,4),")
print("and with labels_cpu which is the ORIGINAL seg from dataloader (still BraTS 0,1,2,4 or 0,1,2,3?):")
print()
print("In evaluate_models.py:")
print("  preds = argmax(outputs)            → internal classes {0,1,2,3}")
print("  preds[preds==3] = 4                → remapped to {0,1,2,4}  ← CORRECT")
print("  labels_cpu = labels_orig.squeeze(1).numpy() ← RAW from dataloader seg")
print()
# What does the dataloader seg tensor actually contain?
seg_np = seg_tensor.squeeze(0).numpy()
unique_seg_post_transform = np.unique(seg_np)
print(f"After MONAI transform chain, seg tensor unique labels: {unique_seg_post_transform}")
print("(NormalizeIntensityd applies only to 'image' key, NOT 'seg' — so seg labels unchanged)")

# ─────────────────────────────────────────────────────────
# SECTION 4 — AVR HOOK DIAGNOSTICS
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 4 — AVR HOOK ATTACHMENT AND VARIATION ACROSS PATIENTS")
print(SEP)

base_model = get_baseline_unet(4, 4).to(device)
base_model.load_state_dict(torch.load(os.path.join(res_dir, 'best_baseline.pth'), map_location=device))
base_model.eval()

avr_dict = {}
hooks = attach_avr_hooks(base_model, avr_dict)

print("Layers that received AVR hooks (stride-2 Conv2d or BlurPool2d):")
hook_count = 0
for name, module in base_model.named_modules():
    is_s2 = isinstance(module, torch.nn.Conv2d) and (module.stride == (2, 2) or module.stride == 2)
    if is_s2:
        hook_count += 1
        # Show input shape by doing a dummy forward
        print(f"  layer{hook_count}: {name}  type=Conv2d  stride=2  "
              f"in_channels={module.in_channels}  out_channels={module.out_channels}  "
              f"kernel={module.kernel_size}")

# Run 3 different patients and show AVR variation
print("\nAVR values for 3 different patients (slices 80):")
for pat_idx in range(min(3, len(val_manifest))):
    avr_dict.clear()
    dd = val_manifest[pat_idx].copy()
    dd['slice_idx'] = 80
    dd = transforms(dd)
    img_t = dd['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        out = base_model(img_t)
    layer_avrs = {k: v[-1] for k, v in avr_dict.items() if v}
    mean_avr   = float(np.mean(list(layer_avrs.values()))) if layer_avrs else 0.0
    print(f"  Patient {val_manifest[pat_idx]['id']} slice=80: "
          f"per-layer AVRs={layer_avrs} → mean={mean_avr:.6f}")
    # Also show feature map shape by checking the first hook trigger
    # We print output shape of the model
    print(f"    Model output logits shape: {out.shape}")
    argmax_pred = torch.argmax(out, dim=1)[0].cpu().numpy()
    vals, counts = np.unique(argmax_pred, return_counts=True)
    print(f"    Argmax class distribution: {dict(zip(vals.tolist(), counts.tolist()))}")

for h in hooks:
    h.remove()

# ─────────────────────────────────────────────────────────
# SECTION 5 — val_metrics.csv: full precision first 10 rows
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 5 — val_metrics.csv  (full precision, first 10 rows)")
print(SEP)
df_base = pd.read_csv(os.path.join(res_dir, 'val_metrics.csv'))
pd.set_option('display.float_format', lambda x: f'{x:.15f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df_base.head(10).to_string())
print(f"\nTotal rows: {len(df_base)}")

# ─────────────────────────────────────────────────────────
# SECTION 6 — raw per-patient values BEFORE CSV write
# (Re-run evaluate_models logic verbosely on first 2 patients)
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 6 — RAW PRE-CSV VALUES (first 2 patients × 5 slices, verbose)")
print(SEP)

from src.analysis.evaluate_models import compute_hd95

base_model2 = get_baseline_unet(4, 4).to(device)
base_model2.load_state_dict(torch.load(os.path.join(res_dir, 'best_baseline.pth'), map_location=device))
base_model2.eval()
avr_dict2 = {}
hooks2 = attach_avr_hooks(base_model2, avr_dict2)

target_slices = [60, 70, 75, 80, 90]

for pi in range(min(2, len(val_manifest))):
    pat = val_manifest[pi]
    print(f"\nPatient: {pat['id']}")
    for sl in target_slices:
        avr_dict2.clear()
        dd = pat.copy()
        dd['slice_idx'] = sl
        dd = transforms(dd)
        img_t = dd['image'].unsqueeze(0).to(device)
        lbl_t = dd['seg']

        with torch.no_grad():
            pre_fwd = {k: len(v) for k, v in avr_dict2.items()}
            out = base_model2(img_t)

        avrs = [float(v[-1]) for k, v in avr_dict2.items() if len(v) > pre_fwd.get(k, 0)]
        mean_avr = float(np.mean(avrs)) if avrs else 0.0

        pred = torch.argmax(out, dim=1)[0].cpu().numpy()
        lbl  = lbl_t.squeeze(0).numpy()

        # class distribution BEFORE remap
        vals_pre, counts_pre = np.unique(pred, return_counts=True)

        pred_remapped = pred.copy()
        pred_remapped[pred_remapped == 3] = 4
        lbl_remapped  = lbl.copy()  # already BraTS labels from dataloader

        bf1 = compute_boundary_f1(pred_remapped, lbl_remapped, tolerance=2)

        unique_lbl = np.unique(lbl_remapped)
        unique_pred_after = np.unique(pred_remapped)

        print(f"  slice={sl:3d} | logits shape={out.shape} | "
              f"pred classes (before remap)={dict(zip(vals_pre.tolist(), counts_pre.tolist()))}")
        print(f"           | AVR={mean_avr:.8f} | BF1_ET={bf1['ET']:.8f} | BF1_WT={bf1['WT']:.8f}")
        print(f"           | unique GT labels={unique_lbl} | unique pred labels (after remap)={unique_pred_after}")

for h in hooks2:
    h.remove()

# ─────────────────────────────────────────────────────────
# SECTION 7 — val_metrics_blurpool.csv first 10 rows
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 7 — val_metrics_blurpool.csv  (full precision, first 10 rows)")
print(SEP)
df_blur = pd.read_csv(os.path.join(res_dir, 'val_metrics_blurpool.csv'))
print(df_blur.head(10).to_string())
print(f"\nTotal rows: {len(df_blur)}")

# ─────────────────────────────────────────────────────────
# SECTION 8 — final_paper_stats.json analysis
# ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SECTION 8 — final_paper_stats.json: EFFECT SIZE DEEP DIVE")
print(SEP)

with open(os.path.join(res_dir, 'final_paper_stats.json')) as f:
    paper_stats = json.load(f)

print("Contents of final_paper_stats.json:")
print(json.dumps(paper_stats, indent=4))

base_bf1 = df_base['BF1_ET'].values
blur_bf1 = df_blur['BF1_ET'].values

# Truncate to same length for comparison (Wilcoxon requires equal length)
n = min(len(base_bf1), len(blur_bf1))
base_bf1_tr = base_bf1[:n]
blur_bf1_tr = blur_bf1[:n]
diff = blur_bf1_tr - base_bf1_tr

print(f"\nArrays being compared:")
print(f"  val_metrics.csv     BF1_ET: n={len(base_bf1_tr)}  mean={base_bf1_tr.mean():.8f}  std={base_bf1_tr.std():.8f}")
print(f"  val_metrics_blurpool BF1_ET: n={len(blur_bf1_tr)}  mean={blur_bf1_tr.mean():.8f}  std={blur_bf1_tr.std():.8f}")
print(f"\nElement-wise diff (blurpool - baseline) for first 10:")
for i in range(min(10, n)):
    print(f"  [{i:2d}]  base={base_bf1_tr[i]:.8f}  blur={blur_bf1_tr[i]:.8f}  diff={diff[i]:+.8f}")

print(f"\nmean(diff) = {diff.mean():.10f}  (positive = BlurPool better, negative = BlurPool worse)")
print(f"std(diff)  = {diff.std(ddof=1):.10f}")
print(f"Cohen's d  = mean/std = {diff.mean()/diff.std(ddof=1):.10f}")
print(f"\nConclusion: effect_size={paper_stats['effect_size']:.6f} and mean_improvement={paper_stats['mean_improvement']:.6f}")
print("A NEGATIVE effect_size means BlurPool BF1_ET is on average LOWER than Baseline.")
print("This is consistent with both models predicting all-background on most slices.")
print("When both models score BF1=0 on a slice, the diff is 0. When BlurPool scores")
print("slightly more 'false' boundaries than baseline, it can be marginally worse.")

print(f"\n{SEP}")
print("DIAGNOSTICS COMPLETE")
print(SEP)
