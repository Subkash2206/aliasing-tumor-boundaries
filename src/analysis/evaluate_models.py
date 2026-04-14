import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.brats_2d_dataset import get_brats_manifest, BraTS2DSliceDataset
from src.data.transforms import get_brats_transforms
from src.models.baseline_unet import get_baseline_unet
from src.models.blurpool_unet import get_blurpool_unet
from src.models.avr_hooks import attach_avr_hooks
from src.metrics.boundary_f1 import compute_boundary_f1, compute_dice, extract_brats_subregions

def compute_hd95(pred_mask, true_mask):
    pred_bin = pred_mask > 0
    true_bin = true_mask > 0
    if not pred_bin.any() and not true_bin.any(): return 0.0
    if not pred_bin.any() or not true_bin.any(): return 100.0
    dist_pred_to_true = distance_transform_edt(~true_bin)
    dist_true_to_pred = distance_transform_edt(~pred_bin)
    pred_boundary = pred_bin & ~binary_erosion(pred_bin, iterations=1)
    true_boundary = true_bin & ~binary_erosion(true_bin, iterations=1)
    fwd = dist_pred_to_true[pred_boundary] if pred_boundary.any() else np.array([0.0])
    bwd = dist_true_to_pred[true_boundary] if true_boundary.any() else np.array([0.0])
    return float(np.percentile(np.concatenate([fwd, bwd]), 95))

def evaluate_run(model, avr_dict, data_loader, device, csv_out):
    model.eval()
    rows = []
    print(f"Executing deep volumetric mapping -> {csv_out}")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels_orig = batch
            images = images.to(device)

            pre_fwd = {k: len(v) for k, v in avr_dict.items()}
            outputs = model(images)
            
            # Sum up average AVR drops across layers
            avrs = [float(v[-1]) for k, v in avr_dict.items() if len(v) > pre_fwd.get(k, 0)]
            batch_avr = sum(avrs) / len(avrs) if avrs else 0.0

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds[preds == 3] = 4
            labels_cpu = labels_orig.squeeze(1).numpy()
            
            for b in range(preds.shape[0]):
                p_b = preds[b]
                t_b = labels_cpu[b]

                b_f1 = compute_boundary_f1(p_b, t_b, tolerance=2)
                p_reg = extract_brats_subregions(p_b)
                t_reg = extract_brats_subregions(t_b)
                
                rows.append({
                    'Avg_AVR': batch_avr,
                    'BF1_WT': b_f1['WT'],
                    'BF1_TC': b_f1['TC'],
                    'BF1_ET': b_f1['ET'],
                    'HD95_WT': compute_hd95(p_reg['WT'], t_reg['WT']),
                    'HD95_TC': compute_hd95(p_reg['TC'], t_reg['TC']),
                    'HD95_ET': compute_hd95(p_reg['ET'], t_reg['ET']),
                })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, index=False)
        print(f"Successfully minted {len(rows)} geometric metrics.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Mounting true validation array (Slices 50-100)...")
    
    _, val_manifest = get_brats_manifest(data_dir)
    transforms = get_brats_transforms()
    
    # We will sample True Tumor Slices by hijacking data_dict processing
    # instead of passing num_slices=3. 
    # Let's bypass BraTS2DSliceDataset standard iteration and force indices.
    
    class TumorSliceDataset(torch.utils.data.Dataset):
        def __init__(self, manifest, t_form, slices=[60, 70, 75, 80, 90]):
            self.manifest = manifest
            self.transforms = t_form
            self.slices = slices
            
        def __len__(self):
            return len(self.manifest) * len(self.slices)
            
        def __getitem__(self, idx):
            v_idx = idx // len(self.slices)
            s_idx = self.slices[idx % len(self.slices)]
            data_dict = self.manifest[v_idx].copy()
            data_dict['slice_idx'] = s_idx
            if self.transforms:
                data_dict = self.transforms(data_dict)
            return data_dict['image'], data_dict['seg']
    
    # Use top 10 volumes to provide a statistically robust 50-sample variance array
    valid_ds = TumorSliceDataset(val_manifest[:10], transforms)
    val_loader = torch.utils.data.DataLoader(valid_ds, batch_size=4, shuffle=False)
    
    print("Loading Baseline Model...")
    base_model = get_baseline_unet(4, 4).to(device)
    base_model.load_state_dict(torch.load(os.path.join(base_dir, 'results', 'best_baseline.pth'), map_location=device))
    base_avr = {}
    attach_avr_hooks(base_model, base_avr)
    evaluate_run(base_model, base_avr, val_loader, device, os.path.join(base_dir, 'results', 'val_metrics.csv'))
    
    print("Loading BlurPool Model...")
    blur_model = get_blurpool_unet(4, 4).to(device)
    blur_model.load_state_dict(torch.load(os.path.join(base_dir, 'results', 'best_blurpool.pth'), map_location=device))
    blur_avr = {}
    attach_avr_hooks(blur_model, blur_avr)
    evaluate_run(blur_model, blur_avr, val_loader, device, os.path.join(base_dir, 'results', 'val_metrics_blurpool.csv'))
    
    print("Complete. Arrays serialized natively!")

if __name__ == "__main__":
    main()
