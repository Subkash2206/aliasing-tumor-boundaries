import os
import argparse
import torch
import numpy as np
import pandas as pd
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from src.data.transforms_3d import get_brats_3d_val_transforms
from src.data.brats_3d_dataset import get_3d_dataloaders
from src.models.blurpool3d import replace_stride_with_blurpool3d
from src.models.avr_hooks_3d import attach_avr_hooks_3d
from src.metrics.boundary_f1 import compute_boundary_f1
import scipy.ndimage as ndimage

# Ensure correct boundary logic
from src.metrics.boundary_f1 import compute_dice

def evaluate_run(model, avr_dict, data_loader, device, csv_out, roi_size=(96, 96, 96)):
    model.eval()
    rows = []
    
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.25)
    
    print(f"Beginning 3D Sliding Window Evaluation -> {csv_out}")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # images: (1, 4, D, H, W), labels: (1, 1, D, H, W)
            images = batch["image"].to(device)
            labels_orig = batch["seg"].squeeze(1).numpy()
            case_id = batch["id"][0] if "id" in batch else f"volume_{i}"
            
            # Record number of hook fires before inference
            pre_fwd = {k: len(v) for k, v in avr_dict.items()}
            
            # Autocast for memory & speed parity with training
            with torch.amp.autocast('cuda'):
                outputs = inferer(inputs=images, network=model)
                
            # Aggregate Alias Violation Ratio across all sliding windows for this volume
            avrs = [float(np.mean(v[pre_fwd.get(k, 0):])) for k, v in avr_dict.items() if len(v) > pre_fwd.get(k, 0)]
            batch_avr = sum(avrs) / len(avrs) if avrs else 0.0

            preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            # Remap class 3 -> 4 for external evaluation continuity
            preds[preds == 3] = 4
            true_b = labels_orig[0]
            true_b[true_b == 3] = 4  # CRITICAL FIX: Remap ground truth as well!

            # 3D native metrics computation
            b_f1 = compute_boundary_f1(preds, true_b, tolerance=2)
            dice = compute_dice(preds, true_b)
            
            print(f"[{case_id}] BF1-ET: {b_f1['ET']:.4f} | AVR: {batch_avr:.4f}")
            
            rows.append({
                'Case_ID': case_id,
                'Alias_Violation_Ratio': batch_avr,
                'BF1_WT': b_f1.get('WT', 0),
                'BF1_TC': b_f1.get('TC', 0),
                'BF1_ET': b_f1.get('ET', 0),
                'Dice_WT': dice.get('WT', 0),
                'Dice_TC': dice.get('TC', 0),
                'Dice_ET': dice.get('ET', 0),
            })
            
            # Fully processing validation fold.
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, index=False)
        print(f"Serialized {len(rows)} geometric metrics -> {csv_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'BraTS2021_Training_Data')

    val_transforms = get_brats_3d_val_transforms()
    # Dummy train transform since we only need validation
    _, val_loader, _, _ = get_3d_dataloaders(
        data_dir=data_dir,
        train_transforms=val_transforms,
        val_transforms=val_transforms,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print("\n--- Evaluating Baseline SegResNet ---")
    base_model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)
    base_avr = {}
    attach_avr_hooks_3d(base_model, base_avr)
    
    # Load the physically trained weights from the 100-Epoch GPU cycle!
    base_weight_path = os.path.join(base_dir, 'results', 'latest_segresnet_bp_False.pth')
    if os.path.exists(base_weight_path):
        base_model.load_state_dict(torch.load(base_weight_path, map_location=device))
        print(f"Successfully loaded baseline weights from {base_weight_path}")
    else:
        print(f"WARNING: No trained weights found at {base_weight_path}. Running randomly initialized baseline.")
        
    evaluate_run(base_model, base_avr, val_loader, device, os.path.join(base_dir, 'results', 'val_metrics_baseline_3d.csv'), roi_size=tuple(args.roi_size))
    
    print("\n--- Evaluating BlurPool SegResNet ---")
    blur_model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)
    replace_stride_with_blurpool3d(blur_model)
    blur_model = blur_model.to(device)
    blur_avr = {}
    attach_avr_hooks_3d(blur_model, blur_avr)

    blur_weight_path = os.path.join(base_dir, 'results', 'latest_segresnet_bp_True.pth')
    if os.path.exists(blur_weight_path):
        blur_model.load_state_dict(torch.load(blur_weight_path, map_location=device))
        print(f"Successfully loaded BlurPool weights from {blur_weight_path}")
        evaluate_run(blur_model, blur_avr, val_loader, device, os.path.join(base_dir, 'results', 'val_metrics_blurpool_3d.csv'), roi_size=tuple(args.roi_size))
    else:
        print(f"Skipping BlurPool Evaluation: File not found at {blur_weight_path}")

if __name__ == "__main__":
    main()
