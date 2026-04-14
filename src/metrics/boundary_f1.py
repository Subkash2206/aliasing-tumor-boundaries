import numpy as np
import scipy.ndimage as ndimage

def get_boundary(mask, tolerance=2):
    """
    Extracts the boundary of a binary mask using the XOR method:
    boundary = mask ^ eroded_mask
    """
    mask_bin = (mask > 0)
    eroded = ndimage.binary_erosion(mask_bin, iterations=tolerance)
    boundary = mask_bin ^ eroded
    return boundary

def calculate_precision_recall_f1(pred_boundary, true_boundary):
    """
    Compute Precision, Recall, and F1 strictly on boundary pixels.
    """
    intersection = np.logical_and(pred_boundary, true_boundary).sum()
    pred_sum = pred_boundary.sum()
    true_sum = true_boundary.sum()

    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall    = intersection / true_sum if true_sum > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def extract_brats_subregions(mask):
    """
    Extracts the Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
    binary masks from the multi-class BraTS mask.
    Labels mapping:
        WT: Labels 1 + 2 + 4
        TC: Labels 1 + 4
        ET: Label 4
    """
    wt = (mask == 1) | (mask == 2) | (mask == 4)
    tc = (mask == 1) | (mask == 4)
    et = (mask == 4)
    return {'WT': wt, 'TC': tc, 'ET': et}

def compute_boundary_f1(y_pred, y_true, tolerance=2):
    """
    Computes Boundary F1 for each BraTS subregion (WT, TC, ET) independently.

    Edge-case policy (Fix #2):
      - true_bnd == 0 AND pred_bnd == 0 → 0.0 (empty slice, not a perfect match)
      - true_bnd == 0, pred_bnd > 0     → 0.0 (false positive boundaries)
      - true_bnd > 0, pred_bnd == 0     → 0.0 (missed boundary)
      - Both non-empty                  → standard F1
    """
    pred_regions = extract_brats_subregions(y_pred)
    true_regions = extract_brats_subregions(y_true)

    results = {}
    for region_name in ['WT', 'TC', 'ET']:
        pred_bnd = get_boundary(pred_regions[region_name], tolerance=tolerance)
        true_bnd = get_boundary(true_regions[region_name], tolerance=tolerance)

        if true_bnd.sum() == 0 and pred_bnd.sum() == 0:
            # Both empty — score 0.0, not 1.0. Empty slices should not inflate BF1.
            results[region_name] = 0.0
        elif true_bnd.sum() == 0:
            # No GT boundary — any prediction is wrong
            results[region_name] = 0.0
        else:
            _, _, f1 = calculate_precision_recall_f1(pred_bnd, true_bnd)
            results[region_name] = f1

    return results

def compute_dice(y_pred, y_true):
    """
    Standard Dice Coefficient per BraTS subregion.

    Edge-case policy:
      - Both regions empty → 0.0 (avoid inflating Dice on empty slices)
      - One empty, one not → 0.0 (total miss)
    """
    pred_regions = extract_brats_subregions(y_pred)
    true_regions = extract_brats_subregions(y_true)

    results = {}
    for region_name in ['WT', 'TC', 'ET']:
        pred_mask = pred_regions[region_name]
        true_mask = true_regions[region_name]

        intersection = np.logical_and(pred_mask, true_mask).sum()
        sum_masks = pred_mask.sum() + true_mask.sum()

        if sum_masks == 0:
            results[region_name] = 0.0  # both empty → 0, not 1
        else:
            results[region_name] = 2.0 * intersection / sum_masks

    return results
