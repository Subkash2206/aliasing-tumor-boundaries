import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics.boundary_f1 import compute_boundary_f1, compute_dice

def create_synthetic_circle(size=100, radius=20, center=(50, 50), label=4):
    """
    Generates a generic 'tumor' circle in a grid.
    Default label=4 means it counts as WT, TC, and ET in our BraTS mapping.
    """
    grid_y, grid_x = np.ogrid[:size, :size]
    mask = (grid_x - center[0])**2 + (grid_y - center[1])**2 <= radius**2
    result = np.zeros((size, size), dtype=int)
    result[mask] = label
    return result

def test_monotonicity():
    print("\n--- Monotonicity Test: Dice vs. Boundary F1 ---")
    
    # Ground truth (ET = 4)
    y_true = create_synthetic_circle(center=(50, 50))
    
    print(f"{'Shift':<10} | {'Dice (WT)':<15} | {'Boundary F1 (WT)':<20}")
    print("-" * 50)
    
    dice_scores = []
    bf1_scores = []
    
    for shift in range(4):
        # Shift predicted circle gradually
        y_pred = create_synthetic_circle(center=(50 + shift, 50))
        
        # Test tolerance=1 to easily see drastic drop
        dice_dict = compute_dice(y_pred, y_true)
        bf1_dict = compute_boundary_f1(y_pred, y_true, tolerance=1)
        
        dsc = dice_dict['WT']
        bf1 = bf1_dict['WT']
        
        dice_scores.append(dsc)
        bf1_scores.append(bf1)
        
        print(f"{shift:<10} | {dsc:<15.4f} | {bf1:<20.4f}")
        
        if shift > 0:
            dsc_drop = 1.0 - dsc
            bf1_drop = 1.0 - bf1
            assert bf1_drop >= dsc_drop, "Boundary F1 is not dropping more sharply than Dice!"

    print("\n[SUCCESS] Monotonicity Test Passed! Boundary F1 decreases more sharply than Dice.")

if __name__ == "__main__":
    test_monotonicity()
