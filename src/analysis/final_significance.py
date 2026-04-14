import os
import json
import pandas as pd
import numpy as np
from scipy import stats

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    
    val_metrics_path = os.path.join(results_dir, 'val_metrics.csv')
    blurpool_metrics_path = os.path.join(results_dir, 'val_metrics_blurpool.csv')
    
    df_base = pd.read_csv(val_metrics_path)
    df_blur = pd.read_csv(blurpool_metrics_path)
    
    N = len(df_base)  # Should specifically be 251
    
    base_bf1 = df_base['BF1_ET'].values
    blur_bf1 = df_blur['BF1_ET'].values
    
    # Wilcoxon signed-rank test on ALL N=251
    stat, p_value = stats.wilcoxon(base_bf1, blur_bf1)
    
    # Cohen's d
    diff = blur_bf1 - base_bf1
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    base_avr = df_base['Alias_Violation_Ratio'].mean()
    blur_avr = df_blur['Alias_Violation_Ratio'].mean()
    avr_reduction_pct = (1.0 - (blur_avr / base_avr)) * 100.0 if base_avr > 0 else 0.0
    
    stats_dict = {
        "dataset_size": N,
        "statistical_test": "Wilcoxon Signed-Rank Test (Paired, N=251)",
        "validation_p_value": float(p_value),
        "effect_size_cohens_d": float(d),
        "mean_bf1_delta": float(np.mean(diff)),
        "baseline_avr_mean": float(base_avr),
        "blurpool_avr_mean": float(blur_avr),
        "avr_reduction_percent": float(avr_reduction_pct)
    }
    
    stats_out_path = os.path.join(results_dir, 'final_paper_stats.json')
    with open(stats_out_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    print(f"Saved exact N={N} stats to {stats_out_path}")
    
    # Re-calculate clean means for the final summary table
    df_cross = pd.DataFrame([
        {
            'Architecture': 'SegResNet (3D)',
            'Intervention': 'Baseline',
            'AVR_Reduction': '0.0%',
            'Dice_ET': f"{df_base['Dice_ET'].mean() * 100:.2f}%",
            'BF1_ET': f"{df_base['BF1_ET'].mean() * 100:.2f}%",
            'HD95_ET': "N/A"
        },
        {
            'Architecture': 'SegResNet (3D)',
            'Intervention': 'BlurPool',
            'AVR_Reduction': f"{avr_reduction_pct:.2f}%",
            'Dice_ET': f"{df_blur['Dice_ET'].mean() * 100:.2f}%",
            'BF1_ET': f"{df_blur['BF1_ET'].mean() * 100:.2f}%",
            'HD95_ET': "N/A"
        }
    ])
    
    final_table_path = os.path.join(results_dir, 'final_summary_table.csv')
    df_cross.to_csv(final_table_path, index=False)
    print(f"Saved aligned N={N} summary table to {final_table_path}")

if __name__ == "__main__":
    main()
