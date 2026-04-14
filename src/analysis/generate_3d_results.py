import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    
    val_base_path = os.path.join(results_dir, 'val_metrics_baseline_3d.csv')
    val_blur_path = os.path.join(results_dir, 'val_metrics_blurpool_3d.csv')
    
    if not os.path.exists(val_base_path) or not os.path.exists(val_blur_path):
        print("Error: 3D CSV metric files are missing!")
        return
        
    df_base = pd.read_csv(val_base_path)
    df_blur = pd.read_csv(val_blur_path)

    # Filtering out exact zeros (empty predictions)
    df_base = df_base[df_base['BF1_ET'] > 0].copy()
    df_blur = df_blur[df_blur['BF1_ET'] > 0].copy()
    
    # 1. Final Summary Table
    base_avr = df_base['Alias_Violation_Ratio'].mean()
    blur_avr = df_blur['Alias_Violation_Ratio'].mean()
    avr_reduction_pct = (1.0 - (blur_avr / base_avr)) * 100.0 if base_avr > 0 else 0.0

    summary_data = [
        {
            'Architecture': 'SegResNet (3D)',
            'Intervention': 'Baseline',
            'AVR_Reduction': '0.0%',
            'Dice_ET': f"{df_base['Dice_ET'].mean() * 100:.2f}%",
            'BF1_ET': f"{df_base['BF1_ET'].mean() * 100:.2f}%"
        },
        {
            'Architecture': 'SegResNet (3D)',
            'Intervention': 'BlurPool',
            'AVR_Reduction': f"{avr_reduction_pct:.1f}%",
            'Dice_ET': f"{df_blur['Dice_ET'].mean() * 100:.2f}%",
            'BF1_ET': f"{df_blur['BF1_ET'].mean() * 100:.2f}%"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, 'final_summary_table_3d.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Serialized: {summary_path}")

    # 2. Final Statistical JSON (Using shared valid subjects)
    valid_subjects = list(set(df_base['Case_ID']).intersection(set(df_blur['Case_ID'])))
    df_base_matched = df_base[df_base['Case_ID'].isin(valid_subjects)].sort_values('Case_ID')
    df_blur_matched = df_blur[df_blur['Case_ID'].isin(valid_subjects)].sort_values('Case_ID')

    base_bf1 = df_base_matched['BF1_ET'].values
    blur_bf1 = df_blur_matched['BF1_ET'].values
    
    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(base_bf1, blur_bf1)
    diff = blur_bf1 - base_bf1
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    stats_dict = {
        "dataset_size": len(valid_subjects),
        "validation_p_value": float(p_value),
        "effect_size_cohens_d": float(d),
        "mean_bf1_delta": float(np.mean(diff)),
        "baseline_avr_mean": float(base_avr),
        "blurpool_avr_mean": float(blur_avr),
        "avr_reduction_percent": float(avr_reduction_pct)
    }
    stats_out_path = os.path.join(results_dir, 'final_paper_stats_3d.json')
    with open(stats_out_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Serialized: {stats_out_path}")

    # 3. Correlation Scatter Graph
    df_base['Hue'] = 'Baseline'
    df_blur['Hue'] = 'BlurPool'
    df_combined = pd.concat([df_base, df_blur], ignore_index=True)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df_combined, x='Alias_Violation_Ratio', y='BF1_ET', hue='Hue', alpha=0.6, s=50)
    plt.title('Spectral Aliasing Variance vs Boundary F1 (3D SegResNet)')
    plt.xlabel('Alias Violation Ratio (AVR)')
    plt.ylabel('Boundary F1 — Enhancing Tumor (ET)')
    
    # Draw reference means
    plt.axvline(base_avr, color='blue', linestyle='--', alpha=0.5, label='Baseline Mean AVR')
    plt.axvline(blur_avr, color='orange', linestyle='--', alpha=0.5, label='BlurPool Mean AVR')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(results_dir, 'scatter_avr_vs_bf1_et_3d.png')
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"Serialized Graph: {scatter_path}")

if __name__ == "__main__":
    main()
