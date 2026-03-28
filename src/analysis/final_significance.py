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
    cross_arch_path = os.path.join(results_dir, 'cross_arch_comparison.csv')
    
    if not os.path.exists(val_metrics_path) or not os.path.exists(blurpool_metrics_path):
        print(f"Error: {val_metrics_path} or {blurpool_metrics_path} not found.")
        return
        
    df_base = pd.read_csv(val_metrics_path)
    df_blur = pd.read_csv(blurpool_metrics_path)
    
    base_bf1 = df_base['BF1_ET'].values
    blur_bf1 = df_blur['BF1_ET'].values
    
    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(base_bf1, blur_bf1)
    
    # Cohen's d (Effect Size)
    diff = blur_bf1 - base_bf1
    d = np.mean(diff) / np.std(diff, ddof=1)
    mean_improvement = float(np.mean(diff))
    
    stats_dict = {
        "p_value": float(p_value),
        "effect_size": float(d),
        "mean_improvement": mean_improvement
    }
    
    stats_out_path = os.path.join(results_dir, 'final_paper_stats.json')
    with open(stats_out_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    print(f"Saved stats to {stats_out_path}")
    print(json.dumps(stats_dict, indent=4))
    
    # Append ResNet50 to cross arch comparison
    if os.path.exists(cross_arch_path):
        df_cross = pd.read_csv(cross_arch_path)
    else:
        df_cross = pd.DataFrame(columns=['Architecture', 'Intervention', 'AVR_Reduction', 'Dice_ET', 'BF1_ET', 'HD95_ET'])
        
    # Baseline ResNet50
    base_avr = df_base['Avg_AVR'].mean()
    blur_avr = df_blur['Avg_AVR'].mean()
    
    avr_reduction_pct = (1.0 - (blur_avr / base_avr)) * 100.0 if base_avr > 0 else 0.0
    
    resnet_base = {
        'Architecture': 'resnet50',
        'Intervention': 'Baseline',
        'AVR_Reduction': '0.0%',
        'Dice_ET': 'N/A',
        'BF1_ET': float(df_base['BF1_ET'].mean()),
        'HD95_ET': float(df_base['HD95_ET'].mean())
    }
    
    resnet_blur = {
        'Architecture': 'resnet50',
        'Intervention': 'BlurPool',
        'AVR_Reduction': f"{avr_reduction_pct:.1f}%",
        'Dice_ET': 'N/A', # Or placeholder if needed
        'BF1_ET': float(df_blur['BF1_ET'].mean()),
        'HD95_ET': float(df_blur['HD95_ET'].mean())
    }
    
    # Append records
    new_rows = pd.DataFrame([resnet_base, resnet_blur])
    df_cross = pd.concat([df_cross, new_rows], ignore_index=True)
    
    # Save final table
    final_table_path = os.path.join(results_dir, 'final_summary_table.csv')
    df_cross.to_csv(final_table_path, index=False)
    print(f"Saved summary table to {final_table_path}")

if __name__ == "__main__":
    main()
