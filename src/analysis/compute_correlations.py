"""
compute_correlations.py

Updated to apply two rounds of filtering before computing Pearson r (Fix #3):
  1. Remove rows where BF1_ET == 0 AND HD95_ET == 100.0
     (guaranteed empty-prediction slices that add noise)
  2. Report ET-only subset where GT has >= 100 ET voxels
     (scientifically valid subset for the ET correlation claim)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def compute_correlations(
    csv_path    = 'results/val_metrics.csv',
    output_json = 'results/baseline_correlation_report.json',
    scatter_path= 'results/scatter_avr_vs_bf1_et.png',
):
    """
    Reads real per-sample validation metrics from csv_path and computes:
      - Pearson r and Spearman rho between Avg_AVR and BF1_{WT,TC,ET}
      - Pearson r and Spearman rho between Avg_AVR and HD95_{WT,TC,ET}
    Applies empty-prediction filtering and ET-only subset analysis.
    Saves the full report to output_json.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metrics file not found: {csv_path}\n"
            "Run train_baseline.py to completion first."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    required_cols = [
        'Avg_AVR',
        'BF1_WT', 'BF1_TC', 'BF1_ET',
        'HD95_WT', 'HD95_TC', 'HD95_ET',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if len(df) < 3:
        raise ValueError(
            f"Only {len(df)} rows found in {csv_path}. "
            "Need at least 3 samples to compute meaningful correlations."
        )

    # -----------------------------------------------------------------------
    # Fix #3: Filter empty-prediction slices before Pearson r
    # Empty-prediction slices: BF1_ET == 0 AND HD95_ET == 100.0
    # -----------------------------------------------------------------------
    mask_empty = (df['BF1_ET'] == 0.0) & (df['HD95_ET'] == 100.0)
    df_filtered = df[~mask_empty].copy()
    n_dropped   = len(df) - len(df_filtered)

    print(f"\nEmpty-prediction filter: {n_dropped} rows dropped "
          f"(BF1_ET==0 AND HD95_ET==100). {len(df_filtered)} rows remain.")

    if len(df_filtered) < 3:
        print("WARNING: Too few rows after filtering — using unfiltered data for correlations.")
        df_use = df
        filtered_note = "unfiltered (too few rows survived empty-pred filter)"
    else:
        df_use = df_filtered
        filtered_note = f"filtered ({n_dropped} empty-pred rows removed)"

    # -----------------------------------------------------------------------
    # Compute correlations on filtered data
    # -----------------------------------------------------------------------
    results = {}

    for region in ['WT', 'TC', 'ET']:
        bf1_col  = f'BF1_{region}'
        hd95_col = f'HD95_{region}'

        avr_vals  = df_use['Avg_AVR'].values
        bf1_vals  = df_use[bf1_col].values
        hd95_vals = df_use[hd95_col].values

        # Only compute if there is enough variance
        if len(set(avr_vals)) < 2 or len(set(bf1_vals)) < 2:
            r_bf1, p_bf1       = float('nan'), float('nan')
            rho_bf1, p_s_bf1   = float('nan'), float('nan')
        else:
            r_bf1,   p_bf1   = stats.pearsonr(avr_vals, bf1_vals)
            rho_bf1, p_s_bf1 = stats.spearmanr(avr_vals, bf1_vals)

        if len(set(avr_vals)) < 2 or len(set(hd95_vals)) < 2:
            r_hd95, p_hd95       = float('nan'), float('nan')
            rho_hd95, p_s_hd95   = float('nan'), float('nan')
        else:
            r_hd95,   p_hd95   = stats.pearsonr(avr_vals, hd95_vals)
            rho_hd95, p_s_hd95 = stats.spearmanr(avr_vals, hd95_vals)

        results[region] = {
            'BF1': {
                'pearson_r':    float(r_bf1)   if not np.isnan(r_bf1)   else None,
                'pearson_p':    float(p_bf1)   if not np.isnan(p_bf1)   else None,
                'spearman_rho': float(rho_bf1) if not np.isnan(rho_bf1) else None,
                'spearman_p':   float(p_s_bf1) if not np.isnan(p_s_bf1) else None,
            },
            'HD95': {
                'pearson_r':    float(r_hd95)   if not np.isnan(r_hd95)   else None,
                'pearson_p':    float(p_hd95)   if not np.isnan(p_hd95)   else None,
                'spearman_rho': float(rho_hd95) if not np.isnan(rho_hd95) else None,
                'spearman_p':   float(p_s_hd95) if not np.isnan(p_s_hd95) else None,
            },
        }

    # -----------------------------------------------------------------------
    # ET-only subset: GT >= 100 ET voxels (Fix #12)
    # -----------------------------------------------------------------------
    et_subset_results = None
    if 'ET_GT_voxels' in df.columns:
        df_et = df[df['ET_GT_voxels'] >= 100].copy()
        print(f"\nET-only subset (GT ≥100 ET voxels): {len(df_et)} rows "
              f"(out of {len(df)} total)")

        if len(df_et) >= 3 and df_et['Avg_AVR'].nunique() > 1:
            r_et, p_et     = stats.pearsonr(df_et['Avg_AVR'], df_et['BF1_ET'])
            rho_et, ps_et  = stats.spearmanr(df_et['Avg_AVR'], df_et['BF1_ET'])
            et_subset_results = {
                'n': int(len(df_et)),
                'pearson_r':    float(r_et),
                'pearson_p':    float(p_et),
                'spearman_rho': float(rho_et),
                'spearman_p':   float(ps_et),
            }
            print(f"  Pearson r={r_et:.4f} (p={p_et:.3e})  "
                  f"Spearman ρ={rho_et:.4f} (p={ps_et:.3e})")
        else:
            print(f"  Not enough variance in ET-only subset (n={len(df_et)}) — skipping.")
    else:
        print("\nNOTE: 'ET_GT_voxels' column not present — "
              "ET-only subset analysis requires retraining with the updated scripts.")

    # -----------------------------------------------------------------------
    # Summary block
    # -----------------------------------------------------------------------
    results['_meta'] = {
        'csv_path':       csv_path,
        'n_total':        int(len(df)),
        'n_after_filter': int(len(df_use)),
        'n_dropped':      int(n_dropped),
        'filter_note':    filtered_note,
        'avg_avr_mean':   float(df_use['Avg_AVR'].mean()),
        'avg_avr_std':    float(df_use['Avg_AVR'].std()),
    }
    if et_subset_results is not None:
        results['_et_subset'] = et_subset_results

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nCorrelation report saved to {output_json}")
    print("\n--- Summary (filtered) ---")
    for region in ['WT', 'TC', 'ET']:
        r = results[region]
        print(f"  {region}  BF1:  Pearson r={r['BF1']['pearson_r']}  "
              f"(p={r['BF1']['pearson_p']})  |  "
              f"Spearman ρ={r['BF1']['spearman_rho']}  "
              f"(p={r['BF1']['spearman_p']})")
        print(f"  {region}  HD95: Pearson r={r['HD95']['pearson_r']}  "
              f"(p={r['HD95']['pearson_p']})  |  "
              f"Spearman ρ={r['HD95']['spearman_rho']}  "
              f"(p={r['HD95']['spearman_p']})")

    # Scatter plot: AVR vs BF1(ET) on filtered data
    if len(df_use) > 2:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df_use, x='Avg_AVR', y='BF1_ET',
                    scatter_kws={'alpha': 0.5})
        plt.title(f'Avg AVR vs. Boundary F1 (ET)\n({filtered_note})')
        plt.xlabel('Average Aliasing Variance Ratio (AVR)')
        plt.ylabel('Boundary F1 — Enhancing Tumor (ET)')
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")

    return results


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    compute_correlations()
