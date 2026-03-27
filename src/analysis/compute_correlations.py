import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def compute_correlations():
    df = pd.read_csv('results/val_metrics.csv')
    
    results = {}
    
    for region in ['WT', 'TC', 'ET']:
        try:
            r_bf1, p_bf1 = stats.pearsonr(df['Avg_AVR'], df[f'BF1_{region}'])
            rho_bf1, p_s_bf1 = stats.spearmanr(df['Avg_AVR'], df[f'BF1_{region}'])
            r_hd, p_hd = stats.pearsonr(df['Avg_AVR'], df[f'HD95_{region}'])
            
            results[region] = {
                'BF1': {'pearson_r': float(r_bf1), 'p_value': float(p_bf1), 'spearman_rho': float(rho_bf1)},
                'HD95': {'pearson_r': float(r_hd), 'p_value': float(p_hd)}
            }
        except Exception as e:
            print(f"Error computing stats for {region}: {e}")
            results[region] = {
                'BF1': {'pearson_r': 0.0, 'p_value': 1.0, 'spearman_rho': 0.0},
                'HD95': {'pearson_r': 0.0, 'p_value': 1.0}
            }
            
    with open('results/baseline_correlation_report.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"ET Pearson r: {results['ET']['BF1']['pearson_r']:.4f}")
    print(f"ET p-value: {results['ET']['BF1']['p_value']:.4e}")
    
    # Scatter 
    plt.figure(figsize=(8,6))
    if 'BF1_ET' in df.columns and len(df) > 2:
        sns.regplot(data=df, x='Avg_AVR', y='BF1_ET', scatter_kws={'alpha':0.5})
        plt.title('Avg AVR vs. Boundary F1 (ET)')
        plt.xlabel('Average AVR')
        plt.ylabel('Boundary F1 (ET)')
        plt.savefig('results/scatter_avr_vs_bf1_et.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    compute_correlations()
