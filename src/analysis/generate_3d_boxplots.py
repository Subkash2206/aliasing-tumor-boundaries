import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(base_dir, 'results')
    
    val_base_path = os.path.join(results_dir, 'val_metrics.csv')
    val_blur_path = os.path.join(results_dir, 'val_metrics_blurpool.csv')
    
    df_base = pd.read_csv(val_base_path)
    df_blur = pd.read_csv(val_blur_path)
    
    df_base['Intervention'] = 'Baseline (SOTA)'
    df_blur['Intervention'] = 'BlurPool (Anti-Aliased)'
    
    df_combined = pd.concat([df_base, df_blur], ignore_index=True)
    
    # 1. Boxplot logic for Alias Violation Ratio
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_combined, x='Intervention', y='Alias_Violation_Ratio', palette=['blue', 'orange'])
    plt.title('Distribution of Alias Violation Ratio (AVR) Across 251 Patients')
    plt.ylabel('Alias Violation Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'boxplot_avr.png'), dpi=200)
    plt.close()
    
    # 2. Boxplot logic for Boundary F1
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_combined, x='Intervention', y='BF1_ET', palette=['blue', 'orange'])
    plt.title('Distribution of Boundary F1-ET Across 251 Patients')
    plt.ylabel('Boundary F1 (Enhancing Tumor)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'boxplot_bf1_et.png'), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
