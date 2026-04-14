import pandas as pd
from scipy import stats

df_base = pd.read_csv('results/val_metrics.csv')
r_base, p_base = stats.pearsonr(df_base['Alias_Violation_Ratio'], df_base['BF1_ET'])

df_blur = pd.read_csv('results/val_metrics_blurpool.csv')
r_blur, p_blur = stats.pearsonr(df_blur['Alias_Violation_Ratio'], df_blur['BF1_ET'])

print(f"Baseline Pearson R: {r_base:.3f}")
print(f"BlurPool Pearson R: {r_blur:.3f}")

# Cross-Model Pearson R
combined_avr = pd.concat([df_base['Alias_Violation_Ratio'], df_blur['Alias_Violation_Ratio']])
combined_bf1 = pd.concat([df_base['BF1_ET'], df_blur['BF1_ET']])
r_cross, _ = stats.pearsonr(combined_avr, combined_bf1)
print(f"Cross-Model Pearson R: {r_cross:.3f}")
