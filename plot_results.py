import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('model_comparison_results.csv')
balanced_results_df = pd.read_csv('model_comparison_results_balanced.csv')

comparison = pd.concat([results_df.assign(strategy='imbalanced'), balanced_results_df.assign(strategy='balanced')])
order = (balanced_results_df.sort_values(by='f1_macro', ascending=False)['model_name'])

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
plot = sns.barplot(
    data=comparison,
    x='model_name',
    y='f1_macro',
    hue='strategy',
    order=order
)

plt.title('Comparison of F1-Macro Scores: Imbalanced vs. Balanced Training')
plt.ylabel('F1-Macro Score')
plt.xlabel('Model Architecture')
plt.ylim(0, 1.0)
plt.legend(title='Data Strategy', loc='upper left')
plt.tight_layout()
plt.savefig('f1_macro_comparison.png')
plt.show()