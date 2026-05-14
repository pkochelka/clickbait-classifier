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
plt.xticks(rotation=45, ha='right')
plt.legend(title='Data Strategy', loc='upper left')
plt.tight_layout()
plt.savefig('results/f1_macro_comparison.png')
plt.show()

def plot_per_class(df, strategy_name, filename):
    class_cols = ["f1_clickbait", "f1_no_clickbait"]
    long_df = df.melt(id_vars='model_name', value_vars=class_cols,
                      var_name='class', value_name='f1')
    long_df['class'] = long_df['class'].str.replace('f1_', '', regex=False)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=long_df, x='model_name', y='f1', hue='class', order=order)
    plt.title(f'Per-Class F1 Scores ({strategy_name} Training)')
    plt.ylabel('F1 Score')
    plt.xlabel('Model Architecture')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Class', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_per_class(results_df, 'Imbalanced', 'results/f1_per_class_imbalanced.png')
plot_per_class(balanced_results_df, 'Balanced', 'results/f1_per_class_balanced.png')