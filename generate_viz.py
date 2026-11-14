import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'results.json')) as f:
    data = json.load(f)

cm_classic = np.array(data['cm_classic'])
cm_llm = np.array(data['cm_llm'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Experiment & Evaluation Summary\nClassical ML (Scikit-learn) vs LLM API (Gemini)', fontsize=16, fontweight='bold', y=1.03)

# 1. Confusion Matrix Classical
sns.heatmap(cm_classic, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix: Classical Model\n(TF-IDF + Logistic Regression)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 2. Confusion Matrix LLM
sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix: LLM API\n(Gemini 3.1 Flash Lite Zero-Shot)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

# 3. Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
classic_scores = [data['acc_classic'], data['prec_classic'], data['rec_classic'], data['f1_classic']]
llm_scores = [data['acc_llm'], data['prec_llm'], data['rec_llm'], data['f1_llm']]

x = np.arange(len(metrics))
width = 0.35

rects1 = axes[2].bar(x - width/2, classic_scores, width, label='Classical (Scikit-learn)', color='#3182bd')
rects2 = axes[2].bar(x + width/2, llm_scores, width, label='LLM API (Gemini)', color='#31a354')

axes[2].set_ylabel('Score (0 - 1.0)')
axes[2].set_title('Evaluation Metrics Comparison\n(Test Set: 40 Samples)', fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics, fontweight='bold')
axes[2].set_ylim(0, 1.15)
axes[2].legend(loc='upper right')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects1:
    height = rect.get_height()
    axes[2].annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    axes[2].annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(BASE_DIR, 'documentation', 'model_comparison_summary.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Visualization saved successfully to {output_path}')
