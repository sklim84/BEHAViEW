import matplotlib.pyplot as plt
import pandas as pd

model='BGRL'
metric='pre'   # pre, rec, f1, auprc


metric_results = pd.read_csv(f'./{model}_{metric}_next.csv', index_col=0)
metric_results.head()

fontsize = 30

fig, ax = plt.subplots(figsize=(12, 9))
plt.xlabel(r'Number of Layers', fontsize=fontsize)
plt.ylabel(r'Hidden Dimension', fontsize=fontsize)
plt.xticks(range(6), [1, 2, 4, 8, 16, 32], fontsize=fontsize)
plt.yticks(range(5), [32, 64, 128, 256, 512], fontsize=fontsize)
im = ax.imshow(metric_results,interpolation="quadric")
cbar = fig.colorbar(im)
# cbar.ax.text(fontsize=fontsize)
cbar.ax.get_yaxis().labelpad = 30

if metric == 'pre':
    ylabel = 'Precision'
elif metric == 'rec':
    ylabel = 'Recall'
elif metric == 'f1':
    ylabel = 'F1-score'
elif metric == 'auprc':
    ylabel = 'AUPRC'

cbar.ax.set_ylabel(ylabel, rotation=270, fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

plt.savefig(f'./{model}_{metric}_cont_next.png', bbox_inches='tight')
plt.tight_layout()
plt.show()
