import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

################################################
# Compare the amplitude and phase distance range
################################################

amp_pha_dist_a = pd.read_csv('./alignment/output/dataframe/amp_pha_dist_summ_stat.csv')
amp_pha_dist_na = pd.read_csv('./no_alignment/output_na/dataframe_na/amp_pha_dist_summ_stat_na.csv')
amp_pha_dist_a['align'] = 'aligned'
amp_pha_dist_na['align'] = 'misaligned'
amp_pha_dist = pd.concat([amp_pha_dist_na, amp_pha_dist_a], ignore_index=True)

# 设置全局字体大小
plt.rcParams.update({'font.size': 18})  # 将 14 替换为你想要的字体大小
for aspect in ['amp', 'pha']:
    fig, axes = plt.subplots(3, 4, figsize=(19, 10))
    # 确保 axes 是一维数组，方便遍历
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        amp_pha_dist_idx = amp_pha_dist[amp_pha_dist['id'] == i + 1]
        ax.errorbar(
            amp_pha_dist_idx['align'],
            amp_pha_dist_idx[f'{aspect}_mean'],
            yerr=amp_pha_dist_idx[f'{aspect}_std'],
            fmt='o',
            capsize=15
        )
        ax.set_xlim([-1, 2])
        ax.set_title(f'Music {i + 1}')

    if aspect == 'amp':
        plt.suptitle('Amplitude Distance Summary Statistics Comparison', fontsize=20)
    elif aspect == 'pha':
        plt.suptitle('Phase Distance Summary Statistics Comparison', fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.close('all')

################################################
# Compare the amplitude clustering results
################################################

clust_results_a = pd.read_csv('./alignment/output/dataframe/clust_pred_results.csv')
clust_results_na = pd.read_csv('./no_alignment/output_na/dataframe_na/clust_pred_results_na.csv')

amp_compare = clust_results_a['amp_pred'] == clust_results_na['amp_pred']

print(f'The agreement between aligned data and misaligned is {sum(amp_compare) / len(amp_compare):.2f}.')






