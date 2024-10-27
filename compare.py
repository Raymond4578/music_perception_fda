import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from utils import check_folder_compare

# check wheter all the required folders are exist or not
check_folder_compare()

################################################
# Compare the amplitude and phase distance range
################################################

amp_pha_dist_a = pd.read_csv('./alignment/output/dataframe/amp_pha_dist_summ_stat.csv')
amp_pha_dist_na = pd.read_csv('./no_alignment/output_na/dataframe_na/amp_pha_dist_summ_stat_na.csv')
amp_pha_dist_a['align'] = 'aligned'
amp_pha_dist_na['align'] = 'misaligned'
amp_pha_dist = pd.concat([amp_pha_dist_na, amp_pha_dist_a], ignore_index=True)
amp_pha_dist['emotion'] = (['anger'] * 3 + ['happiness'] * 3 + ['sadness'] * 3 + ['tenderness'] * 3) * 2
amp_dist_plot_df = amp_pha_dist[['id', 'amp_mean', 'amp_std', 'align', 'emotion']].copy()
# adjust id value for ploting
amp_dist_plot_df['id'] = amp_dist_plot_df['id'].astype(float)
amp_dist_plot_df.loc[:11, 'id'] -= 0.2 # 修改前12行 - 0.2
amp_dist_plot_df.loc[12:, 'id'] += 0.2 # 修改后12行 + 0.2

# 绘制图表
emo_color = {'anger': 'red', 'happiness': 'blue', 'sadness': 'green', 'tenderness': 'purple'}
align_color = {'misaligned': 0.2, 'aligned': 1}
plt.figure(figsize=(10, 6))
for (emotion, align), group_df in amp_dist_plot_df.groupby(['emotion', 'align']):
    color = mcolors.to_rgba(emo_color[emotion], alpha=align_color[align])
    plt.errorbar(
        group_df['id'],
        group_df['amp_mean'],
        yerr=group_df['amp_std'],
        fmt='o', capsize=5, label=f"{emotion} ({align})", color=color
    )
for x in np.arange(1.5, 12, 1):
    plt.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)
# 设置 x 轴刻度为 1 到 12
plt.xticks(np.arange(1, 13, step=1))
plt.xlim(0.5, 12.5)
plt.title("Amplitude Distance Compare Between Aligned and Misaligned Data")
plt.xlabel("Music ID")
plt.ylabel("Amplitude Distance")
plt.legend(title='Emotion', bbox_to_anchor=(1.36, 0.5), loc='center right')
plt.tight_layout()
plt.savefig(f'./compare/new_amp_dist_compare.png', dpi=300)
plt.close('all')



# # 设置全局字体大小
# plt.rcParams.update({'font.size': 18})  # 将 14 替换为你想要的字体大小
# # for aspect in ['amp', 'pha']:
# for aspect in ['amp']:
#     fig, axes = plt.subplots(3, 4, figsize=(19, 10))
#     # 确保 axes 是一维数组，方便遍历
#     axes = axes.flatten()
#     for i, ax in enumerate(axes):
#         amp_pha_dist_idx = amp_pha_dist[amp_pha_dist['id'] == i + 1]
#         ax.errorbar(
#             amp_pha_dist_idx['align'],
#             amp_pha_dist_idx[f'{aspect}_mean'],
#             yerr=amp_pha_dist_idx[f'{aspect}_std'],
#             fmt='o',
#             capsize=15
#         )
#         ax.set_xlim([-1, 2])
#         ax.set_title(f'Music {i + 1}')
#
#     if aspect == 'amp':
#         plt.suptitle('Amplitude Distance Summary Statistics Comparison', fontsize=20)
#     elif aspect == 'pha':
#         plt.suptitle('Phase Distance Summary Statistics Comparison', fontsize=20)
#     plt.tight_layout()
#     plt.savefig(f'./compare/{aspect}_dist_compare.png', dpi=300)
#     plt.close('all')
#
# ################################################
# # Compare the amplitude clustering results
# ################################################
#
# clust_results_a = pd.read_csv('./alignment/output/dataframe/clust_pred_results.csv')
# clust_results_na = pd.read_csv('./no_alignment/output_na/dataframe_na/clust_pred_results_na.csv')
#
# # A table for the count of predicted abnormal people
# amp_pred_count_na = clust_results_na.groupby('Piece')['amp_pred'].sum().astype(int).reset_index()
# amp_pred_count = clust_results_a.groupby('Piece')['amp_pred'].sum().astype(int).reset_index()
# pha_pred_count = clust_results_a.groupby('Piece')['pha_pred'].sum().astype(int).reset_index()
# pred_count = pd.concat([amp_pred_count_na, amp_pred_count['amp_pred'], pha_pred_count['pha_pred']], axis = 1)
# pred_count.columns = ['Piece', 'amp_pred_na', 'amp_pred_a', 'pha_pred_a']
# pred_count.to_csv(f'./compare/pred_abnormal_count.csv', index=False)
#
# # Generate a heatmap to visualize the difference
# amp_compare = clust_results_a['amp_pred'] == clust_results_na['amp_pred']
#
# clust_comp_df = clust_results_a[['ID', 'Piece']].copy()
# clust_comp_df['compare'] = [0 if val == False else 1 for val in amp_compare]
#
# # Change the long table into wide table
# clust_comp_wide_df = clust_comp_df.pivot(index='ID', columns='Piece', values='compare')
# clust_comp_wide_df = clust_comp_wide_df.reindex(sorted(clust_comp_wide_df.index, key=lambda x: int(x[1:])))
#
# # 创建 heatmap
# plt.figure(figsize=(10, 8))  # 设置图表大小
# sns.heatmap(clust_comp_wide_df, cmap="plasma", annot=False, cbar=True)
# plt.title('Agreement of Clustering Results')
# plt.tight_layout()
# plt.savefig(f'./compare/clust_compare.png', dpi=300)
# plt.close('all')
#
# print(f'The agreement between aligned data and misaligned is {sum(amp_compare) / len(amp_compare):.2f}.')






