import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import skfda
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
import fdasrsf.utility_functions as uf

from utils import load_data, align_data, l2norm, get_emo

output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the data
target_dict, data_dict = load_data('../data/')
aligned_data_dict, warp_func_dict = align_data(target_dict, data_dict)

#############################################################################
# show the original data, aligned data and warping function
#############################################################################

music_idx = 6
t = np.linspace(0, 1, target_dict[music_idx].shape[0])

obs_data_df = data_dict[music_idx].copy()
# Drop first two column about idx and ID
obs_data_df.drop(obs_data_df.columns[:2], axis=1, inplace=True)
obs_data = np.array(obs_data_df).astype(float)

# Draw the plot
plt.figure(figsize=(5, 4))
for i, response in enumerate(obs_data):
    # Draw all the responses with dashed line
    plt.plot(t, response, linewidth=0.2, color='black', linestyle='--')
plt.plot(t, target_dict[music_idx], linewidth=2, color='blue', linestyle='-') # target
# plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(f'./output/showNotAlignedDataMusic{music_idx}.png', dpi=300)
plt.close('all')

aligned_data_df = aligned_data_dict[music_idx].copy()
# Drop first two column about idx and ID
aligned_data_df.drop(aligned_data_df.columns[:2], axis=1, inplace=True)
aligned_data = np.array(aligned_data_df).astype(float)

# Draw the plot
plt.figure(figsize=(5, 4))
for i, response in enumerate(aligned_data):
    # Draw all the responses with dashed line
    plt.plot(t, response, linewidth=0.2, color='black', linestyle='--')
plt.plot(t, target_dict[music_idx], linewidth=2, color='blue', linestyle='-') # target
# plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(f'./output/showAlignedDataMusic{music_idx}.png', dpi=300)
plt.close('all')

warp_func_df = warp_func_dict[music_idx].copy()
# Drop first two column about idx and ID
warp_func_df.drop(warp_func_df.columns[:2], axis=1, inplace=True)
warp_func = np.array(warp_func_df).astype(float)

# Draw the warping function plot
plt.figure(figsize=(5, 4))
for i, response in enumerate(warp_func):
    # Draw all the responses with dashed line
    plt.plot(t, response, linewidth=0.2, color='black', linestyle='--')
# plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(f'./output/showWarpFuncMusic{music_idx}.png', dpi=300)
plt.close('all')

#############################################################################
# calculate the percentage of variance explianed by PC1
#############################################################################

# Initialize a dataframe to save fpca variance explained
fpca_var_explained = list()
# Initialize dict to save mean function and pc1
fpca_info_dict = dict()

# Set a BSpline Basis
bspline_basis = BSplineBasis(n_basis=10)
for music_idx in range(1, 13):
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    obs_data_fd = skfda.FDataGrid(data_matrix=data_dict[music_idx].iloc[:, 2:], grid_points=t)
    # fit data with a basis for smoothness
    basis_obs_data_fd = obs_data_fd.to_basis(bspline_basis)

    obs_data_fpca = FPCA(n_components=2)
    obs_data_fpca.fit(basis_obs_data_fd)
    # extract the explained variance ratio
    obs_data_pc_explained_ratio = obs_data_fpca.explained_variance_ratio_
    # prepare for plot information
    obs_data_fpca_dict = {
        "mean": basis_obs_data_fd.mean()(t).flatten(),
        "pc1": obs_data_fpca.components_(t)[0, :, :].flatten()
    }


    aligned_data_fd = skfda.FDataGrid(data_matrix=aligned_data_dict[music_idx].iloc[:, 2:], grid_points=t)
    # fit data with a basis for smoothness
    basis_aligned_data_fd = aligned_data_fd.to_basis(bspline_basis)
    aligned_data_fpca = FPCA(n_components=2)
    aligned_data_fpca.fit(basis_aligned_data_fd)
    # extract the explained variance ratio
    aligned_data_pc_explained_ratio = aligned_data_fpca.explained_variance_ratio_
    # prepare for plot information
    aligned_data_fpca_dict = {
        "mean": basis_aligned_data_fd.mean()(t).flatten(),
        "pc1": aligned_data_fpca.components_(t)[0, :, :].flatten()
    }


    warp_func_fd = skfda.FDataGrid(data_matrix=warp_func_dict[music_idx].iloc[:, 2:], grid_points=t)
    # fit data with a basis for smoothness
    basis_warp_func_fd = warp_func_fd.to_basis(bspline_basis)
    warp_func_fpca = FPCA(n_components=2)
    warp_func_fpca.fit(basis_warp_func_fd)
    # extract the explained variance ratio
    warp_func_pc_explained_ratio = warp_func_fpca.explained_variance_ratio_
    # prepare for plot information
    warp_func_fpca_dict = {
        "mean": basis_warp_func_fd.mean()(t).flatten(),
        "pc1": warp_func_fpca.components_(t)[0, :, :].flatten()
    }


    # save all the explianed variance into the list
    fpca_var_explained.append(
        np.hstack((obs_data_pc_explained_ratio, aligned_data_pc_explained_ratio, warp_func_pc_explained_ratio))
    )

    # save all fpca info to fpca_info_dict
    fpca_info_dict[music_idx] = {
        "obs_data": obs_data_fpca_dict,
        "aligned_data": aligned_data_fpca_dict,
        "warp_func": warp_func_fpca_dict
    }

# transfer to a dataframe
fpca_var_explained = pd.DataFrame(fpca_var_explained)
fpca_var_explained.insert(0, "ID", np.arange(1, 13))
fpca_var_explained.columns = ["ID", "NA_AMP_PC1", "NA_AMP_PC2", "A_AMP_PC1", "A_AMP_PC2", "A_PHA_PC1", "A_PHA_PC2"]
fpca_var_explained.to_csv("./output/fpca_var_explained.csv", index=False)

# 定义格式化规则（除了 'ID' 列之外，其余列都保留 4 位小数）
formatters = {
    'NA_AMP_PC1': '{:.4f}'.format,
    'NA_AMP_PC2': '{:.4f}'.format,
    'A_AMP_PC1': '{:.4f}'.format,
    'A_AMP_PC2': '{:.4f}'.format,
    'A_PHA_PC1': '{:.4f}'.format,
    'A_PHA_PC2': '{:.4f}'.format,
}

# 生成 LaTeX 表格代码（不输出行索引）
latex_code = fpca_var_explained.to_latex(index=False, formatters=formatters)
# print(latex_code)

#############################################################################
# calculate the amplitude and phase distance for not aligned and aligned data
#############################################################################

not_aligned_amp_dist_full_df, aligned_amp_pha_dist_full_df = pd.DataFrame(), pd.DataFrame()
for music_idx, response_df in data_dict.items():
    valid_id = response_df.iloc[:, 1].values
    not_aligned_amp_dist_ls, aligned_amp_dist_ls, aligned_pha_dist_ls = list(), list(), list()
    for row_idx, row in response_df.iterrows():
        row_data = row.iloc[2:].values.astype(float)
        t = np.linspace(0, 1, row_data.shape[0])
        # calculate the amplitude distance for not aligned data
        not_aligned_amp_dist = l2norm(target_dict[music_idx], row_data, t)
        not_aligned_amp_dist_ls.append(not_aligned_amp_dist)
        # calculate the amplitude and phase distance for aligned data
        aligned_amp_dist, aligned_pha_dist = uf.elastic_distance(target_dict[music_idx], row_data, t, method='DP2', lam=0.0)
        aligned_amp_dist_ls.append(aligned_amp_dist)
        aligned_pha_dist_ls.append(aligned_pha_dist)
    # save not aligned amp dist data
    not_aligned_new_df = {
        'ID': valid_id,
        'Emotion': get_emo(music_idx),
        'Piece': music_idx,
        'Amp_dist': not_aligned_amp_dist_ls
    }
    # save aligned amp and pha dist data
    aligned_new_df = {
        'ID': valid_id,
        'Emotion': get_emo(music_idx),
        'Piece': music_idx,
        'Amp_dist': aligned_amp_dist_ls,
        "Pha_dist": aligned_pha_dist_ls
    }
    not_aligned_amp_dist_full_df = pd.concat(
        [not_aligned_amp_dist_full_df, pd.DataFrame(not_aligned_new_df)], ignore_index=True
    )
    aligned_amp_pha_dist_full_df = pd.concat(
        [aligned_amp_pha_dist_full_df, pd.DataFrame(aligned_new_df)], ignore_index=True
    )
# save all the data to csv
not_aligned_amp_dist_full_df.to_csv('./output/notAlignedAmpDist.csv', index=False)
aligned_amp_pha_dist_full_df.to_csv('./output/alignedAmpPhaDist.csv', index=False)
# print out the mean and std for the data
# print(
#     "The summary statistics of amplitude distance for not aligned data:\n",
#     not_aligned_amp_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).round(2)
# )
# print(
#     "The summary statistics of amplitude distance for aligned data:\n",
#     aligned_amp_pha_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).round(2)
# )
# print(
#     "The summary statistics of phase distance for aligned data:\n",
#     aligned_amp_pha_dist_full_df.groupby("Piece")["Pha_dist"].agg(["mean", "std"]).round(2)
# )

#############################################################################
# compare the amplitude distance range
#############################################################################

amp_dist_full_df = pd.concat(
    [
        not_aligned_amp_dist_full_df,
        aligned_amp_pha_dist_full_df.drop("Pha_dist", axis=1)
    ],
    ignore_index=True
).assign(
    Align=(["misaligned"] * not_aligned_amp_dist_full_df.shape[0] + ["aligned"] * aligned_amp_pha_dist_full_df.shape[0])
)
# adjust id value for ploting
amp_dist_plot_df = amp_dist_full_df[["Emotion", "Piece", "Amp_dist", "Align"]].copy()
amp_dist_plot_df['Piece'] = amp_dist_plot_df['Piece'].astype(float)

misaligned_amp_dist_plot_df = amp_dist_plot_df[amp_dist_plot_df["Align"] == "misaligned"]
aligned_amp_dist_plot_df = amp_dist_plot_df[amp_dist_plot_df["Align"] == "aligned"]
misaligned_amp_dist_plot_df.loc[:, "Piece"] -= 0.2 # 修改前12行 - 0.2
aligned_amp_dist_plot_df.loc[:, "Piece"] += 0.2 # 修改前12行 - 0.2

# 按照 group 分组，提取每个组的 value 列数据，并保证顺序正确
grouped_misaligned_amp_dist_plot_df = misaligned_amp_dist_plot_df.groupby('Piece')['Amp_dist'].apply(list).sort_index()
grouped_aligned_amp_dist_plot_df = aligned_amp_dist_plot_df.groupby('Piece')['Amp_dist'].apply(list).sort_index()
# grouped_amp_dist_plot_df = amp_dist_plot_df.groupby('Piece')['Amp_dist'].apply(list).sort_index()

# 取出分组的实际值（作为 x 轴位置）和对应的数据列表
positions_misaligned = grouped_misaligned_amp_dist_plot_df.index.tolist()
positions_aligned = grouped_aligned_amp_dist_plot_df.index.tolist()

# 创建图形，使用 patch_artist=True 使得箱形图可以填充颜色，同时设置宽度方便两组图形并列
plt.figure(figsize=(10, 6))
bp1 = plt.boxplot(grouped_misaligned_amp_dist_plot_df.tolist(), positions=positions_misaligned,
                  patch_artist=True, widths=0.15)
bp2 = plt.boxplot(grouped_aligned_amp_dist_plot_df.tolist(), positions=positions_aligned,
                  patch_artist=True, widths=0.15)

# 设置不同箱形图的颜色
for box in bp1['boxes']:
    box.set_facecolor('lightblue')  # 第一个类别的颜色
for box in bp2['boxes']:
    box.set_facecolor('lightgreen')  # 第二个类别的颜色

emo_range = {
    'anger': (0.5, 3.5),
    'happiness': (3.5, 6.5),
    'sadness': (6.5, 9.5),
    'tenderness': (9.5, 12.5)
}
# 用箭头表示emotion的范围
for emo, (start, end) in emo_range.items():
    plt.annotate(
        '', xy=(end, max(amp_dist_plot_df['Amp_dist']) + 0.7),
        xytext=(start, max(amp_dist_plot_df['Amp_dist']) + 0.7),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    plt.text(
        (start + end) / 2, max(amp_dist_plot_df['Amp_dist']) + 0.8,
        f'{emo}', ha='center', va='bottom', fontsize=10
    )

for x in np.arange(1.5, 12, 1):
    plt.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)
for x in np.arange(3.5, 10, 3):
    plt.axvline(x=x, color='black', linestyle='-', linewidth=1)

# 创建代理对象，分别代表两个类别
blue_patch = mpatches.Patch(color='lightblue', label='Misaligned')
green_patch = mpatches.Patch(color='lightgreen', label='Aligned')

# 设置 x 轴刻度为 1 到 12
plt.xticks(np.arange(1, 13, step=1), np.arange(1, 13, step=1))
plt.xlim(0.5, 12.5)
plt.ylim(1, 11.5)
plt.title("Amplitude Distance Compare Between Aligned and Misaligned Data")
plt.xlabel("Music ID")
plt.ylabel("Amplitude Distance")
# 添加图例
plt.legend(title="Alignment Status", handles=[blue_patch, green_patch], bbox_to_anchor=(1.2, 0.5), loc='center right')
plt.tight_layout()
plt.savefig(f'./output/ampDistBoxplotCompare.png', dpi=300)
plt.close('all')


