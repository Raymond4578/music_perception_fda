import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fdasrsf.utility_functions as uf

from utils import load_data, l2norm, get_emo

output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the data
target_dict, data_dict = load_data('../data/')

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
print(
    "The summary statistics of amplitude distance for not aligned data:\n",
    not_aligned_amp_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).round(2)
)
print(
    "The summary statistics of amplitude distance for aligned data:\n",
    aligned_amp_pha_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).round(2)
)
print(
    "The summary statistics of phase distance for aligned data:\n",
    aligned_amp_pha_dist_full_df.groupby("Piece")["Pha_dist"].agg(["mean", "std"]).round(2)
)

#############################################################################
# compare the amplitude distance range
#############################################################################

print(not_aligned_amp_dist_full_df)

# # set up a
# amp_dist_df = pd.concat(
#     [
#         not_aligned_amp_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).reset_index(),
#         aligned_amp_pha_dist_full_df.groupby("Piece")["Amp_dist"].agg(["mean", "std"]).reset_index()
#     ],
#     ignore_index=True
# )
# amp_dist_df = amp_dist_df.assign(
#     align=(["misaligned"] * 12 + ["aligned"] * 12),
#     emotion=(['anger'] * 3 + ['happiness'] * 3 + ['sadness'] * 3 + ['tenderness'] * 3) * 2
# )

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
amp_dist_plot_df = amp_dist_full_df[["Emotion", "Piece", "Amp_dist"]]
amp_dist_plot_df['Piece'] = amp_dist_plot_df['Piece'].astype(float)
amp_dist_plot_df.loc[:not_aligned_amp_dist_full_df.shape[0] - 1, 'Piece'] -= 0.2 # 修改前12行 - 0.2
amp_dist_plot_df.loc[not_aligned_amp_dist_full_df.shape[0]:, 'Piece'] += 0.2 # 修改后12行 + 0.2

amp_dist_plot_df.boxplot(column="Amp_dist", by="Piece", grid=False)
plt.show()

