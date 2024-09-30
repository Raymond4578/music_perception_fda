# import dtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from fdasrsf import pairwise_align_functions
import matplotlib.pyplot as plt

from utils import load_data

# Load the data
target_dict, data_dict = load_data('../data/')

music_idx = 1

t = np.linspace(0, 1, len(target_dict[music_idx]))
target = target_dict[music_idx]

data_df = data_dict[music_idx]

count = 0
for index, row in data_df.iterrows():
    # 这里可以看extract 数据index和受试者ID
    row_data = row.iloc[2:].values.astype(float)
    if np.isnan(row_data).all() == True:
        print('有NA！！！！！')
    # dtw results
    path = dtw.warping_path(target, row_data)
    x_coor = [i for i, j in path]
    y_coor = [j for i, j in path]
    # fda results
    aligned_data, warp_func = pairwise_align_functions(target, row_data, t, omethod='DP2')[:2]
    if count in [0]:
        # warping function compare
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(x_coor, y_coor)
        axes[0].set_title('DTW Warping Function')
        axes[1].plot(t, warp_func)
        axes[1].set_title('FDA Warping Function')
        plt.tight_layout()
        plt.savefig(f'./music{music_idx}_data{count}_warp.png', dpi=300)
        plt.close('all')
        # Compare plot
        dtwvis.plot_warping(target, row_data, path)
        # plt.plot(alignment.index1, alignment.index2)
        plt.tight_layout()
        plt.savefig(f'./music{music_idx}_data{count}_dtw_map.png', dpi=300)
        plt.close('all')

        # target and aligned data
        warp_data = [row_data[j] for i, j in path]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(range(len(target)), target, label='target')
        axes[0].plot(x_coor, warp_data, label='warpped data')
        axes[0].legend()
        axes[0].set_title('DTW Aligned Data')
        axes[1].plot(t, target, label='target')
        axes[1].plot(t, aligned_data, label='aligned data')
        axes[1].legend()
        axes[1].set_title('FDA Aligned Data')
        plt.tight_layout()
        plt.savefig(f'./music{music_idx}_data{count}_aligned_data.png', dpi=300)
        plt.close('all')

    count += 1
