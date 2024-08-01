import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfda
from fdasrsf import pairwise_align_functions
import fdasrsf.utility_functions as uf

from dataloader import load_data

# check wheter all the requred folders are exist or not
output_path = './output'
path_ls = [
    output_path,
    os.path.join(output_path, 'dataframe'),
    os.path.join(output_path, 'plot'),
]
for music_idx in range(1, 13):
    folder_name = f'music{music_idx}'
    path_ls.append(os.path.join(output_path, 'plot', folder_name))
path_ls.append(os.path.join(output_path, 'plot', 'func_reg'))

for path in path_ls:
    if not os.path.exists(path):
        os.makedirs(path)

########################################################

target_dict, data_dict = load_data('./data/')

def align_data(target_dict:dict, data_dict:dict, music_idx=1):
    # Then, do alignment for a specific music piece
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    target = target_dict[music_idx]
    data_df = data_dict[music_idx]
    obs_data_ls, aligned_data_ls, warp_func_ls = list(), list(), list()
    for index, row in data_df.iterrows():
        # 这里可以看extract 数据index和受试者ID
        row_data = row.iloc[2:].values.astype(float)
        if np.isnan(row_data).all() == True:
            print('有NA！！！！！')
        obs_data_ls.append(row_data)
        aligned_data, warp_func = pairwise_align_functions(target, row_data, t, omethod='DP2')[:2]
        aligned_data_ls.append(aligned_data)
        warp_func_ls.append(warp_func)
    return obs_data_ls, aligned_data_ls, warp_func_ls

# formulate a dictionary for observed data, aligned data and warpping functions
obs_data_dict, aligned_data_dict, warp_func_dict = dict(), dict(), dict()
for music_idx in range(1, 13):
    obs_data_ls, aligned_data_ls, warp_func_ls = align_data(target_dict, data_dict, music_idx=music_idx)
    obs_data_dict[music_idx] = obs_data_ls
    aligned_data_dict[music_idx] = aligned_data_ls
    warp_func_dict[music_idx] = warp_func_ls

# Extract the amp and pha distance statistics
amp_pha_dist_tb, amp_dist_dict, pha_dist_dict = list(), dict(), dict()
for music_idx in range(1, 13):
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    target = target_dict[music_idx]
    aligned_data_ls = aligned_data_dict[music_idx]
    amp_dist_ls, pha_dist_ls = list(), list()
    # calculate all the amp and pha dist for this specific music
    for aligned_data in aligned_data_ls:
        # 算一下amp和phase的dist吧
        amp_dist, pha_dist = uf.elastic_distance(target, aligned_data, t, method='DP2', lam=0.0)
        amp_dist_ls.append(amp_dist)
        pha_dist_ls.append(pha_dist)
    amp_dist_dict[music_idx] = amp_dist_ls
    pha_dist_dict[music_idx] = pha_dist_ls
    summ_stat_row = [
        music_idx,
        round(np.mean(amp_dist_ls), 2),
        round(np.std(amp_dist_ls), 2),
        round(np.mean(pha_dist_ls), 2),
        round(np.std(pha_dist_ls), 2)
    ]
    amp_pha_dist_tb.append(summ_stat_row)
amp_pha_dist_tb = pd.DataFrame(amp_pha_dist_tb, columns=['id', 'amp_mean', 'amp_std', 'pha_mean', 'pha_std'])
amp_pha_dist_tb.to_csv('./output/dataframe/amp_pha_dist_summ_stat.csv', index=False)

# Draw obs data, aligned data and wrapping functions
for music_idx in range(1, 13):
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    obs_data_fd = skfda.FDataGrid(data_matrix=obs_data_dict[music_idx], grid_points=t)
    align_data_fd = skfda.FDataGrid(data_matrix=aligned_data_dict[music_idx], grid_points=t)
    warp_func_fd = skfda.FDataGrid(data_matrix=warp_func_dict[music_idx], grid_points=t)
    # draw observed data
    plt.figure()
    obs_data_fd.plot()
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_obs_data.png', dpi=300)
    plt.close('all')
    # draw aligned data
    plt.figure()
    align_data_fd.plot()
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_aligned_data.png', dpi=300)
    plt.close('all')
    # draw warpping functions
    plt.figure()
    warp_func_fd.plot()
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_warp_func.png', dpi=300)
    plt.close('all')

# Draw amp/pha dist distribution
for music_idx in range(1, 13):
    amp_dist_ls = amp_dist_dict[music_idx]
    plt.figure()
    plt.hist(amp_dist_ls, bins=15, density=True)
    plt.xlabel('Amplitude Distance')
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_amp_dist.png', dpi=300)
    plt.close('all')

    pha_dist_ls = pha_dist_dict[music_idx]
    plt.figure()
    plt.hist(pha_dist_ls, bins=15, density=True)
    plt.xlabel('Phase Distance')
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_pha_dist.png', dpi=300)
    plt.close('all')