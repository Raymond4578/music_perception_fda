import argparse
import numpy as np
import pandas as pd
from fdasrsf import pairwise_align_functions
import fdasrsf.utility_functions as uf

from dataloader import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--music', type = int, default = 1, help = 'A music number from 1 to 12.')
args = parser.parse_args()

target_dict, data_dict = load_data('./data/')

def align_data(target_dict:dict, data_dict:dict, music_idx=1):
    # Then, do alignment for a specific music piece
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    target = target_dict[music_idx]
    data_df = data_dict[music_idx]
    obs_data_ls, aligned_data_ls, warp_func_ls = list(), list(), list()
    amp_dist_ls, pha_dist_ls = list(), list()
    for index, row in data_df.iterrows():
        # 这里可以看extract 数据index和受试者ID
        row_data = row.iloc[2:].values.astype(float)
        if np.isnan(row_data).all() == True:
            print('有NA！！！！！')
        obs_data_ls.append(row_data)
        aligned_data, warp_func = pairwise_align_functions(target, row_data, t, omethod='DP2')[:2]
        aligned_data_ls.append(aligned_data)
        warp_func_ls.append(warp_func)
        # 算一下amp和phase的dist吧
        amp_dist, pha_dist = uf.elastic_distance(target, aligned_data, t, method='DP2', lam=0.0)
        amp_dist_ls.append(amp_dist)
        pha_dist_ls.append(pha_dist)

    return obs_data_ls, aligned_data_ls, warp_func_ls, amp_dist_ls, pha_dist_ls

# formulate a table for amp/pha dist, a dictionary for aligned data and warpping function
amp_pha_dist_tb, obs_data_dict, aligned_data_dict, warp_func_dict = list(), dict(), dict(), dict()
for music_idx in range(1, 13):
    obs_data_ls, aligned_data_ls, warp_func_ls, amp_dist_ls, pha_dist_ls = align_data(target_dict, data_dict, music_idx=music_idx)
    data_row = [
        music_idx,
        round(np.mean(amp_dist_ls), 2),
        round(np.std(amp_dist_ls), 2),
        round(np.mean(pha_dist_ls), 2),
        round(np.std(pha_dist_ls), 2)
    ]
    amp_pha_dist_tb.append(data_row)
    obs_data_ls[music_idx] = obs_data_ls
    aligned_data_dict[music_idx] = aligned_data_ls
    warp_func_dict[music_idx] = warp_func_ls
amp_pha_dist_tb = pd.DataFrame(amp_pha_dist_tb, columns=['id', 'amp_mean', 'amp_std', 'pha_mean', 'pha_std'])

