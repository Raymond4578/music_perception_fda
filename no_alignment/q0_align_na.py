import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfda
import fdasrsf.utility_functions as uf

from utils import load_data, check_folder_na, data_after_align, l2norm

# check wheter all the requred folders are exist or not
check_folder_na()

########################################################

# Load the data
target_dict, data_dict = load_data('../data/')
# Get data after alignemnt
obs_data_dict, aligned_data_dict, warp_func_dict = data_after_align(target_dict, data_dict)

# Extract the amp and pha distance statistics
# amp_pha_dist_tb, amp_dist_dict, pha_dist_dict = list(), dict(), dict()
amp_pha_dist_tb, amp_dist_dict = list(), dict()
for music_idx in range(1, 13):
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    target = target_dict[music_idx]
    obs_data_ls = obs_data_dict[music_idx]
    # amp_dist_ls, pha_dist_ls = list(), list()
    amp_dist_ls = list()
    # calculate all the amp and pha dist for this specific music
    for obs_data in obs_data_ls:
        # Calculate the amplitude and phase distance
        # amp_dist, pha_dist = uf.elastic_distance(target, obs_data, t, method='DP2', lam=0.0)
        amp_dist = l2norm(target, obs_data, t)
        amp_dist_ls.append(amp_dist)
        # pha_dist_ls.append(pha_dist)
    amp_dist_dict[music_idx] = amp_dist_ls
    # pha_dist_dict[music_idx] = pha_dist_ls
    summ_stat_row = [
        music_idx,
        round(np.mean(amp_dist_ls), 2),
        round(np.std(amp_dist_ls), 2),
        None,
        None
    ]
    amp_pha_dist_tb.append(summ_stat_row)
amp_pha_dist_tb = pd.DataFrame(amp_pha_dist_tb, columns=['id', 'amp_mean', 'amp_std', 'pha_mean', 'pha_std'])
amp_pha_dist_tb.to_csv('./output_na/dataframe_na/amp_pha_dist_summ_stat_na.csv', index=False)

# Draw amp/pha dist distribution
for music_idx in range(1, 13):
    amp_dist_ls = amp_dist_dict[music_idx]
    plt.figure()
    plt.hist(amp_dist_ls, bins=15, density=True)
    plt.xlabel('Amplitude Distance')
    plt.tight_layout()
    plt.savefig(f'./output_na/plot_na/music{music_idx}_na/music{music_idx}_amp_dist_na.png', dpi=300)
    plt.close('all')

    # pha_dist_ls = pha_dist_dict[music_idx]
    # plt.figure()
    # plt.hist(pha_dist_ls, bins=15, density=True)
    # plt.xlabel('Phase Distance')
    # plt.tight_layout()
    # plt.savefig(f'./output_na/plot_na/music{music_idx}_na/music{music_idx}_pha_dist_na.png', dpi=300)
    # plt.close('all')