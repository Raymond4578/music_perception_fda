import re
import glob
import numpy as np
import pandas as pd
import skfda
import matplotlib.pyplot as plt
from fdasrsf import pairwise_align_functions
from skfda.ml.clustering import KMeans
from skfda.exploratory.visualization.clustering import ClusterPlot

from utils import check_folder

# check wheter all the requred folders are exist or not
check_folder()

########################################################

# read in targets
targets = pd.read_csv('./data/target_music.csv')
target_dict = dict()
for i in range(1, targets.shape[1]):
    target = targets.iloc[:, i].values
    target_dict[i] = target[~np.isnan(target)]


# read in data
file_paths = glob.glob('./data/music*.csv')
# Use regular expressions to extract numbers from each path
def extract_number(path):
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())
    return None  # For path without number，return None
# Sort the list of paths by the numbers extracted from the paths
file_paths = sorted(file_paths, key=extract_number)
pattern = re.compile(r'music(\d+)\.csv')

normal_data_clust = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1, 12: 1}
normal_warp_clust = {1: 0, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 1, 12: 0}

all_id = pd.read_csv(file_paths[0]).iloc[:, 1].values
# Initialize dict to store clustering results
aligned_data_clust = {ID: list() for ID in all_id}
warp_func_clust = {ID: list() for ID in all_id}
for path in file_paths:
    # get music id
    match = pattern.search(path)
    idx = int(match.group(1))
    df = pd.read_csv(path).dropna()
    valid_id = df.iloc[:, 1].values

    # then, extract each line, subtract with the target
    diff_aligned_data, diff_warp_func = list(), list()
    for index, row in df.iterrows():
        row_data = row.iloc[2:].values.astype(float)
        t = np.linspace(0, 1, row_data.shape[0])
        aligned_data, warp_func = pairwise_align_functions(target_dict[idx], row_data, t, omethod='DP2')[:2]
        # Get the absolute diff of aligned func
        diff_func_data = abs(aligned_data - target_dict[idx])
        diff_aligned_data.append(diff_func_data)
        # Get the absolute diff of warp func
        diff_warp_data = abs(warp_func - t)
        diff_warp_func.append(diff_warp_data)

    # Then do the clustering
    # transfrom data to skfda data
    dat_diff_fd = skfda.FDataGrid(data_matrix=diff_aligned_data, grid_points=t)
    warp_diff_fd = skfda.FDataGrid(data_matrix=diff_warp_func, grid_points=t)

    n_clusters = 2
    seed = 5003
    # First, do the clustering for the aligned data
    dat_diff_kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    dat_diff_kmeans.fit(dat_diff_fd)
    dat_diff_group = dat_diff_kmeans.predict(dat_diff_fd)
    # visualization of the clustering
    ClusterPlot(dat_diff_kmeans, dat_diff_fd).plot()
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{idx}/music{idx}_data_diff_clust.png', dpi=300)
    plt.close('all')

    data_diff_pred = {ID: pred for ID, pred in zip(valid_id, dat_diff_group)}
    for key in aligned_data_clust.keys():
        if key in valid_id:
            if data_diff_pred[key] == normal_data_clust[idx]:
                aligned_data_clust[key].append(0)
            else:
                aligned_data_clust[key].append(1)
        else:
            aligned_data_clust[key].append(np.nan)

    # Then, do the clustering for warping function
    warp_diff_kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    warp_diff_kmeans.fit(warp_diff_fd)
    warp_diff_group = warp_diff_kmeans.predict(warp_diff_fd)

    # visualization of the clustering
    ClusterPlot(warp_diff_kmeans, warp_diff_fd).plot()
    plt.tight_layout()
    plt.savefig(f'./output/plot/music{idx}/music{idx}_warp_diff_clust.png', dpi=300)
    plt.close('all')

    warp_diff_pred = {ID: pred for ID, pred in zip(valid_id, warp_diff_group)}
    for key in warp_func_clust.keys():
        if key in valid_id:
            if warp_diff_pred[key] == normal_warp_clust[idx]:
                warp_func_clust[key].append(0)
            else:
                warp_func_clust[key].append(1)
        else:
            warp_func_clust[key].append(np.nan)
# Finish doing the clustering
def get_emotion(idx):
    if idx in [1, 2, 3]:
        return 'anger'
    elif idx in [4, 5, 6]:
        return 'happiness'
    elif idx in [7, 8, 9]:
        return 'sadness'
    elif idx in [10, 11, 12]:
        return 'tenderness'
# Generate a dataframe to save the data
pred_df = []
for ID in all_id:
    for i, pred in enumerate(zip(aligned_data_clust[ID], warp_func_clust[ID])):
        amp_pred, pha_pred = pred
        row_data = {'ID': ID, 'Emotion': get_emotion(i + 1), 'Piece': i + 1, 'amp_pred': amp_pred, 'pha_pred': pha_pred}
        pred_df.append(row_data)
pred_df = pd.DataFrame(pred_df).dropna()
pred_df.to_csv(f'./output/dataframe/clust_pred_results.csv', index=False)