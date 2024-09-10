import re
import glob
import numpy as np
import pandas as pd
from fdasrsf import pairwise_align_functions
import fdasrsf.utility_functions as uf
from scipy.interpolate import make_interp_spline
import skfda
from skfda.representation.basis import BSplineBasis
from skfda.ml.regression import LinearRegression
import matplotlib.pyplot as plt

from utils import check_folder_na

# check wheter all the required folders are exist or not
check_folder_na()

########################################################

# read in targets
targets = pd.read_csv('../data/target_music.csv')
target_dict = dict()
for i in range(1, targets.shape[1]):
    target = targets.iloc[:, i].values
    target_dict[i] = target[~np.isnan(target)]


# read in data
file_paths = glob.glob('../data/music*.csv')
# Use regular expressions to extract numbers from each path
def extract_number(path):
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())
    return None  # For path without number，return None
# Sort the list of paths by the numbers extracted from the paths
file_paths = sorted(file_paths, key=extract_number)
pattern = re.compile(r'music(\d+)\.csv')

info_df = pd.DataFrame()
all_diff_obs_data = list()
for path in file_paths:
    # get music id
    match = pattern.search(path)
    idx = int(match.group(1))
    df = pd.read_csv(path).dropna()

    # first get all the information for multivariate variables
    temp_df = pd.DataFrame({'ID': df['ID']})
    if idx in [1, 2, 3]:
        temp_df['Emotion'] = 'anger'
    elif idx in [4, 5, 6]:
        temp_df['Emotion'] = 'happiness'
    elif idx in [7, 8, 9]:
        temp_df['Emotion'] = 'sadness'
    elif idx in [10, 11, 12]:
        temp_df['Emotion'] = 'tenderness'
    temp_df['Piece'] = idx

    t_new = np.linspace(0, 1, 61)
    # then, extract each line, subtract with the target, and resample
    diff_obs_data = list()
    amp_dist_ls, pha_dist_ls = list(), list()
    for index, row in df.iterrows():
        row_data = row.iloc[2:].values.astype(float)
        t = np.linspace(0, 1, row_data.shape[0])
        # Calculate the amplitude and phase distance
        amp_dist, pha_dist = uf.elastic_distance(target_dict[idx], row_data, t, method='DP2', lam=0.0)
        amp_dist_ls.append(amp_dist)
        pha_dist_ls.append(pha_dist)
        # Calculate the differencing function for the aligned function
        diff_func_data = row_data - target_dict[idx]
        # Create a BSpline interpolation object
        bspline = make_interp_spline(t, diff_func_data, k=3)
        diff_obs_data.append(bspline(t_new))
    all_diff_obs_data += diff_obs_data
    # add amp and pha dist value in dataframe
    temp_df['Amp_dist'] = amp_dist_ls
    temp_df['Pha_dist'] = pha_dist_ls
    info_df = pd.concat([info_df, temp_df], ignore_index=True)
# Save the amp and pha dist for regression as a csv file
info_df.to_csv(f'./output_na/dataframe_na/amp_pha_dist_reg_na.csv', index=False)

# all_diff_aligned_data_fd = skfda.FDataGrid(data_matrix=all_diff_aligned_data, grid_points=t_new)
# all_diff_warp_func_fd = skfda.FDataGrid(data_matrix=all_diff_warp_func, grid_points=t_new)
#
# # Define the B-spline basis function: defined by the degree and the number of knots.
# # For example, use a degree-3 B-spline basis and set 8 internal knots (excluding boundaries),
# # so 3 + 8 + 2 knots in total.
# bspline_basis = BSplineBasis(domain_range=(0, 1), n_basis=10, order=4)  # order = degree + 1
# all_diff_aligned_data_fd_basis = all_diff_aligned_data_fd.to_basis(bspline_basis)
# all_diff_warp_func_fd_basis = all_diff_warp_func_fd.to_basis(bspline_basis)
#
# emotions = ['anger', 'happiness', 'sadness', 'tenderness']
# X = pd.get_dummies(info_df['Emotion'], prefix='Emotion').astype(int)
#
# # Do regression on aligned data differencing function
# diff_aligned_data_linear_reg = LinearRegression(regularization=None, fit_intercept=False)
# diff_aligned_data_linear_reg.fit(X, all_diff_aligned_data_fd_basis)
# diff_aligned_data_coef_func_ls = diff_aligned_data_linear_reg.coef_
# for i in range(len(diff_aligned_data_coef_func_ls)):
#     diff_aligned_data_coef_func_ls[i].plot()
#     plt.axhline(0, color='r', linestyle='--')
#     # plt.title(f'Aligned Data Coefficient {emotions[i]}')
#     plt.ylim(-2.7, 3)
#     plt.tight_layout()
#     plt.savefig(f'./output/plot/func_reg/data_diff_coef_{emotions[i]}.png', dpi=300)
#     plt.close('all')
#
# # Do regression on warping function differencing function
# diff_warp_func_linear_reg = LinearRegression(regularization=None, fit_intercept=False)
# diff_warp_func_linear_reg.fit(X, all_diff_warp_func_fd_basis)
# diff_warp_func_coef_func_ls = diff_warp_func_linear_reg.coef_
# for i in range(len(diff_warp_func_coef_func_ls)):
#     diff_warp_func_coef_func_ls[i].plot()
#     plt.axhline(0, color='r', linestyle='--')
#     # plt.title(f'Warping Function Coefficient {emotions[i]}')
#     plt.ylim(-0.1, 0.1)
#     plt.tight_layout()
#     plt.savefig(f'./output/plot/func_reg/warp_diff_coef_{emotions[i]}.png', dpi=300)
#     plt.close('all')
