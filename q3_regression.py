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
# 使用正则表达式从每个路径中提取数字
def extract_number(path):
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())
    return None  # 对于没有数字的路径，返回None
# 对路径列表进行排序，排序键为路径中的数字
file_paths = sorted(file_paths, key=extract_number)
pattern = re.compile(r'music(\d+)\.csv')

data_dict = dict()
info_df = pd.DataFrame()
all_diff_aligned_data = list()
all_diff_warp_func = list()
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
    diff_aligned_data, diff_warp_func = list(), list()
    amp_dist_ls, pha_dist_ls = list(), list()
    for index, row in df.iterrows():
        row_data = row.iloc[2:].values.astype(float)
        t = np.linspace(0, 1, row_data.shape[0])
        aligned_data, warp_func = pairwise_align_functions(target_dict[idx], row_data, t, omethod='DP2')[:2]
        # 算一下amp和phase的dist吧
        amp_dist, pha_dist = uf.elastic_distance(target_dict[idx], aligned_data, t, method='DP2', lam=0.0)
        amp_dist_ls.append(amp_dist)
        pha_dist_ls.append(pha_dist)
        # 搞定aligned func的diff
        diff_func_data = aligned_data - target_dict[idx]
        # 创建BSpline插值对象
        bspline = make_interp_spline(t, diff_func_data, k=3)
        diff_aligned_data.append(bspline(t_new))
        # 搞定warp func的diff
        diff_warp_data = warp_func - t
        # 创建BSpline插值对象
        bspline = make_interp_spline(t, diff_warp_data, k=3)
        diff_warp_func.append(bspline(t_new))
    all_diff_aligned_data += diff_aligned_data
    all_diff_warp_func += diff_warp_func
    # add amp and pha dist value in dataframe
    temp_df['Amp_dist'] = amp_dist_ls
    temp_df['Pha_dist'] = pha_dist_ls
    info_df = pd.concat([info_df, temp_df], ignore_index=True)
# Save the amp and pha dist for regression as a csv file
info_df.to_csv(f'./output/dataframe/amp_pha_dist_reg.csv', index=False)

all_diff_aligned_data_fd = skfda.FDataGrid(data_matrix=all_diff_aligned_data, grid_points=t_new)
all_diff_warp_func_fd = skfda.FDataGrid(data_matrix=all_diff_warp_func, grid_points=t_new)

# 定义B样条基函数：从度数和节点数来定义
# 例如，使用3度的B样条基，并设定8个内节点（不包含边界），因此总节点数为 3 + 8 + 2
bspline_basis = BSplineBasis(domain_range=(0, 1), n_basis=10, order=4)  # order = degree + 1
all_diff_aligned_data_fd_basis = all_diff_aligned_data_fd.to_basis(bspline_basis)
all_diff_warp_func_fd_basis = all_diff_warp_func_fd.to_basis(bspline_basis)

emotions = ['anger', 'happiness', 'sadness', 'tenderness']
X = pd.get_dummies(info_df['Emotion'], prefix='Emotion').astype(int)

# Do regression on aligned data differencing function
diff_aligned_data_linear_reg = LinearRegression(regularization=None, fit_intercept=False)
diff_aligned_data_linear_reg.fit(X, all_diff_aligned_data_fd_basis)
diff_aligned_data_coef_func_ls = diff_aligned_data_linear_reg.coef_
for i in range(len(diff_aligned_data_coef_func_ls)):
    diff_aligned_data_coef_func_ls[i].plot()
    plt.axhline(0, color='r', linestyle='--')
    # plt.title(f'Aligned Data Coefficient {emotions[i]}')
    plt.ylim(-2.7, 3)
    plt.tight_layout()
    plt.savefig(f'./output/plot/func_reg/data_diff_coef_{emotions[i]}.png', dpi=300)
    plt.close('all')

# Do regression on warping function differencing function
diff_warp_func_linear_reg = LinearRegression(regularization=None, fit_intercept=False)
diff_warp_func_linear_reg.fit(X, all_diff_warp_func_fd_basis)
diff_warp_func_coef_func_ls = diff_warp_func_linear_reg.coef_
for i in range(len(diff_warp_func_coef_func_ls)):
    diff_warp_func_coef_func_ls[i].plot()
    plt.axhline(0, color='r', linestyle='--')
    # plt.title(f'Warping Function Coefficient {emotions[i]}')
    plt.ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(f'./output/plot/func_reg/warp_diff_coef_{emotions[i]}.png', dpi=300)
    plt.close('all')
