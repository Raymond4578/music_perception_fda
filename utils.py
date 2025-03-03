import os
import re
import glob
import numpy as np
import pandas as pd
from fdasrsf import pairwise_align_functions
import fdasrsf.utility_functions as uf
from scipy.integrate import trapezoid
from numpy import sqrt

def load_data(data_filepath):
    # read in targets
    target_path = os.path.join(data_filepath, 'target_music.csv')
    targets = pd.read_csv(target_path)
    target_dict = dict()
    for i in range(1, targets.shape[1]):
        target = targets.iloc[:, i].values
        target_dict[i] = target[~np.isnan(target)]

    # read in data
    data_path = os.path.join(data_filepath, 'music*.csv')
    file_paths = glob.glob(data_path)
    # 使用正则表达式从每个路径中提取数字
    def extract_number(path):
        match = re.search(r'\d+', path)
        if match:
            return int(match.group())
        return None  # 对于没有数字的路径，返回None
    # 对路径列表进行排序，排序键为路径中的数字
    file_paths = sorted(file_paths, key=extract_number)

    data_dict = dict()
    pattern = re.compile(r'music(\d+)\.csv')
    for path in file_paths:
        # get music id
        match = pattern.search(path)
        idx = int(match.group(1))
        data_dict[idx] = pd.read_csv(path).dropna()

    return target_dict, data_dict

def check_folder():
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

    return None

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

def data_after_align(target_dict, data_dict):
    # formulate a dictionary for observed data, aligned data and warpping functions
    obs_data_dict, aligned_data_dict, warp_func_dict = dict(), dict(), dict()
    for music_idx in range(1, 13):
        obs_data_ls, aligned_data_ls, warp_func_ls = align_data(target_dict, data_dict, music_idx=music_idx)
        obs_data_dict[music_idx] = obs_data_ls
        aligned_data_dict[music_idx] = aligned_data_ls
        warp_func_dict[music_idx] = warp_func_ls
    return obs_data_dict, aligned_data_dict, warp_func_dict


###################
# no alignment
###################
def check_folder_na():
    output_path = './output_na'
    path_ls = [
        output_path,
        os.path.join(output_path, 'dataframe_na'),
        os.path.join(output_path, 'plot_na'),
    ]
    for music_idx in range(1, 13):
        folder_name = f'music{music_idx}_na'
        path_ls.append(os.path.join(output_path, 'plot_na', folder_name))
    path_ls.append(os.path.join(output_path, 'plot_na', 'func_reg_na'))

    for path in path_ls:
        if not os.path.exists(path):
            os.makedirs(path)

    return None

def check_folder_compare():
    comopare_output_path = './compare'
    path_ls = [
        comopare_output_path
    ]

    for path in path_ls:
        if not os.path.exists(path):
            os.makedirs(path)

    return None

def l2norm(f1, f2, time):
    q1 = uf.f_to_srsf(f1, time)
    q2 = uf.f_to_srsf(f2, time)
    Dy = sqrt(trapezoid((q2 - q1) ** 2, time))
    return Dy

def get_emo(music_idx):
    if music_idx in [1, 2, 3]:
        return 'anger'
    elif music_idx in [4, 5, 6]:
        return 'happiness'
    elif music_idx in [7, 8, 9]:
        return 'sadness'
    elif music_idx in [10, 11, 12]:
        return 'tenderness'
    else:
        raise 'Out of music index range.'