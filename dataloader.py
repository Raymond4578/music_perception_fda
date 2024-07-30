import os
import re
import glob
import numpy as np
import pandas as pd

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