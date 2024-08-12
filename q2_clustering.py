

from utils import load_data, check_folder, data_after_align

# check wheter all the requred folders are exist or not
check_folder()

########################################################

# Load the data
target_dict, data_dict = load_data('./data/')
# Get data after alignemnt
obs_data_dict, aligned_data_dict, warp_func_dict = data_after_align(target_dict, data_dict)

