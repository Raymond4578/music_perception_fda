import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data

output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

music_idx = 3

# Load the data
target_dict, data_dict = load_data('../data/')
t = np.arange(8, 8 + 2 * len(target_dict[music_idx]), 2)

##########################################################################################
# illustration for target and observed responses with pointwise mean function in section 2
##########################################################################################

target = target_dict[music_idx]
obs_data_df = data_dict[music_idx]
# Drop first two column about idx and ID
obs_data_df.drop(obs_data_df.columns[:2], axis=1, inplace=True)
obs_data = np.array(obs_data_df).astype(float)
obs_data_mean = np.mean(obs_data, axis=0)

# Draw the plot
plt.figure(figsize=(8, 3.5))
for i, response in enumerate(obs_data):
    # Draw all the responses with dashed line
    if i == 0:
        plt.plot(t, response, linewidth=0.2, color='black', linestyle='--', label="Ovservations")
    else:
        plt.plot(t, response, linewidth=0.2, color='black', linestyle='--')
plt.plot(t, target, linewidth=2, color='blue', linestyle='-', label="Target") # target
plt.plot(t, obs_data_mean, linewidth=2, color='red', linestyle='-', label="Pointwise Mean") # pointwise mean function
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.xlabel('$t$')
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()
plt.savefig('./output/showData.png', dpi=300)
plt.close('all')

