import numpy as np
import matplotlib.pyplot as plt
import skfda
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot

from utils import load_data, check_folder_na, data_after_align

# check wheter all the requred folders are exist or not
check_folder_na()

########################################################

# Load the data
target_dict, data_dict = load_data('../data/')
# Get data after alignemnt
obs_data_dict, aligned_data_dict, warp_func_dict = data_after_align(target_dict, data_dict) # Only use obs_data_dict

for music_idx in range(1, 13):
    t = np.linspace(0, 1, len(target_dict[music_idx]))
    obs_data_fd = skfda.FDataGrid(data_matrix=obs_data_dict[music_idx], grid_points=t)
    # Set a BSpline Basis
    bspline_basis = BSplineBasis(n_basis=10)

    # Do FPCA on observed data
    basis_obs_data_fd = obs_data_fd.to_basis(bspline_basis)
    obs_data_fpca = FPCA(n_components=2)
    obs_data_fpca.fit(basis_obs_data_fd)
    # 但是我这里还是先用这个explained variance ratio
    obs_data_pc_explained_ratio = obs_data_fpca.explained_variance_ratio_
    # Then generate the PC function plot
    labels = ['Principle Component 1', 'Principle Component 2']
    fig = plt.figure(figsize=(2 * 6, 1 * 4))  # Set the figure size
    for i in range(1, 3):
        ax = fig.add_subplot(1, 2, i)
        function_values = obs_data_fpca.components_(t)[i - 1, :]
        ax.plot(t, function_values)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_ylim(-2.1, 1.7)
        ax.set_title(f'Principle Component {i} ({obs_data_pc_explained_ratio[i - 1]:.2%})')
    plt.tight_layout()
    plt.savefig(f'./output_na/plot_na/music{music_idx}_na/music{music_idx}_obs_data_pc_na.png', dpi=300)
    plt.close()
    # Then generate the Mean Variation plot (mean function +- PC function)
    obs_data_fpca_mean_pc_plot = FPCAPlot(
        basis_obs_data_fd.mean(),
        obs_data_fpca.components_,
        fig=plt.figure(figsize=(2 * 6, 1 * 4)),
        n_rows=1
    ).plot()
    i = 0
    for ax, new_title in zip(obs_data_fpca_mean_pc_plot.get_axes(), labels):
        ax.set_title(f'{labels[i]} ({obs_data_pc_explained_ratio[i]:.2%})')
        ax.set_ylim(2.2, 6.4)
        i += 1
    plt.tight_layout()
    plt.savefig(f'./output_na/plot_na/music{music_idx}_na/music{music_idx}_aligned_data_pc_var_na.png', dpi=300)
    plt.close()
