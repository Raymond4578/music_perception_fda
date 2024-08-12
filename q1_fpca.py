import numpy as np
import matplotlib.pyplot as plt
import skfda
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot

from utils import load_data, check_folder, data_after_align

# check wheter all the requred folders are exist or not
check_folder()

########################################################

# Load the data
target_dict, data_dict = load_data('./data/')
# Get data after alignemnt
obs_data_dict, aligned_data_dict, warp_func_dict = data_after_align(target_dict, data_dict)

music_idx = 1

t = np.linspace(0, 1, len(target_dict[music_idx]))
aligned_data_fd = skfda.FDataGrid(data_matrix=aligned_data_dict[music_idx], grid_points=t)
warp_func_fd = skfda.FDataGrid(data_matrix=warp_func_dict[music_idx], grid_points=t)
# Set a BSpline Basis
bspline_basis = BSplineBasis(n_basis=10)

# Do FPCA on aligned data
basis_aligned_data_fd = aligned_data_fd.to_basis(bspline_basis)
aligned_data_fpca = FPCA(n_components=2)
aligned_data_fpca.fit(basis_aligned_data_fd)
# 但是我这里还是先用这个explained variance ratio
aligned_data_pc_explained_ratio = aligned_data_fpca.explained_variance_ratio_
# Then generate the PC function plot
labels = ['Principle Component 1', 'Principle Component 2']
fig = plt.figure(figsize=(2 * 6, 1 * 4))  # 设置适当的尺寸
for i in range(1, 3):
    ax = fig.add_subplot(1, 2, i)
    function_values = aligned_data_fpca.components_(t)[i - 1, :]
    ax.plot(t, function_values)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_ylim(-2.1, 1.7)
    ax.set_title(f'Principle Component {i} ({aligned_data_pc_explained_ratio[i - 1]:.2%})')
plt.tight_layout()
plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_aligned_data_pc.png', dpi=300)
plt.close()
# Then generate the Mean Variation plot (mean function +- PC function)
aligned_data_fpca_mean_pc_plot = FPCAPlot(
    basis_aligned_data_fd.mean(),
    aligned_data_fpca.components_,
    fig=plt.figure(figsize=(2 * 6, 1 * 4)),
    n_rows=1
).plot()
i = 0
for ax, new_title in zip(aligned_data_fpca_mean_pc_plot.get_axes(), labels):
    ax.set_title(f'{labels[i]} ({aligned_data_pc_explained_ratio[i]:.2%})')
    ax.set_ylim(2.2, 6.4)
    i += 1
plt.tight_layout()
plt.savefig(f'./output/plot/music{music_idx}/music{music_idx}_aligned_data_pc_var.png', dpi=300)
plt.close()