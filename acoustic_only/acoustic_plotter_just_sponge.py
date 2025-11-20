'''

Plot the pressure and velocity fields at a single time

Just look inside the sponge layer!
Compare with and without PML

'''

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import matplotlib
import matplotlib.colors as colors
import cartopy.crs as ccrs
import imageio
import os
from os.path import abspath, dirname
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, tomplot_contours,
    extract_gusto_coords, extract_gusto_field, reshape_gusto_data
)


#########################################

# Give the time to plot at:
t_idx = 30

results_orig = 'acoustic_test'

#results_PML = 'PML_acoustic_default'
#extra_name = ''

#results_PML = 'PML_acoustic_tol_1e-4'
#extra_name = 'tol_1e-4'

#results_PML = 'PML_acoustic_deltaf_0.1'
#extra_name = 'deltaf_0.1'

results_PML = 'PML_acoustic_A6'
extra_name = 'A6'

results_file_name_orig = f'{abspath(dirname(__file__))}/results/{results_orig}/field_output.nc'
results_file_name_PML = f'{abspath(dirname(__file__))}/results/{results_PML}/field_output.nc'
figure_stem = f'{abspath(dirname(__file__))}/figures/'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file_orig = Dataset(results_file_name_orig, 'r')
data_file_PML = Dataset(results_file_name_PML, 'r')

time = data_file_orig['time'][t_idx]

colour_scheme = 'PiYG'
field_label = r'$w$ (m s$^{-1}$)'
contour_method = 'contour'  # Need to use this method to show mountains!


field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
field_titles = ['u', 'w', 'p', 'u', 'w', 'p']

ylims = [18,20]

fig, axarray = plt.subplots(2,3, figsize=(10,3), sharey='all', constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    if i < 3:
        data_file = data_file_orig
    else:
        data_file = data_file_PML
    
    data = extract_gusto_field(data_file, field_name, time_idx=t_idx)
    
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    print(coords_Y)
    field_data, coords_X, coords_Y = \
        reshape_gusto_data(data, coords_X, coords_Y)

    contours = tomplot_contours(field_data)

    if contours[0] == contours[-1]:
        contours = np.arange(0,1,0.1)

    print(contours)

    if contours[0] < 0:
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour = 0.0)
    else:
        cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   


    # Extract just values in the sponge layer
    inds = np.where(coords_Y>18)
    field_data_cropped = field_data[inds]
    if i < 3:
        tomplot_field_title(ax, f'{field_title} \n \n', minmax=True, minmax_format='.4f', field_data=field_data_cropped)
    else:
        add_colorbar_ax(ax, cf, field_label, location='bottom')
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_cropped)
    ax.set_ylim(ylims)
    ax.set_aspect('equal')

t_val = np.round(time, 0)

savename = f'{figure_stem}acoustic_only_compare_in_sponge_t{t_val}{extra_name}.jpg'
plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
