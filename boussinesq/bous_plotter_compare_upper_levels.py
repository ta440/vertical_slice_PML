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
t_idx = 80

results_orig = 'bous_mount_acoustic_wave_hydro_balance'
results_PML = 'PML_bous_mount_acoustic_wave'
extra_name = ''

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

y_low = 10
ylims = [y_low,20]

fig, axarray = plt.subplots(2,3, figsize=(10,6), sharey='all', constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    if i < 3:
        data_file = data_file_orig
    else:
        data_file = data_file_PML
    
    data = extract_gusto_field(data_file, field_name, time_idx=t_idx)
    
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Y = \
        reshape_gusto_data(data, coords_X, coords_Y)


    # Extract just values in the sponge layer
    inds = np.where(coords_Y>y_low)
    field_data_crop = field_data[inds]
    coords_X_crop = coords_X[inds]
    coords_Y_crop = coords_Y[inds]

    if i==0:
        contours = tomplot_contours(field_data_crop)
        u_x_contour = contours
    elif i == 1:
        contours = tomplot_contours(field_data_crop)
        u_z_contour = contours
    elif i == 2:
        contours = tomplot_contours(field_data_crop)
        p_contour = contours
    elif i == 3:
        contours = u_x_contour
    elif i == 4:
        contours = u_z_contour
    elif i == 5:
        contours = p_contour

    print(contours)

    cmap, lines = tomplot_cmap(contours, colour_scheme)

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   

    if i < 3:
        tomplot_field_title(ax, f'{field_title} \n \n', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    else:
        add_colorbar_ax(ax, cf, field_label, location='bottom')
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    ax.set_ylim(ylims)
    #ax.set_aspect('equal')

    if i ==0:
        ax.set_ylabel('No PML\n' r"$z$ (km)")
    if i == 3:
        ax.set_ylabel('With the PML\n' r"$z$ (km)")

t_val = np.round(time, 0)

savename = f'{figure_stem}bous_compare_upper_domain_t{t_val}{extra_name}.jpg'
plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
