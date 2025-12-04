'''

Plot the pressure and velocity fields at a single time

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
t_idx = 78

#results_dir = 'bous_mount_acoustic_wave'
#results_dir = 'bous_mount_acoustic_wave_hydro_balance'
results_dir = 'PML_bous_mount_gravity_wave_trapz_dt_10s_TT_10000'

plot_PML_vars = False

extra_name = ''

results_file_name = f'{abspath(dirname(__file__))}/results/{results_dir}/field_output.nc'
figure_stem = f'{abspath(dirname(__file__))}/figures/'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

time = data_file['time'][t_idx]

colour_scheme = 'PiYG'
field_label = r'$w$ (m s$^{-1}$)'
contour_method = 'contour'  # Need to use this method to show mountains!

if plot_PML_vars:
    field_names = ['q_u_x', 'q_u_z', 'q_p', 'q_b']
    field_titles = ['q_u', 'q_w', 'q_p', 'q_b']
else:
    field_names = ['u_x', 'u_z', 'p', 'b']
    field_titles = ['u', 'w', 'p', 'b']

fig, axarray = plt.subplots(2,2, figsize=(10,6), sharey='all', constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    data = extract_gusto_field(data_file, field_name, time_idx=t_idx)

    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Y = \
        reshape_gusto_data(data, coords_X, coords_Y)


    contours = tomplot_contours(field_data)

    if contours[0] == contours[-1]:
        contours = np.arange(0,1,0.1)

    print('\n')
    print(field_name)
    #print(contours)
    #print(np.where(contours==0.0)[0])

    if (contours[0] < 0) and (len(np.where(contours==0.0)[0]) > 0):
        cmap, lines = tomplot_cmap(contours, colour_scheme)#, remove_contour = 0.0)
    else:
        cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(ax, f'{field_title} \n', minmax=True, minmax_format='.4f', field_data=field_data)

    ax.set_aspect('equal')

t_val = np.round(time, 0)

if plot_PML_vars:
    savename = f'{figure_stem}{results_dir}_PML_vars_t{t_val}.jpg'
else:
    savename = f'{figure_stem}{results_dir}_t{t_val}.jpg'

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
