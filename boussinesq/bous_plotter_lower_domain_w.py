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
t_idx = 100

# Specify if it's the acoustic or gravity wave test
test_type = 'acoustic'
#test_type = 'gravity_wave'

if test_type == 'acoustic':
    results_ref = 'bous_mount_acoustic_ref'
    results_no_damp = 'bous_mount_acoustic'
    results_sponge = 'PML_bous_mount_acoustic_gamma0_0.0'
    results_PML = 'PML_bous_mount_acoustic_gamma0_0.5'
elif test_type == 'gravity_wave':
    results_ref = 'bous_mount_gravity_wave_ref'
    results_no_damp = 'bous_mount_gravity_wave_trapz_10s_TT_10000s_vec_adv'
    results_PML = 'PML_bous_mount_gravity_wave_trapz_dt_10s_TT_10000_gamma0_0'

results_file_name_ref = f'{abspath(dirname(__file__))}/results/{results_ref}/field_output.nc'
results_file_name_no_damp = f'{abspath(dirname(__file__))}/results/{results_no_damp}/field_output.nc'
results_file_name_sponge = f'{abspath(dirname(__file__))}/results/{results_sponge}/field_output.nc'
results_file_name_PML = f'{abspath(dirname(__file__))}/results/{results_PML}/field_output.nc'
figure_stem = f'{abspath(dirname(__file__))}/figures/'

data_file_ref = Dataset(results_file_name_ref, 'r')
data_file_sponge = Dataset(results_file_name_sponge, 'r')
data_file_no_damp = Dataset(results_file_name_no_damp, 'r')
data_file_PML = Dataset(results_file_name_PML, 'r')

set_tomplot_style()

time = data_file_ref['time'][t_idx]

colour_scheme = 'PiYG'
contour_method = 'contour'  # Need to use this method to show mountains!

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
#field_titles = ['u', 'w', 'p', 'u', 'w', 'p']

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
field_names = ['u_z', 'u_z', 'u_z', 'u_z']
field_titles = [r"$w$", r"$w$", r"$w$", r"$w$"]
field_labels = [r"$w$ (m s$^{-1}$)"]

z_max = 18
zlims = [0, z_max]

fig, axarray = plt.subplots(1,4, figsize=(10,5), sharey='all', constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    if i == 0:
        data_file = data_file_no_damp
    elif i == 1:
        data_file = data_file_sponge
    elif i ==2:
        data_file = data_file_PML
    else:
        data_file = data_file_ref
    
    data = extract_gusto_field(data_file, field_name, time_idx=t_idx)
    
    coords_X, coords_Z = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Z = \
        reshape_gusto_data(data, coords_X, coords_Z)


    # Extract just values in the sponge layer
    inds = np.where(coords_Z < z_max)
    field_data_crop = field_data[inds]
    coords_X_crop = coords_X[inds]
    coords_Z_crop = coords_Z[inds]

    if i==0:
        contours = tomplot_contours(field_data_crop)
        u_z_contour = contours
    else:
        contours = u_z_contour

    cmap, lines = tomplot_cmap(contours, colour_scheme)

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Z, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   

    tomplot_field_title(ax, f'{field_title} \n \n', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    
    ax.set_ylim(zlims)
    #ax.set_aspect('equal')

    if i == 0:
        ax.set_ylabel('No damping \n' r"$z$ (km)")
    elif i == 1:
        ax.set_ylabel('Sponge \n' r"$z$ (km)")
    elif i == 2:
        ax.set_ylabel('PML \n' r"$z$ (km)")
    elif i == 3:
        ax.set_ylabel('Reference \n' r"$z$ (km)")

t_val = np.round(time, 0)

savename = f'{figure_stem}bous_{test_type}_compare_lower_domain_w_only_t{t_val}.jpg'
plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
