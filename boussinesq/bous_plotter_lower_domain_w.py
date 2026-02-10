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
    extract_gusto_coords, extract_gusto_field, reshape_gusto_data,
    add_colorbar_fig
)


#########################################

# Give the time to plot at:
t_idx = 50

# Specify if it's the acoustic or gravity wave test
#test_type = 'acoustic'
test_type = 'gravity_wave'

if test_type == 'acoustic':
    results_ref = 'bous_mount_acoustic_ref'
    results_no_damp = 'bous_mount_acoustic'
    results_sponge = 'bous_mount_sponge_acoustic_mudt_0.25'
    results_PML = 'PML_bous_mount_acoustic_gamma0_0.0'
    #results_PML = 'PML_bous_mount_acoustic_gamma0_0.1'
elif test_type == 'gravity_wave':
    results_ref = 'bous_mount_gravity_wave_ref_feb10'
    results_no_damp = 'bous_mount_gravity_wave'
    results_sponge = 'bous_mount_sponge_gravity_wave_mudt_0.25'
    results_PML = 'PML_bous_mount_gravity_wave_gamma0_0.0'

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
print(time)

colour_scheme = 'PiYG'
contour_method = 'contour'  # Need to use this method to show mountains!

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
#field_titles = ['u', 'w', 'p', 'u', 'w', 'p']

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
field_names = ['u_z', 'u_z', 'u_z', 'u_z']
field_titles = ['No damping', 'Sponge', 'PML', 'Reference']
field_labels = [r'$w$ (m s$^{-1}$)']

if test_type == 'gravity_wave':
    x_min = 40
    x_max = 80
    figsize=(12,8)
else:
    x_min=0
    x_max=100
    figsize=(12,5)

z_max = 18
zlims = [0, z_max]
xlims = [x_min, x_max]

fig, axarray = plt.subplots(2,2, figsize=figsize, sharey='all', sharex='all')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    print(i)
    
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
    inds = np.where((coords_Z < z_max) & (coords_X > x_min) & (coords_X < x_max))
    field_data_crop = field_data[inds]
    coords_X_crop = coords_X[inds]
    coords_Z_crop = coords_Z[inds]

    #if i==0:
    #    contours = tomplot_contours(field_data_crop)
    #    u_z_contour = contours
    #else:
    #    contours = u_z_contour

    if test_type == 'acoustic':
        contours = np.linspace(-0.25,0.25,11)
    elif test_type == 'gravity_wave':
        contours = np.linspace(-2,2,9)
        #contours = np.linspace(-4,4,9)

    cmap, lines = tomplot_cmap(contours, colour_scheme, extend_cmap='both')

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Z, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   

    tomplot_field_title(ax, f'{field_title} \n \n', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    
    ax.set_ylim(zlims)
    ax.set_xlim(xlims)
    ax.set_aspect('equal')

    if i == 0:
        ax.set_ylabel(r"$z$ (km)")
    elif i == 2:
        ax.set_ylabel(r"$z$ (km)")
        ax.set_xlabel(r"$x$ (km)")
    elif i == 3:
        ax.set_xlabel(r"$x$ (km)")
        add_colorbar_fig(
            fig, cf, r"$w$ (m s$^{-1}$)", location='right', cbar_labelpad=0.4
        )

t_val = np.round(time, 0)

savename = f'{figure_stem}bous_{test_type}_compare_lower_domain_w_only_t{t_val}.jpg'
plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
