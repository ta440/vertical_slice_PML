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
    results_PML = 'PML_bous_mount_acoustic_gamma0_0.0'
elif test_type == 'gravity_wave':
    results_ref = 'bous_mount_gravity_wave_ref'
    results_no_damp = 'bous_mount_gravity_wave_trapz_10s_TT_10000s_vec_adv'
    results_PML = 'PML_bous_mount_gravity_wave_trapz_dt_10s_TT_10000_gamma0_0'

results_file_name_ref = f'{abspath(dirname(__file__))}/results/{results_ref}/field_output.nc'
results_file_name_no_damp = f'{abspath(dirname(__file__))}/results/{results_no_damp}/field_output.nc'
results_file_name_PML = f'{abspath(dirname(__file__))}/results/{results_PML}/field_output.nc'
figure_stem = f'{abspath(dirname(__file__))}/figures/'

data_file_ref = Dataset(results_file_name_ref, 'r')
data_file_no_damp = Dataset(results_file_name_no_damp, 'r')
data_file_PML = Dataset(results_file_name_PML, 'r')

set_tomplot_style()

time = data_file_ref['time'][t_idx]

colour_scheme = 'PiYG'
contour_method = 'contour'  # Need to use this method to show mountains!

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
#field_titles = ['u', 'w', 'p', 'u', 'w', 'p']

#field_names = ['u_x', 'u_z', 'p', 'u_x', 'u_z', 'p', 'u_x', 'u_z', 'p']
field_names = ['u_x', 'u_z', 'b_perturbation', 'u_x', 'u_z', 'b_perturbation', 'u_x', 'u_z', 'b_perturbation']
field_titles = [r"$u$", r"$w$", r"$b$ perturbation", r"$u$", r"$w$", r"$b$ perturbation", r"$u$", r"$w$", r"$b$ perturbation"]
field_labels = [r'$u$ (m s$^{-1}$)', r'$w$ (m s$^{-1}$)', r'$\Delta b$ (m s$^{-2}$)']

z_max = 18
zlims = [0, z_max]

fig, axarray = plt.subplots(3,3, figsize=(10,8), sharey='all', constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    if i < 3:
        data_file = data_file_no_damp
    elif i < 6:
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
        u_x_contour = contours
    elif i == 1:
        contours = tomplot_contours(field_data_crop)
        u_z_contour = contours
    elif i == 2:
        contours = tomplot_contours(field_data_crop)
        other_contour = contours
    elif (i%3) == 0:
        contours = u_x_contour
    elif (i%3) == 1:
        contours = u_z_contour
    else:
        contours = other_contour

    if contours[0] == contours[-1]:
        contours = np.arange(0,1,0.1)
    print(contours)

    cmap, lines = tomplot_cmap(contours, colour_scheme)

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Z, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )   

    if i < 3:
        tomplot_field_title(ax, f'{field_title} \n \n', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    elif i > 5:
        field_label = field_labels[int((i)%3)]
        add_colorbar_ax(ax, cf, field_label, location='bottom')
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    else:
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    ax.set_ylim(zlims)
    #ax.set_aspect('equal')

    if i ==0:
        ax.set_ylabel('No damping \n' r"$z$ (km)")
    elif i == 3:
        ax.set_ylabel('PML\n' r"$z$ (km)")
    elif i == 6:
        ax.set_ylabel('Reference\n' r"$z$ (km)")

t_val = np.round(time, 0)

savename = f'{figure_stem}bous_{test_type}_compare_lower_domain_t{t_val}.jpg'
plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()
