'''
Compare lower domains with no damping, a sponge,
and the PML in the acoustic-buoyancy equations.

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
from firedrake import errornorm
####################################
# Reference solution:
ref_sol = 'acoustic_only_ref'
no_damp_sol = 'acoustic_only_no_damp'
PML_sol1 = 'PML_acoustic_only'
PML_sol2 = 'PML_acoustic_only_gamma0_0.5'

# Time index:
t_idx = 50

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]
z_ticks = [0,5,10,15,18]

figure_stem = f'{abspath(dirname(__file__))}/figures/'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/field_output.nc'
PML1_results = f'{abspath(dirname(__file__))}/results/{PML_sol1}/field_output.nc'
PML2_results = f'{abspath(dirname(__file__))}/results/{PML_sol2}/field_output.nc'

data_file_ref = Dataset(ref_results, 'r')
data_file_no_damp = Dataset(no_damp_results, 'r')
data_file_PML1 = Dataset(PML1_results, 'r')
data_file_PML2 = Dataset(PML2_results, 'r')

time = data_file_ref['time'][t_idx]

# Field to compute the error in
field_names = ['u_z', 'p', 'u_z', 'p', 'u_z', 'p', 'u_z', 'p']
field_titles = ['w', 'p', 'w', 'p', 'w', 'p', 'w', 'p']
contour_method = 'contour'  # Need to use this method to show mountains!

colour_scheme = 'PiYG'
field_label_w = r'$w$ (m s$^{-1}$)'
field_label_p = r'$p$ (hPa)'

fig, axarray = plt.subplots(4,2, figsize=(6,10), sharey='all', sharex=True, constrained_layout='True')

for i, (ax, field_name, field_title) in \
        enumerate(zip(axarray.flatten(), field_names, field_titles)):
    
    if i < 2:
        data_file = data_file_no_damp
    elif i < 4:
        data_file = data_file_PML1
    elif i < 6:
        data_file = data_file_PML2
    else:
        data_file = data_file_ref

    print(i)
    
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
        w_contours = contours
    elif i == 1:
        contours = tomplot_contours(field_data_crop)
        p_contours = contours
    elif (i % 2) == 0:
        contours = w_contours
    else:
        contours = p_contours

    print(contours)

    if (contours[0] < 0) and (len(np.where(contours==0.0)[0]) > 0):
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour = 0.0)
    else:
        cmap, lines = tomplot_cmap(contours, colour_scheme)

    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Z, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    ) 

    if i == 6:  
        add_colorbar_ax(ax, cf, field_label_w, location='bottom')
        ax.set_xlabel(r'$x$ (km)')
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    elif i == 7:
        add_colorbar_ax(ax, cf, field_label_p, location='bottom')
        ax.set_xlabel(r'$x$ (km)')
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    elif i < 2:
        tomplot_field_title(ax, f'{field_title} \n', minmax=True, minmax_format='.4f', field_data=field_data_crop)
    else:
        tomplot_field_title(ax, f'', minmax=True, minmax_format='.4f', field_data=field_data_crop)

    ax.set_ylim(zlims)
    ax.set_yticks(z_ticks)
    ax.set_aspect('equal')

    if i ==0:
        ax.set_ylabel('No damping \n' r"$z$ (km)")
    elif i == 2:
        ax.set_ylabel('PML 1 \n' r"$z$ (km)")
    elif i == 4:
        ax.set_ylabel('PML 2 \n' r"$z$ (km)")
    elif i == 6:
        ax.set_ylabel('Reference \n' r"$z$ (km)")

t_val = np.round(time, 0)

savename = f'{figure_stem}acoustic_only_compare_lower_domains_{field_name}_t{t_val}.jpg'

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()