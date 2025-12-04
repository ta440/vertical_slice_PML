'''

Make a video by reading in data and using Tomplot
Plot the vertical velocity field

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

vanilla_results_dir = 'bous_mount_acoustic_wave_hydro_balance'
PML_results_dir = 'PML_bous_mount_acoustic_wave'
extra_name = 'bous_acoustic_'

field_name = 'u_z'
#field_name = 'q_u_z'
#field_name = 'p'
#field_name = 'q_p'

vanilla_results_file_name = f'{abspath(dirname(__file__))}/results/{vanilla_results_dir}/field_output.nc'
PML_results_file_name = f'{abspath(dirname(__file__))}/results/{PML_results_dir}/field_output.nc'
video_stem = f'{abspath(dirname(__file__))}/videos/'

savename = f'{extra_name}{field_name}'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
vanilla_data_file = Dataset(vanilla_results_file_name, 'r')
PML_data_file = Dataset(PML_results_file_name, 'r')

colour_scheme = 'PiYG'
field_label = r'$w$ (m s$^{-1}$)'
contour_method = 'contour'  # Need to use this method to show mountains!

# Contours for all plots:
#w_max = 2
#np.linspace(-w_max, w_max, 23)

# List of times to have on the video:
t_range = np.arange(0,100,1)

individual_plots = []

for i in np.arange(len(t_range)):
    t_val = t_range[i]
    print(t_val)

    fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharex='all', sharey='all', layout='constrained')
    (ax1, ax2) = ax

    # Plot undamped data
    vanilla_field_data = extract_gusto_field(vanilla_data_file, field_name, time_idx=t_val)
    coords_X, coords_Y = extract_gusto_coords(vanilla_data_file, field_name)
    vanilla_field_data, coords_X, coords_Y = \
        reshape_gusto_data(vanilla_field_data, coords_X, coords_Y)
    time = vanilla_data_file['time'][t_val]
    contours = tomplot_contours(vanilla_field_data)
    print(contours)
    if np.min(contours) == np.max(contours):
        contours = np.linspace(0,1,10)

    if (contours[0] < 0) and (len(np.where(contours==0.0)[0]) > 0):
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour = 0.0)
    else:
        cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf1, _ = plot_contoured_field(
        ax1, coords_X, coords_Y, vanilla_field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    # Plot PML data
    PML_field_data = extract_gusto_field(PML_data_file, field_name, time_idx=t_val)
    coords_X, coords_Y = extract_gusto_coords(PML_data_file, field_name)
    PML_field_data, coords_X, coords_Y = \
        reshape_gusto_data(PML_field_data, coords_X, coords_Y)
    cf2, _ = plot_contoured_field(
        ax2, coords_X, coords_Y, PML_field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    plt.colorbar(cf2, label=field_label)
    plt.suptitle(f'Compressible Boussinesq, orographic acoustic wave \n t={time:.1f} s \n')

    # Find max in upper half of domain:
    inds = np.where(coords_Y>10)
    vanilla_crop = vanilla_field_data[inds]
    PML_crop = PML_field_data[inds]

    tomplot_field_title(ax1, 'Vanilla \n' r"Max $w$ for $z >$ 10 km" '\n', minmax=True, field_data=vanilla_crop)
    tomplot_field_title(ax2, 'PML \n' r"Max $w$ for $z >$ 10 km" '\n', minmax=True, field_data=PML_crop)

    name = str(i) + '.png'
    plt.savefig(name, dpi=300)
    individual_plots.append(name)
    plt.close()


w = imageio.get_writer(f'{video_stem}{savename}.gif')

for filename in individual_plots:
    image = imageio.imread(filename)
    w.append_data(image)
w.close()

for filename in set(individual_plots):
    os.remove(filename)

print(f'saved video to {savename}')