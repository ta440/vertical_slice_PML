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

results_dir = 'acoustic_test'
#results_dir = 'PML_melvin_bous_acoustic_wave_rk4_hydro_bal_sigma1'
extra_name = ''

field_name = 'u_z'
#field_name = 'q_u_z'
#field_name = 'p'
#field_name = 'q_p'

results_file_name = f'{abspath(dirname(__file__))}/results/{results_dir}/field_output.nc'
video_stem = f'{abspath(dirname(__file__))}/videos/'
savename = f'{results_dir}{extra_name}_{field_name}'

# Things that are likely the same for all plots --------------------------------
set_tomplot_style()
data_file = Dataset(results_file_name, 'r')

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
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex='all', sharey='all')
    field_data = extract_gusto_field(data_file, field_name, time_idx=t_val)
    coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Y = \
        reshape_gusto_data(field_data, coords_X, coords_Y)
    time = data_file['time'][t_val]
    contours = tomplot_contours(field_data)
    print(contours)
    if np.min(contours) == np.max(contours):
        contours = np.linspace(0,1,10)
    cmap, lines = tomplot_cmap(contours, colour_scheme)

    # Plot data ----------------------------------------------------------------
    cf, _ = plot_contoured_field(
        ax, coords_X, coords_Y, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines
    )

    add_colorbar_ax(ax, cf, field_label, location='bottom')
    tomplot_field_title(ax, f't={time:.1f} s \n', minmax=True, field_data=field_data)

    name = str(i) + '.png'
    plt.savefig(name, dpi=300, bbox_inches='tight')
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