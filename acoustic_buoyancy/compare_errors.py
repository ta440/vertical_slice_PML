'''
Compare errors with no damping, a sponge,
and the PML in the acoustic-buoyancy equations.
Compute errors using Firedrake error().

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
ref_sol = 'acoustic_buoyancy_runA_ref'
no_damp_sol = 'acoustic_buoyancy_runA'
PML_sol = 'PML_acoustic_buoyancy_runA_delta_0.1_gamma0_0'

# Field to compute the error in
field_name = 'u_z'

savename = f'compare_errors_runA_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/field_output.nc'
PML_results = f'{abspath(dirname(__file__))}/results/{PML_sol}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
no_damp_data = Dataset(no_damp_results, 'r')
PML_data = Dataset(PML_results, 'r')

figure_stem = f'{abspath(dirname(__file__))}/figures/'

# Arrays to store errors
no_damp_errs = []
PML_errs = []

# Time values:
time = data_file['time'][t_idx]

for i in np.arange(len(time));
    t = time[i]

    ref_field = extract_gusto_field(ref_data, field_name, time_idx=i)
    no_damp_field = extract_gusto_field(no_damp_data, field_name, time_idx=i)
    PML_field = extract_gusto_field(PML_data, field_name, time_idx=i)

    no_damp_error = errornorm(ref_field, no_damp_field)
    PML_error = errornorm(ref_field, PML_field)

    no_damp_errs.append(no_damp_error)
    PML_errs.append(PML_error)


no_damp_errs = np.asarray(no_damp_errs)
PML_errs = np.asarray(PML_errs)

plt.figure()
plt.plot(time, no_damp_errs, label='No damping')
plt.plot(time, PML_errs, label='PML')
plt.legend()

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()