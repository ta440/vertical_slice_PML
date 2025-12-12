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
ref_sol = 'acoustic_only_ref'
no_damp_sol = 'acoustic_only_no_damp'
PML_sol1 = 'PML_acoustic_only'
PML_sol2 = 'PML_acoustic_only_gamma0_0.5'

# Field to compute the error in
field_name = 'u_z'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}acoustic_only_compare_errors_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/field_output.nc'
PML_results1 = f'{abspath(dirname(__file__))}/results/{PML_sol1}/field_output.nc'
PML_results2 = f'{abspath(dirname(__file__))}/results/{PML_sol2}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
no_damp_data = Dataset(no_damp_results, 'r')
PML_data1 = Dataset(PML_results1, 'r')
PML_data2 = Dataset(PML_results2, 'r')

# Arrays to store errors
no_damp_errs = []
PML_errs1 = []
PML_errs2 = []

# Time values:
time = ref_data['time']

for i in np.arange(len(time)):
    t = time[i]

    ref_field = extract_gusto_field(ref_data, field_name, time_idx=i)
    no_damp_field = extract_gusto_field(no_damp_data, field_name, time_idx=i)
    PML_field1 = extract_gusto_field(PML_data1, field_name, time_idx=i)
    PML_field2 = extract_gusto_field(PML_data2, field_name, time_idx=i)

    # Crop fields to be outside of the PML
    ref_coords_X, ref_coords_Z = extract_gusto_coords(ref_data, field_name)
    ref_inds = np.where(ref_coords_Z <= z_max)
    ref_field = ref_field[ref_inds]

    test_coords_X, test_coords_Z = extract_gusto_coords(no_damp_data, field_name)
    test_inds = np.where(test_coords_Z <= z_max)
    no_damp_field = no_damp_field[test_inds]
    PML_field1 = PML_field1[test_inds]
    PML_field2 = PML_field2[test_inds]

    # Normalised L1 error
    #no_damp_error = np.sum(np.abs(ref_field - no_damp_field))/np.sum(np.abs(ref_field))
    #sponge_error = np.sum(np.abs(ref_field - sponge_field))/np.sum(np.abs(ref_field))
    #PML_error = np.sum(np.abs(ref_field - PML_field))/np.sum(np.abs(ref_field))

    # What about L2?
    no_damp_error = np.sqrt(np.sum((ref_field - no_damp_field)**2))/np.sum(np.abs(ref_field))
    PML_error1 = np.sqrt(np.sum((ref_field - PML_field1)**2))/np.sum(np.abs(ref_field))
    PML_error2 = np.sqrt(np.sum((ref_field - PML_field2)**2))/np.sum(np.abs(ref_field))

    no_damp_errs.append(no_damp_error)
    PML_errs1.append(PML_error1)
    PML_errs2.append(PML_error2)

no_damp_errs = np.asarray(no_damp_errs)
PML_errs1 = np.asarray(PML_errs1)
PML_errs2 = np.asarray(PML_errs2)

print(f'Final undamped error is {no_damp_errs[-1]}')
print(f'Final PML 1 error is {PML_errs1[-1]}')
print(f'Final PML 2 error is {PML_errs2[-1]}')

plt.figure()
plt.semilogy(time[:], no_damp_errs, label='No damping')
plt.semilogy(time[:], PML_errs1, label='PML')
plt.semilogy(time[:], PML_errs2, label='PML with gamma0 = 0.5')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Error in the lower domain')
plt.xlim([0,time[-1]])

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()