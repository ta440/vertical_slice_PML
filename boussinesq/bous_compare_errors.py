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

# Choose acoustic or gravity wave setting
test = 'acoustic'
# test = 'gravity_wave'

####################################

if test == 'acoustic':
    ref_sol = 'bous_mount_acoustic_ref'
    no_damp_sol = 'bous_mount_acoustic'
    sponge_sol = 'PML_bous_mount_acoustic_gamma0_0.0'
    PML_sol = 'PML_bous_mount_acoustic_gamma0_0.5'
    extra_name = '_acoustic'
else:
    ref_sol = 'bous_mount_gravity_wave_ref'
    no_damp_sol = 'bous_mount_gravity_wave_trapz_10s_TT_10000s_vec_adv'
    sponge_sol = 'bous_mount_gravity_wave_trapz_10s_TT_10000s_vec_adv'
    PML_sol = 'PML_bous_mount_gravity_wave_trapz_dt_10s_TT_10000_gamma0_0'
    extra_name = '_gravity_wave'

# Field to compute the error in
field_name = 'u_z'

# Maximum height to compute errors
z_max = 1
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}bous_compare_errors{extra_name}_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/field_output.nc'
sponge_results = f'{abspath(dirname(__file__))}/results/{sponge_sol}/field_output.nc'
PML_results = f'{abspath(dirname(__file__))}/results/{PML_sol}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
no_damp_data = Dataset(no_damp_results, 'r')
sponge_data = Dataset(sponge_results, 'r')
PML_data = Dataset(PML_results, 'r')

# Arrays to store errors
no_damp_errs = []
PML_errs = []
sponge_errs = []

# Time values:
time = ref_data['time']

for i in np.arange(len(time)):
    
    ref_field = extract_gusto_field(ref_data, field_name, time_idx=i)
    no_damp_field = extract_gusto_field(no_damp_data, field_name, time_idx=i)
    PML_field = extract_gusto_field(PML_data, field_name, time_idx=i)
    sponge_field = extract_gusto_field(sponge_data, field_name, time_idx=i)

    # Crop fields to be outside of the PML
    ref_coords_X, ref_coords_Z = extract_gusto_coords(ref_data, field_name)
    ref_inds = np.where(ref_coords_Z <= z_max)
    ref_field = ref_field[ref_inds]

    test_coords_X, test_coords_Z = extract_gusto_coords(no_damp_data, field_name)
    test_inds = np.where(test_coords_Z < z_max)
    no_damp_field = no_damp_field[test_inds]
    sponge_field = sponge_field[test_inds]
    PML_field = PML_field[test_inds]

    # Normalised L1 error
    #no_damp_error = np.sum(np.abs(ref_field - no_damp_field))/np.sum(np.abs(ref_field))
    #sponge_error = np.sum(np.abs(ref_field - sponge_field))/np.sum(np.abs(ref_field))
    #PML_error = np.sum(np.abs(ref_field - PML_field))/np.sum(np.abs(ref_field))

    # What about L2?
    no_damp_error = np.sqrt(np.sum((ref_field - no_damp_field)**2))/np.sum(np.abs(ref_field))
    sponge_error = np.sqrt(np.sum((ref_field - sponge_field)**2))/np.sum(np.abs(ref_field))
    PML_error = np.sqrt(np.sum((ref_field - PML_field)**2))/np.sum(np.abs(ref_field))

    no_damp_errs.append(no_damp_error)
    sponge_errs.append(sponge_error)
    PML_errs.append(PML_error)

no_damp_errs = np.asarray(no_damp_errs)
sponge_errs = np.asarray(sponge_errs)
PML_errs = np.asarray(PML_errs)

print('\n')
print(f'Initial undamped error is {no_damp_errs[0]}')
print(f'Initial sponge error is {sponge_errs[0]}')
print(f'Initial PML error is {PML_errs[0]} \n')

print(f'Final undamped error is {no_damp_errs[-1]}')
print(f'Final sponge error is {sponge_errs[-1]}')
print(f'Final PML error is {PML_errs[-1]} \n')

plt.figure()
plt.semilogy(time[:], no_damp_errs, label='No damping')
plt.semilogy(time[:], sponge_errs, label='Sponge')
plt.semilogy(time[:], PML_errs, label='PML')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Error in the lower domain')
plt.xlim([0,time[-1]])

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()