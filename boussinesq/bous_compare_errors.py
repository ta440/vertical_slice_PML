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
#test = 'acoustic'
test = 'gravity_wave'

####################################

if test == 'acoustic':
    ref_sol = 'bous_mount_acoustic_ref'
    no_damp_sol = 'bous_mount_acoustic'
    sponge_sol = 'bous_mount_sponge_acoustic_mudt_0.25'
    #PML_sol = 'PML_bous_mount_acoustic_gamma0_0.0_test2'
    PML_sol = 'PML_bous_mount_acoustic_gamma0_0.0'
    #PML_sol = 'PML_bous_mount_acoustic_gamma0_0.1'
    #PML_sol = 'PML_bous_mount_acoustic_gamma0_0.2_test2'
    #sponge_sol = 'PML_bous_mount_acoustic_gamma0_0.5'
    #PML_sol = 'PML_bous_mount_acoustic_gamma0_0.0_extra_transport_terms'
    extra_name = '_acoustic'
else:
    ref_sol = 'bous_mount_gravity_wave_ref_feb10'
    no_damp_sol = 'bous_mount_gravity_wave'
    sponge_sol = 'bous_mount_sponge_gravity_wave_mudt_0.25'
    PML_sol = 'PML_bous_mount_gravity_wave_gamma0_0.0'
    extra_name = '_gravity_wave'

# Field to compute the error in
field_name = 'u_x'

# Maximum height to compute errors
z_min = 0
z_max = 10
zlims = [z_min,z_max]

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
    ref_inds = np.where((ref_coords_Z <= z_max) & (ref_coords_Z >= z_min))
    ref_field = ref_field[ref_inds]

    test_coords_X, test_coords_Z = extract_gusto_coords(no_damp_data, field_name)
    test_inds = np.where((test_coords_Z <= z_max) & (test_coords_Z >= z_min))
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

print(f'Time averaged error, undamped, is', np.sum(no_damp_errs)/len(time))
print(f'Time averaged error, sponge, is', np.sum(sponge_errs)/len(time))
print(f'Time averaged error, PML, is', np.sum(PML_errs)/len(time))

t_init = 0
t_end = 100


plt.figure()
plt.semilogy(time[t_init:t_end], no_damp_errs[t_init:t_end], label='No damping')
plt.semilogy(time[t_init:t_end], sponge_errs[t_init:t_end], label='Sponge')
plt.semilogy(time[t_init:t_end], PML_errs[t_init:t_end], label='PML')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Error in the lower domain')
plt.xlim([time[t_init], time[t_end]])

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()