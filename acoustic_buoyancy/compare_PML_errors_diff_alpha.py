'''
Compare errors with different 
values of gamma0 for the PML.

'''
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import matplotlib
import matplotlib.colors as colors
import cartopy.crs as ccrs
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
ref_sol = 'acoustic_buoyancy_ref'

gamma = 0.25

alpha1 = 0
alpha2 = 0.025
alpha3 = 0.05
alpha4 = 0.075
alpha5 = 0.1


# Different gamma solutions
PML_sol1 = f'PML_acoustic_buoyancy_alpha_{alpha1}_gamma0_{gamma}'
PML_sol2 = f'PML_acoustic_buoyancy_alpha_{alpha2}_gamma0_{gamma}'
PML_sol3 = f'PML_acoustic_buoyancy_gamma0_{gamma}'
PML_sol4 = f'PML_acoustic_buoyancy_alpha_{alpha4}_gamma0_{gamma}'
PML_sol5 = f'PML_acoustic_buoyancy_alpha_{alpha5}_gamma0_{gamma}'

# Undamped error for reference
undamped_sol = 'acoustic_buoyancy_no_damp'

# Field to compute the error in
field_name = 'u_z'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}acoustic_buoyancy_compare_PML_errors_diff_alpha_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
PML_results1 = f'{abspath(dirname(__file__))}/results/{PML_sol1}/field_output.nc'
PML_results2 = f'{abspath(dirname(__file__))}/results/{PML_sol2}/field_output.nc'
PML_results3 = f'{abspath(dirname(__file__))}/results/{PML_sol3}/field_output.nc'
PML_results4 = f'{abspath(dirname(__file__))}/results/{PML_sol4}/field_output.nc'
PML_results5 = f'{abspath(dirname(__file__))}/results/{PML_sol5}/field_output.nc'
undamped_results = f'{abspath(dirname(__file__))}/results/{undamped_sol}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
PML_data1 = Dataset(PML_results1, 'r')
PML_data2 = Dataset(PML_results2, 'r')
PML_data3 = Dataset(PML_results3, 'r')
PML_data4 = Dataset(PML_results4, 'r')
PML_data5 = Dataset(PML_results5, 'r')
undamped_data = Dataset(undamped_results, 'r')

# Arrays to store errors
PML_errs1 = []
PML_errs2 = []
PML_errs3 = []
PML_errs4 = []
PML_errs5 = []
undamped_errs = []

# Time values:
time = ref_data['time']

for i in np.arange(len(time)):
    t = time[i]

    ref_field = extract_gusto_field(ref_data, field_name, time_idx=i)
    PML_field1 = extract_gusto_field(PML_data1, field_name, time_idx=i)
    PML_field2 = extract_gusto_field(PML_data2, field_name, time_idx=i)
    PML_field3 = extract_gusto_field(PML_data3, field_name, time_idx=i)
    PML_field4 = extract_gusto_field(PML_data4, field_name, time_idx=i)
    PML_field5 = extract_gusto_field(PML_data5, field_name, time_idx=i)
    undamped_field = extract_gusto_field(undamped_data, field_name, time_idx=i)

    # Crop fields to be outside of the PML
    ref_coords_X, ref_coords_Z = extract_gusto_coords(ref_data, field_name)
    ref_inds = np.where(ref_coords_Z <= z_max)
    ref_field = ref_field[ref_inds]

    test_coords_X, test_coords_Z = extract_gusto_coords(PML_data1, field_name)
    test_inds = np.where(test_coords_Z <= z_max)
    PML_field1 = PML_field1[test_inds]
    PML_field2 = PML_field2[test_inds]
    PML_field3 = PML_field3[test_inds]
    PML_field4 = PML_field4[test_inds]
    PML_field5 = PML_field5[test_inds]
    undamped_field = undamped_field[test_inds]

    # Normalised L1 error
    #no_damp_error = np.sum(np.abs(ref_field - no_damp_field))/np.sum(np.abs(ref_field))
    #sponge_error = np.sum(np.abs(ref_field - sponge_field))/np.sum(np.abs(ref_field))
    #PML_error = np.sum(np.abs(ref_field - PML_field))/np.sum(np.abs(ref_field))

    # What about L2?
    PML_error1 = np.sqrt(np.sum((ref_field - PML_field1)**2))/np.sum(np.abs(ref_field))
    PML_error2 = np.sqrt(np.sum((ref_field - PML_field2)**2))/np.sum(np.abs(ref_field))
    PML_error3 = np.sqrt(np.sum((ref_field - PML_field3)**2))/np.sum(np.abs(ref_field))
    PML_error4 = np.sqrt(np.sum((ref_field - PML_field4)**2))/np.sum(np.abs(ref_field))
    PML_error5 = np.sqrt(np.sum((ref_field - PML_field5)**2))/np.sum(np.abs(ref_field))
    undamped_error = np.sqrt(np.sum((ref_field - undamped_field)**2))/np.sum(np.abs(ref_field))

    PML_errs1.append(PML_error1)
    PML_errs2.append(PML_error2)
    PML_errs3.append(PML_error3)
    PML_errs4.append(PML_error4)
    PML_errs5.append(PML_error5)
    undamped_errs.append(undamped_error)

PML_errs1 = np.asarray(PML_errs1)
PML_errs2 = np.asarray(PML_errs2)
PML_errs3 = np.asarray(PML_errs3)
PML_errs4 = np.asarray(PML_errs4)
PML_errs5 = np.asarray(PML_errs5)
undamped_errs = np.asarray(undamped_errs)

print(f'Final PML 1 error is {PML_errs1[-1]}')
print(f'Final PML 2 error is {PML_errs2[-1]}')
print(f'Final PML 3 error is {PML_errs3[-1]}')
print(f'Final PML 4 error is {PML_errs4[-1]}')
print(f'Final PML 5 error is {PML_errs5[-1]}')
print(f'Final undamped error is {undamped_errs[-1]}')

t_min = 20

plt.figure()
plt.semilogy(time[t_min:], PML_errs1[t_min:], label=r'$\alpha_F$ = ' f'{alpha1}')
plt.semilogy(time[t_min:], PML_errs2[t_min:], label=r'$\alpha_F$ = ' f'{alpha2}')
plt.semilogy(time[t_min:], PML_errs3[t_min:], label=r'$\alpha_F$ = ' f'{alpha3}')
plt.semilogy(time[t_min:], PML_errs4[t_min:], label=r'$\alpha_F$ = ' f'{alpha4}')
plt.semilogy(time[t_min:], PML_errs5[t_min:], label=r'$\alpha_F$ = ' f'{alpha5}')
# Additionally plot the undamped error?
plt.semilogy(time[t_min:], undamped_errs[t_min:], label='No damping')

plt.legend(loc='center right', prop={'size': 10}, bbox_to_anchor=(1.3, 0.5))
plt.xlabel('Time (s)', size=14)
plt.ylabel('Error in the lower domain', size=14)
plt.xlim([t_min,time[-1]])
plt.ylim([2e-6,1e-1])
plt.xticks([25,50,75,100])
plt.grid()


plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()