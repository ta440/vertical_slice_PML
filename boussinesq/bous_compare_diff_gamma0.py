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
ref_sol = 'bous_mount_acoustic_ref'

case_type = 'pert_hydro_sin2_acoustic'

gamma1 = 0.0
gamma2 = 0.1
gamma3 = 0.25
gamma4 = 0.5
gamma5 = 1.0

PML_sol1 = f'PML_bous_mount_{case_type}_gamma0_{gamma1}'
PML_sol2 = f'PML_bous_mount_{case_type}_gamma0_{gamma2}'
PML_sol3 = f'PML_bous_mount_{case_type}_gamma0_{gamma3}'
PML_sol4 = f'PML_bous_mount_{case_type}_gamma0_{gamma4}'
PML_sol5 = f'PML_bous_mount_{case_type}_gamma0_{gamma5}'

# Field to compute the error in
field_name = 'u_z'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}bous_compare_PML_errors_{case_type}_diff_gamma0_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
PML_results1 = f'{abspath(dirname(__file__))}/results/{PML_sol1}/field_output.nc'
PML_results2 = f'{abspath(dirname(__file__))}/results/{PML_sol2}/field_output.nc'
PML_results3 = f'{abspath(dirname(__file__))}/results/{PML_sol3}/field_output.nc'
PML_results4 = f'{abspath(dirname(__file__))}/results/{PML_sol4}/field_output.nc'
PML_results5 = f'{abspath(dirname(__file__))}/results/{PML_sol5}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
PML_data1 = Dataset(PML_results1, 'r')
PML_data2 = Dataset(PML_results2, 'r')
PML_data3 = Dataset(PML_results3, 'r')
PML_data4 = Dataset(PML_results4, 'r')
PML_data5 = Dataset(PML_results5, 'r')

# Arrays to store errors
PML_errs1 = []
PML_errs2 = []
PML_errs3 = []
PML_errs4 = []
PML_errs5 = []

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

    PML_errs1.append(PML_error1)
    PML_errs2.append(PML_error2)
    PML_errs3.append(PML_error3)
    PML_errs4.append(PML_error4)
    PML_errs5.append(PML_error5)

PML_errs1 = np.asarray(PML_errs1)
PML_errs2 = np.asarray(PML_errs2)
PML_errs3 = np.asarray(PML_errs3)
PML_errs4 = np.asarray(PML_errs4)
PML_errs5 = np.asarray(PML_errs5)

print(f'Final PML 1 error is {PML_errs1[-1]}')
print(f'Final PML 2 error is {PML_errs2[-1]}')
print(f'Final PML 3 error is {PML_errs3[-1]}')
print(f'Final PML 4 error is {PML_errs4[-1]}')
print(f'Final PML 5 error is {PML_errs5[-1]}')

t_min = 50

plt.figure()
plt.semilogy(time[t_min:], PML_errs1[t_min:], label=r'$\gamma_0$ = ' f'{gamma1}')
plt.semilogy(time[t_min:], PML_errs2[t_min:], label=r'$\gamma_0$ = ' f'{gamma2}')
plt.semilogy(time[t_min:], PML_errs3[t_min:], label=r'$\gamma_0$ = ' f'{gamma3}')
plt.semilogy(time[t_min:], PML_errs4[t_min:], label=r'$\gamma_0$ = ' f'{gamma4}')
plt.semilogy(time[t_min:], PML_errs5[t_min:], label=r'$\gamma_0$ = ' f'{gamma5}')
plt.legend(loc='center right', prop={'size': 10}, bbox_to_anchor=(1.3, 0.5))
plt.xlabel('Time (s)', size=14)
plt.ylabel('Error in the lower domain', size=14)
plt.xlim([t_min,time[-1]])
plt.ylim([1e-4,2e-3])

plt.grid()

# What about some black lines to show the time that the wave hits the model top?
# The acoustic wave of c=350 m/s travels 10 km in 28.57 s
y_lim_range = np.logspace(-6, -3, num=20)
#plt.plot(np.ones_like(y_lim_range)*28.57, y_lim_range, c='k', linewidth=0.5)
#plt.plot(np.ones_like(y_lim_range)*28.57*2, y_lim_range, c='k', linewidth=0.5)
#plt.plot(np.ones_like(y_lim_range)*28.57*3, y_lim_range, c='k', linewidth=0.5)


plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()