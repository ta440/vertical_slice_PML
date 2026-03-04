'''
Compare errors with different strengths of sponge layer.

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

case_type = 'runA'

mudt1 = 0.1
mudt2 = 0.25
mudt3 = 0.5
mudt4 = 0.75
mudt5 = 1

sponge_sol1 = f'acoustic_buoyancy_sponge_{case_type}_mudt_{mudt1}'
sponge_sol2 = f'acoustic_buoyancy_sponge_{case_type}_mudt_{mudt2}'
sponge_sol3 = f'acoustic_buoyancy_sponge_{case_type}_mudt_{mudt3}'
sponge_sol4 = f'acoustic_buoyancy_sponge_{case_type}_mudt_{mudt4}'
sponge_sol5 = f'acoustic_buoyancy_sponge_{case_type}_mudt_{mudt5}'

# Field to compute the error in
field_name = 'u_z'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}acoustic_buoyancy_compare_sponge_{case_type}_diff_mudt_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/field_output.nc'
sponge_results1 = f'{abspath(dirname(__file__))}/results/{sponge_sol1}/field_output.nc'
sponge_results2 = f'{abspath(dirname(__file__))}/results/{sponge_sol2}/field_output.nc'
sponge_results3 = f'{abspath(dirname(__file__))}/results/{sponge_sol3}/field_output.nc'
sponge_results4 = f'{abspath(dirname(__file__))}/results/{sponge_sol4}/field_output.nc'
sponge_results5 = f'{abspath(dirname(__file__))}/results/{sponge_sol5}/field_output.nc'

ref_data = Dataset(ref_results, 'r')
sponge_data1 = Dataset(sponge_results1, 'r')
sponge_data2 = Dataset(sponge_results2, 'r')
sponge_data3 = Dataset(sponge_results3, 'r')
sponge_data4 = Dataset(sponge_results4, 'r')
sponge_data5 = Dataset(sponge_results5, 'r')

# Arrays to store errors
sponge_errs1 = []
sponge_errs2 = []
sponge_errs3 = []
sponge_errs4 = []
sponge_errs5 = []

# Time values:
time = ref_data['time']

for i in np.arange(len(time)):
    t = time[i]

    ref_field = extract_gusto_field(ref_data, field_name, time_idx=i)
    sponge_field1 = extract_gusto_field(sponge_data1, field_name, time_idx=i)
    sponge_field2 = extract_gusto_field(sponge_data2, field_name, time_idx=i)
    sponge_field3 = extract_gusto_field(sponge_data3, field_name, time_idx=i)
    sponge_field4 = extract_gusto_field(sponge_data4, field_name, time_idx=i)
    sponge_field5 = extract_gusto_field(sponge_data5, field_name, time_idx=i)

    # Crop fields to be outside of the sponge
    ref_coords_X, ref_coords_Z = extract_gusto_coords(ref_data, field_name)
    ref_inds = np.where(ref_coords_Z <= z_max)
    ref_field = ref_field[ref_inds]

    test_coords_X, test_coords_Z = extract_gusto_coords(sponge_data1, field_name)
    test_inds = np.where(test_coords_Z <= z_max)
    sponge_field1 = sponge_field1[test_inds]
    sponge_field2 = sponge_field2[test_inds]
    sponge_field3 = sponge_field3[test_inds]
    sponge_field4 = sponge_field4[test_inds]
    sponge_field5 = sponge_field5[test_inds]

    # Normalised L1 error
    #no_damp_error = np.sum(np.abs(ref_field - no_damp_field))/np.sum(np.abs(ref_field))
    #sponge_error = np.sum(np.abs(ref_field - sponge_field))/np.sum(np.abs(ref_field))
    #sponge_error = np.sum(np.abs(ref_field - sponge_field))/np.sum(np.abs(ref_field))

    # What about L2?
    sponge_error1 = np.sqrt(np.sum((ref_field - sponge_field1)**2))/np.sum(np.abs(ref_field))
    sponge_error2 = np.sqrt(np.sum((ref_field - sponge_field2)**2))/np.sum(np.abs(ref_field))
    sponge_error3 = np.sqrt(np.sum((ref_field - sponge_field3)**2))/np.sum(np.abs(ref_field))
    sponge_error4 = np.sqrt(np.sum((ref_field - sponge_field4)**2))/np.sum(np.abs(ref_field))
    sponge_error5 = np.sqrt(np.sum((ref_field - sponge_field5)**2))/np.sum(np.abs(ref_field))

    sponge_errs1.append(sponge_error1)
    sponge_errs2.append(sponge_error2)
    sponge_errs3.append(sponge_error3)
    sponge_errs4.append(sponge_error4)
    sponge_errs5.append(sponge_error5)

sponge_errs1 = np.asarray(sponge_errs1)
sponge_errs2 = np.asarray(sponge_errs2)
sponge_errs3 = np.asarray(sponge_errs3)
sponge_errs4 = np.asarray(sponge_errs4)
sponge_errs5 = np.asarray(sponge_errs5)

print(f'Final sponge 1 error is {sponge_errs1[-1]}')
print(f'Final sponge 2 error is {sponge_errs2[-1]}')
print(f'Final sponge 3 error is {sponge_errs3[-1]}')
print(f'Final sponge 4 error is {sponge_errs4[-1]}')
print(f'Final sponge 5 error is {sponge_errs5[-1]}')

t_min = 25

plt.figure()
plt.semilogy(time[t_min:], sponge_errs1[t_min:], label=r'$\mu \Delta t$ = ' f'{mudt1}')
plt.semilogy(time[t_min:], sponge_errs2[t_min:], label=r'$\mu \Delta t$ = ' f'{mudt2}')
plt.semilogy(time[t_min:], sponge_errs3[t_min:], label=r'$\mu \Delta t$ = ' f'{mudt3}')
plt.semilogy(time[t_min:], sponge_errs4[t_min:], label=r'$\mu \Delta t$ = ' f'{mudt4}')
plt.semilogy(time[t_min:], sponge_errs5[t_min:], label=r'$\mu \Delta t$ = ' f'{mudt5}')
plt.legend(loc='center right', prop={'size': 10}, bbox_to_anchor=(1.3, 0.5))
plt.xlabel('Time (s)', size=14)
plt.ylabel('Error in the lower domain', size=14)
plt.xlim([t_min,time[-1]])
plt.ylim([1e-3,1e-1])
plt.xticks([25,50,75,100])

plt.grid()

# What about some black lines to show the time that the wave hits the model top?
# The acoustic wave of c=350 m/s travels 10 km in 28.57 s
#y_lim_range = np.logspace(-6, -3, num=20)
#plt.plot(np.ones_like(y_lim_range)*28.57, y_lim_range, c='k', linewidth=0.5)
#plt.plot(np.ones_like(y_lim_range)*28.57*2, y_lim_range, c='k', linewidth=0.5)
#plt.plot(np.ones_like(y_lim_range)*28.57*3, y_lim_range, c='k', linewidth=0.5)


plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()