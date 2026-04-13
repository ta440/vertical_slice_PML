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
ref_sol = 'acoustic_buoyancy_runA_ref_test'
no_damp_sol = 'acoustic_buoyancy_runA'
sponge_sol = 'acoustic_buoyancy_sponge_runA_mudt_0.25'
PML_sol = 'PML_acoustic_buoyancy_runA_gamma0_0'
PML_gamma_sol = 'PML_acoustic_buoyancy_runA_gamma0_1.0'

# Field to compute the error in
field_name = 'VerticalKineticEnergy'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}acoustic_buoyany_{field_name}'

ref_results = f'{abspath(dirname(__file__))}/results/{ref_sol}/diagnostics.nc'
no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/diagnostics.nc'
sponge_results = f'{abspath(dirname(__file__))}/results/{sponge_sol}/diagnostics.nc'
PML_results = f'{abspath(dirname(__file__))}/results/{PML_sol}/diagnostics.nc'
PML_gamma_results = f'{abspath(dirname(__file__))}/results/{PML_gamma_sol}/diagnostics.nc'

ref_data = Dataset(ref_results, 'r')
no_damp_data = Dataset(no_damp_results, 'r')
sponge_data = Dataset(sponge_results, 'r')
PML_data = Dataset(PML_results, 'r')
PML_gamma_data = Dataset(PML_gamma_results, 'r')

print(ref_data.groups)

ref_KE = ref_data.groups[field_name]['total'][:]
no_damp_KE = no_damp_data.groups[field_name]['total'][:]
sponge_KE = sponge_data.groups[field_name]['total'][:]
PML_KE = PML_data.groups[field_name]['total'][:]
PML_gamma_KE = PML_gamma_data.groups[field_name]['total'][:]

print('Initial kinetic energy', ref_KE[0])
print('Max kinetic energy', np.max(ref_KE))

time = np.asarray(ref_data['time'])

t_end = 300


plt.figure()
#plt.semilogy(time[0:t_end], ref_KE[0:t_end], label='Reference')
plt.semilogy(time[0:t_end], no_damp_KE[0:t_end], label='No damping')
plt.semilogy(time[0:t_end], sponge_KE[0:t_end], label='Sponge')
plt.semilogy(time[0:t_end], PML_KE[0:t_end], label=r'PML unstretched')
plt.semilogy(time[0:t_end], PML_gamma_KE[0:t_end], label=r'PML with $\gamma_0 = 1$')
#plt.plot(28.6, np.arange([1e-15,]))
plt.legend()
plt.xlabel('Time (s)')
if field_name == 'VerticalKineticEnergy':
    plt.ylabel(r"Vertical kinetic energy, m$^2$ s$^{-2}$")
elif field_name == 'KineticEnergy':
    plt.ylabel(r"Kinetic energy, m$^2$ s$^{-2}$")

plt.xlim([0,time[t_end]])

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()