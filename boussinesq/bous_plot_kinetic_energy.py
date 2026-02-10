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

# Choose test variant
variant = 'gravity_wave'

# Solutions:
if variant == 'acoustic':
    no_damp_sol = 'bous_mount_gravity_wave'
    sponge_sol = 'bous_mount_sponge_acoustic_mudt_0.25'
    PML_sol = 'PML_bous_mount_acoustic_gamma0_0.0'
elif variant == 'gravity_wave':
    no_damp_sol = 'bous_mount_gravity_wave'
    sponge_sol = 'bous_mount_sponge_gravity_wave_mudt_0.25'
    PML_sol = 'PML_bous_mount_gravity_wave_gamma0_0.0'


# Field to compute the error in
field_name = 'VerticalKineticEnergy'

# Maximum height to compute errors
z_max = 18
zlims = [0,z_max]

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}bous_{variant}_{field_name}'

no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_sol}/diagnostics.nc'
sponge_results = f'{abspath(dirname(__file__))}/results/{sponge_sol}/diagnostics.nc'
PML_results = f'{abspath(dirname(__file__))}/results/{PML_sol}/diagnostics.nc'

no_damp_data = Dataset(no_damp_results, 'r')
sponge_data = Dataset(sponge_results, 'r')
PML_data = Dataset(PML_results, 'r')

no_damp_KE = no_damp_data.groups[field_name]['total'][:]
sponge_KE = sponge_data.groups[field_name]['total'][:]
PML_KE = PML_data.groups[field_name]['total'][:]

print('Initial kinetic energy', no_damp_KE[0])
print('Max kinetic energy, no damp', np.max(no_damp_KE))
print('Max kinetic energy, sponge', np.max(sponge_KE))
print('Max kinetic energy, PML', np.max(PML_KE))

time = np.asarray(no_damp_data['time'])

t_start = 10
t_end = -1


plt.figure()
plt.semilogy(time[t_start:t_end], no_damp_KE[t_start:t_end], label='No damping')
plt.semilogy(time[t_start:t_end], sponge_KE[t_start:t_end], label='Sponge')
plt.semilogy(time[t_start:t_end], PML_KE[t_start:t_end], label=r'PML unstretched')
plt.legend()
plt.xlabel('Time (s)')
if field_name == 'VerticalKineticEnergy':
    plt.ylabel(r"Vertical kinetic energy, m$^2$ s$^{-2}$")
elif field_name == 'KineticEnergy':
    plt.ylabel(r"Kinetic energy, m$^2$ s$^{-2}$")

plt.xlim([t_start,time[t_end]])

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')

plt.close()