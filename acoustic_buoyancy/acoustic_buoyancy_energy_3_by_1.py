'''
Plot energies from using a PML in the linear
acoustic-buoyancy equations.

'''
import numpy as np
import scipy
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

###############################################
# Choose the simulation data

no_damp_name = 'acoustic_buoyancy_no_damp'
sponge_name = 'acoustic_buoyancy_sponge_mudt_0.25'
PML_name = 'PML_acoustic_buoyancy_gamma0_0.25'


#################################################

no_damp_results = f'{abspath(dirname(__file__))}/results/{no_damp_name}/diagnostics.nc'
sponge_results = f'{abspath(dirname(__file__))}/results/{sponge_name}/diagnostics.nc'
PML_results = f'{abspath(dirname(__file__))}/results/{PML_name}/diagnostics.nc'

no_damp_data = Dataset(no_damp_results, 'r')
sponge_data = Dataset(sponge_results, 'r')
PML_data = Dataset(PML_results, 'r')

time = np.asarray(no_damp_data['time'])

fig, axs = plt.subplots(3,1, figsize = (8,11), sharey=True, constrained_layout=True)
ax1, ax2, ax3 = axs

for i in np.arange(3):

    if i == 0:
        data = no_damp_data
        ax_val = ax1
        ax_val.set_title('No damping')
    elif i == 1:
        data = sponge_data
        ax_val = ax2
        ax_val.set_title('Sponge layer')
    else: 
        data = PML_data
        ax_val = ax3
        ax_val.set_title(r'PML, $\gamma_0 = 0.25$')

    E_K = data.groups['KineticEnergy']['total'][:]
    E_I = data.groups['BousInternalEnergy']['total'][:]
    E_P = data.groups['BousPotentialEnergy']['total'][:]

    E_T = E_K + E_I + E_P

    ax_val.plot(time, E_K, label='Kinetic energy')
    ax_val.plot(time, E_I, label='Internal energy')
    ax_val.plot(time, E_P, label='Potential energy')
    ax_val.plot(time, E_T, label='Total energy')

    ax_val.set_xlim([0,100])
    ax_val.set_ylim([0,650])

    y_lim_range = np.linspace(0,650,100)
    ax_val.plot(np.ones_like(y_lim_range)*28.57, y_lim_range, c='k', linewidth=0.5, linestyle='--')
    ax_val.plot(np.ones_like(y_lim_range)*28.57*2, y_lim_range, c='k', linewidth=0.5, linestyle='--')
    ax_val.plot(np.ones_like(y_lim_range)*28.57*3, y_lim_range, c='k', linewidth=0.5, linestyle='--')  

ax3.legend(loc='lower center', prop={'size': 10}, bbox_to_anchor=(0.5, -0.62))

fig.supxlabel('Time (s)', x=0.55, y=0.11)
fig.supylabel(r'Domain-integrated energy (m$^4$ s$^{-2}$)')

figure_stem = f'{abspath(dirname(__file__))}/figures/'
savename = f'{figure_stem}acoustic_buoyancy_energies_3_by_1'

plt.savefig(savename, bbox_inches='tight')
print(f'saved figure to {savename}')