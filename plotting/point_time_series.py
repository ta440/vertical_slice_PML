'''
Plot oscillations at points along the gravity wave to
try and estimate the frequency of the wave.

'''

from os.path import abspath, dirname
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tomplot import (
    set_tomplot_style, tomplot_cmap, plot_contoured_field,
    add_colorbar_ax, tomplot_field_title, tomplot_contours,
    extract_gusto_coords, extract_gusto_field, reshape_gusto_data
)

point_no = 9

pointsX = np.linspace(54,62,point_no)
pointsZ = np.linspace(5,45,point_no)

print(pointsX)
print(pointsZ)

w_vals = np.zeros(())

field_name = 'u_z'


results_dir = 'cs2023_nonhydrostatic_gaussian_h1000_looong'
results_file_name = f'{abspath(dirname(__file__))}/results/{results_dir}/field_output.nc'
plot_stem = f'{abspath(dirname(__file__))}/figures/{results_dir}'


data_file = Dataset(results_file_name, 'r')

time = data_file['time']
#NT = 200
NT = int(len(time))

w_vals = np.zeros((point_no,NT))

for i in np.arange(NT):
    print(i)
    field_data = extract_gusto_field(data_file, field_name, time_idx=i)
    coords_X, coords_Z = extract_gusto_coords(data_file, field_name)
    field_data, coords_X, coords_Z = \
        reshape_gusto_data(field_data, coords_X, coords_Z)

    for j in np.arange(point_no):
        w_vals[j,i] = field_data[int(pointsX[j]),int(pointsZ[j])]


fig, axes = plt.subplots(point_no,1,constrained_layout=True,sharex=True)
for j, ax in enumerate(axes):
    ax.plot(time[0:NT], w_vals[j,:])
    #ax.set_title(f'z = {np.round(coords_Z[int(pointsX[j]),int(pointsZ[j])],1)} km',size=10)
fig.supylabel('w')
fig.supxlabel('Time (s)')

plot_name = f'{plot_stem}_point_time_series.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(1,1)
for j in np.arange(point_no):   
    ax.plot(time[0:NT], w_vals[j,:], label=f'z = {np.round(coords_Z[int(pointsX[j]),int(pointsZ[j])],1)} km')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Time (s)')
plt.ylabel('w m/s')

plot_name = f'{plot_stem}_point_time_series_all.png'
print(f'Saving figure to {plot_name}')
fig.savefig(plot_name, bbox_inches='tight')
plt.close()

print('\n')
print(pointsX)
print(pointsZ)