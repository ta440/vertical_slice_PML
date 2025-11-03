'''

Test vertical transport for the PML.
Do I get the correct vertical transport
when the gradient operator is just in the vertical?

'''

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, cos, exp, sin, SpatialCoordinate, pi,
    min_value, as_vector, FunctionSpace, BrokenElement
)
from gusto import *

#################
# Parameters:
ncells_1d = 100
dt = 1.0
tmax = 500.
dumpfreq = 50.

Lx = 1000.       # width of domain, in m
Hz = 1000.       # height of domain, in m
xc = Lx/2.   # x-coordinate of centre of Gaussian bump
zc = Hz/4.      # z-coordinate of centre of Gaussian bump
lc = 2.*Lx/25.   # Decay rate of Gaussian
f0 = 0.0        # Base tracer value
f_pert = 0.1


period_mesh = PeriodicIntervalMesh(ncells_1d, Lx)
mesh = ExtrudedMesh(period_mesh, layers=ncells_1d, layer_height=Hz/ncells_1d)
domain = Domain(mesh, dt, "CG", 1)
x, z = SpatialCoordinate(mesh)


# Define the mixing ratio and density as tracers
f = ActiveTracer(
    name='f', space=domain.spaces('DG'),
    variable_type=TracerVariableType.mixing_ratio,
    transport_eqn=TransportEquationType.advective
)

V = domain.spaces("DG")
eqn = AdvectionEquation(domain, V, 'f')


dirname = 'z_transport'
    
# I/O
output = OutputParameters(
    dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
)

io = IO(domain, output)

transport_method = DGUpwind(eqn, 'f')
transport_scheme = SSPRK3(domain)

time_varying_velocity = False

u_expr = as_vector([0.0, 1.0])

timestepper = PrescribedTransport(
    eqn, transport_scheme, io, time_varying_velocity, transport_method
)

def l2_dist(xc,zc):
    return min_value(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

f0_expr = f0 + f_pert*exp(-l2_dist(xc, zc)/lc**2)

# Initial conditions
timestepper.fields("f").interpolate(f0_expr)
timestepper.fields("u").project(u_expr)

timestepper.run(t=0, tmax=tmax)