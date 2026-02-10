'''
Solving the Boussinesq equations in the presence
of a Gaussian mountain.
There are two confiugrations to examine:
1) A short timescale for acoustic waves.
2) A longer timescale for orographic gravity waves.

This script uses a higher domain for fewer reflections.
This is to provide a reference solution.
'''

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, exp, pi, cos, Function, Mesh, Constant,
    conditional
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, logger,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, SUPGOptions, TrapeziumRule, MaxKernel, MinKernel,
    CompressibleEulerEquations, SubcyclingOptions, RungeKuttaFormulation,
    Timestepper, RK4, XComponent,ForwardEuler, BoussinesqEquations, BoussinesqParameters,
    BoussinesqSolver, boussinesq_hydrostatic_balance,
)

variant = 'gravity_wave'

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 50.e3    # height of the extended model top, in m
z_base = 20.e3           # height of the main domain, in m

# Set dx = dz = 500 m
ncolumns=200
nlayers=100

# Time parameters. Create 100 outputs
# For the acoustic wave:
if variant == 'acoustic':
    # For the acoustic wave:
    dt = 0.25
    tmax = 150.
    dumpfreq=4
    savename = 'bous_mount_acoustic_ref'
elif variant == 'gravity_wave':
    # For the orographic gravity wave:
    dt=10
    tmax=10000.0
    dumpfreq=10
    savename = 'bous_mount_gravity_wave_ref_feb10'

# Mountain parameters
a = 10.e3                # scale width of mountain profile, in m
hm = 1000.               # height of mountain, in m

# Other parameters
Tsurf = 300.             # temperature of surface, in K
initial_wind = 10.0      # initial horizontal wind, in m/s
g = 9.810616             # acceleration due to gravity, in m/s^2
max_iterations = 20      # maximum number of hydrostatic balance iterations
tolerance = 1e-8         # tolerance for hydrostatic balance iteration
cs = 350                 # Speed of sound, m/s

# ------------------------------------------------------------------------ #
# Our settings for this set up
# ------------------------------------------------------------------------ #

element_order = 1
u_eqn_type = 'vector_advection_form'

base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
ext_mesh = ExtrudedMesh(
    base_mesh, layers=nlayers, layer_height=domain_height/nlayers
)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

# Describe the mountain
xc = domain_width/2.
x, z = SpatialCoordinate(ext_mesh)

# Make the mountain a simple Gaussian one
zs = hm * exp(-((x - xc)/a)**2)

# Hybrid z blending factor. Only make nonzero from z in [0,20] km.
A_z = conditional(z <= z_base, (z_base - z) / z_base, 0)
xexpr = as_vector(
    [x, z + A_z * zs]
)

# Make new mesh
new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)
mesh._base_mesh = base_mesh  # Force new mesh to inherit original base mesh
domain = Domain(mesh, dt, "CG", element_order)

# Equation
parameters = BoussinesqParameters(mesh, cs=cs)

# Try both with and without the sponge
eqns = BoussinesqEquations(
    domain, parameters, u_transport_option=u_eqn_type
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [Perturbation('b'), ZComponent('u'), XComponent('u')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
b_opts = SUPGOptions()
transport_methods = [
    DGUpwind(eqns, "u"),
    DGUpwind(eqns, "p"),
    DGUpwind(eqns, "b", ibp=b_opts.ibp)
]

# Choose timestepper depending on simulation time
if variant == 'acoustic':
    # For the acoustic waves, run for a short time,
    # use an explicit timestepper and small dt
    stepper = Timestepper(
        eqns, RK4(domain), io, transport_methods, physics_parametrisations=None
    )
elif variant == 'gravity_wave':
    # If resolving the gravity waves, use an implicit method, theta=0.5.
    stepper = Timestepper(
        eqns, TrapeziumRule(domain), io, transport_methods, physics_parametrisations=None
    )


# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
p0 = stepper.fields("p")
b0 = stepper.fields("b")

# spaces
Vu = u0.function_space()
Vb = b0.function_space()
Vp = p0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
N = parameters.N

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
x, z = SpatialCoordinate(mesh)

# First, setup the background buoyancy profile
# db/dz = N**2
bref = z*(N**2)
b_b = Function(Vb).interpolate(bref)
b0.assign(b_b)

# Define pressure ourselves to be in hydrostatic balance
pref = (z**2)*(N**2)/2 - 20000
p_b = Function(Vp).interpolate(pref)
p0.assign(p_b)

# Derive pressure by hydrostatic pressure
#boussinesq_hydrostatic_balance(eqns, b_b, p_b)
#p0.assign(p_b)

# Zonal wind field
u_b = Function(Vu).project(as_vector([initial_wind, 0.0]))

# For the steady test:
#u_b = Function(Vu).project(as_vector([Constant(0.0), 0.0]))

u0.assign(u_b)

# set the background buoyancy
stepper.set_reference_profiles([('u', u_b), ('p', p_b), ('b', b_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
