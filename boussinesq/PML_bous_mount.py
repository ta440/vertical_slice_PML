'''
Solving the Boussinesq equations in the presence
of a Gaussian mountain.
There are two confiugrations to examine:
1) A short timescale for acoustic waves.
2) A longer timescale for orographic gravity waves.

This script implements a PML to damp waves near the model
top.

'''

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, exp, pi, cos, Function, Mesh, Constant
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, logger,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, SUPGOptions, TrapeziumRule, MaxKernel, MinKernel,
    CompressibleEulerEquations, SubcyclingOptions, RungeKuttaFormulation,
    Timestepper, RK4, XComponent,ForwardEuler, BoussinesqEquations, BoussinesqParameters,
    BoussinesqSolver, boussinesq_hydrostatic_balance, PMLParameters, BackwardEuler
)

##################################################################

variant = 'acoustic'
# variant = 'gravity_wave'

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 20.e3    # height of model top, in m

# Set dx = dx = 500 m
ncolumns=200
nlayers=40

# Time parameters. Create 100 outputs
# For the acoustic wave:
if variant == 'acoustic':
    # For the acoustic wave:
    dt = 0.1
    tmax = 100.
    dumpfreq=10
elif variant == 'gravity_wave':
    # For the orographic gravity wave:
    dt=10
    tmax=10000.0
    dumpfreq=10

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

# PML parameters:
#gamma0 = 0.0               # Try with no stretching
gamma0 = 0.5              # PML stretching factor

savename = f'PML_bous_mount_{variant}_gamma0_{gamma0}'

# ------------------------------------------------------------------------ #
# Our settings for this set up
# ------------------------------------------------------------------------ #

spinup_steps = 5  # Not necessary but helps balance initial conditions
alpha = 0.51      # Necessary to absorb grid scale waves
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

xexpr = as_vector(
    [x, z + ((domain_height - z) / domain_height) * zs]
)

# Make new mesh
new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)
mesh._base_mesh = base_mesh  # Force new mesh to inherit original base mesh
domain = Domain(mesh, dt, "CG", element_order)

# Equation
parameters = BoussinesqParameters(mesh, cs=cs)

# Modify parameters as necessary
PML_pars = PMLParameters(mesh, H=domain_height, gamma0=0)

# Try both with and without the sponge
eqns = BoussinesqEquations(
    domain, parameters, PML_options=PML_pars, u_transport_option=u_eqn_type
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [Perturbation('b'), ZComponent('u'), XComponent('u'), ZComponent('q_u'), XComponent('q_u')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
b_opts = SUPGOptions()
transported_fields = [
    TrapeziumRule(domain, "u"),
    SSPRK3(domain, "p"),
    SSPRK3(domain, "b", options=b_opts)
]

transport_methods = [
    DGUpwind(eqns, "u"),
    DGUpwind(eqns, "p"),
    DGUpwind(eqns, "b", ibp=b_opts.ibp)
]


# OLD: transport PML variables also
#transport_methods = [
#    DGUpwind(eqns, "u"),
#    DGUpwind(eqns, "p"),
#    DGUpwind(eqns, "b", ibp=b_opts.ibp),
#    DGUpwind(eqns, "q_u"),
#    DGUpwind(eqns, "q_p"),
#    DGUpwind(eqns, "q_b", ibp=b_opts.ibp)
#]

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

# Zonal wind field
u_b = Function(Vu).project(as_vector([initial_wind, 0.0]))

u0.assign(u_b)

# set the background profile
stepper.set_reference_profiles([('u', u_b), ('p', p_b), ('b', b_b)])

# Set PML variables to zero.
# This should be the case by default.
#q_u0 = stepper.fields("q_u")
#q_p0 = stepper.fields("q_p")
#q_b0 = stepper.fields("q_b")
#q_u0.project(as_vector([Constant(0.0), Constant(0.0)]))
#q_b0.assign(Constant(0.0))
#q_p0.assign(Constant(0.0))

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
