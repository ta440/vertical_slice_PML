'''
A script to solve the comppressible Euler equations with the addition
of topography. This is part of experiments to determine if I 
can make a simple system that contains
orographically-driven gravity waves.
'''

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, exp, pi, cos, Function, Mesh, Constant, conditional
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, logger,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, SUPGOptions, TrapeziumRule, MaxKernel, MinKernel,
    CompressibleEulerEquations, SubcyclingOptions, RungeKuttaFormulation,
    Timestepper, RK4, XComponent,ForwardEuler, BoussinesqEquations, BoussinesqParameters,
    BoussinesqSolver, boussinesq_hydrostatic_balance, LinearAcousticBuoyancyEquations
)

savename = 'acousbuoy_test'

ncolumns=100
nlayers=50
dt=0.1
tmax=100.0
dumpfreq=10

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 20.e3    # height of model top, in m
a = 10.e3                # scale width of mountain profile, in m
hm = 1000.               # height of mountain, in m
Tsurf = 300.             # temperature of surface, in K
initial_wind = 10.0      # initial horizontal wind, in m/s
sponge_depth = 10000.0   # depth of sponge layer, in m
g = 9.810616             # acceleration due to gravity, in m/s^2
max_iterations = 20      # maximum number of hydrostatic balance iterations
tolerance = 1e-8         # tolerance for hydrostatic balance iteration
cs = 350                 # Speed of sound, m/s


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

# Default is Charney-Phillips staggering.
#eqns = LinearAcousticBuoyancyEquations(
#    domain=domain, parameters=parameters
#)
#  What if we try Lorenz?

space_names = {'u': 'HDiv', 'p': 'L2', 'b': 'L2'}
eqns = LinearAcousticBuoyancyEquations(
    domain=domain, parameters=parameters, space_names=space_names
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [Perturbation('b'), ZComponent('u')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
b_opts = SUPGOptions()
transported_fields = [
    TrapeziumRule(domain, "u"),
    SSPRK3(domain, "p"),
    SSPRK3(domain, "b", options=b_opts)
]

# Linear solver
#linear_solver = BoussinesqSolver(eqns)

# Time stepper
#stepper = SemiImplicitQuasiNewton(
#    eqns, io, transported_fields, transport_methods,
#    linear_solver=linear_solver
#)

stepper = Timestepper(
    eqns, RK4(domain), io
)


# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
p0 = stepper.fields("p")
b0 = stepper.fields("b")

# spaces
Vb = b0.function_space()
Vp = p0.function_space()

# Thermodynamic constants required for setting initial conditions
# and reference profiles
N = parameters.N

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
x, z = SpatialCoordinate(mesh)

# first setup the background buoyancy profile
# z.grad(bref) = N**2
#bref = Constant(0)
# interpolate the expression to the function
#b_b = Function(Vb).interpolate(bref)

# interpolate the expression to the function
#b0.interpolate(b_b)

#p_b = Function(Vp)
#boussinesq_hydrostatic_balance(eqns, b_b, p_b)
#p0.assign(p_b)

b0.assign(Constant(0.0))
p0.assign(Constant(0.0))

b_b = Function(Vb).interpolate(b0)
p_b = Function(Vp).interpolate(p0)

u_expr = conditional(z<(0.9*domain_height),
                     initial_wind*cos(z/(0.9*domain_height)),
                     0.0)

u0.project(as_vector([initial_wind, 0.0]))

# set the background buoyancy
stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
