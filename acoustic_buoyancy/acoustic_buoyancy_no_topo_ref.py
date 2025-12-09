'''
A script to solve the linear acoustic buoyancy equations.
A pressure perturbation is applied to excite a radially
propagating acoustic wave.
Here, we compute a reference solution, where the domain
is 50 km high instead of 20 km. This allows us to let the acoustic
wave travel vertically unimpeded.
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

savename = 'acoustic_buoyancy_runA_ref'

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 50.e3    # height of model top, in m

# Set dx = dx = 500 m
ncolumns=200
nlayers=100

# Time parameters
dt=0.25
tmax=100.0
dumpfreq=4

# Pressure perturbation
pert = 10              # Amplitude of perturbation
d = 1.e3                 # Gaussian half-width for the pressure perturbation
xc = 50.e3   # x location of the perturbation (50 km)
zc = 10.e3  # z location of the perturbation (10 km)

# Other parameters
cs = 350                 # Speed of sound, m/s


# ------------------------------------------------------------------------ #
# Our settings for this set up
# ------------------------------------------------------------------------ #

spinup_steps = 5  # Not necessary but helps balance initial conditions
alpha = 0.51      # Necessary to absorb grid scale waves
element_order = 1
u_eqn_type = 'vector_advection_form'

base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
mesh = ExtrudedMesh(
    base_mesh, layers=nlayers, layer_height=domain_height/nlayers
)
Vc = VectorFunctionSpace(mesh, "DG", 2)

domain = Domain(mesh, dt, "CG", element_order)

# Equation
parameters = BoussinesqParameters(mesh, cs=cs)

eqns = LinearAcousticBuoyancyEquations(
    domain=domain, parameters=parameters
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
transported_fields = [
    TrapeziumRule(domain, "u"),
    SSPRK3(domain, "p"),
    SSPRK3(domain, "b", options=b_opts)
]

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

x, z = SpatialCoordinate(mesh)

# Define a Gaussian pertubration.
# Confine it to a box of x in [30,70], z in [6,14]
expr = conditional((x > 30.e3),
                     conditional(x < 70.e3,
                                 conditional(z > 6.e3,
                                             conditional(z < 14.e3,
                                                         pert*exp(-((x-xc)**2 + (z-zc)**2)/d**2),
                                                         0), 0), 0), 0)

p_b = Function(Vp).interpolate(expr)
b0.assign(Constant(0.0))
p0.assign(p_b)
b_b = Function(Vb).interpolate(b0)

u0.project(as_vector([Constant(0.0), Constant(0.0)]))

# set the background buoyancy
stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
