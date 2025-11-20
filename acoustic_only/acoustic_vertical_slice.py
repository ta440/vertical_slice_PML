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
    BoussinesqSolver, boussinesq_hydrostatic_balance, LinearAcousticBuoyancyEquations,
    AcousticEquations
)

savename = 'acoustic_test'

ncolumns=50
nlayers=50
dt=0.1
tmax=50.0
dumpfreq=10

domain_width = 20.e3    # width of domain in x direction, in m
domain_height = 20.e3    # height of model top, in m
p_pert = 50             # Maximum amplitude of pressure perturbation
d = 1.e3                 # Gaussian half-width for the pressure perturbation
xc = 0.5*domain_width   # x location of the perturbation
zc = 0.5*domain_height  # z location of the perturbation
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
mesh = ExtrudedMesh(
    base_mesh, layers=nlayers, layer_height=domain_height/nlayers
)
Vc = VectorFunctionSpace(mesh, "DG", 2)

domain = Domain(mesh, dt, "CG", element_order)

# Equation
parameters = BoussinesqParameters(mesh, cs=cs)

eqns = AcousticEquations(
    domain=domain, parameters=parameters
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [XComponent('u'), ZComponent('u')]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [
    TrapeziumRule(domain, "u"),
    SSPRK3(domain, "p")
]

stepper = Timestepper(
    eqns, RK4(domain), io
)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
p0 = stepper.fields("p")

# spaces
Vp = p0.function_space()

x, z = SpatialCoordinate(mesh)

# Construct the Gaussian perturbation on the pressure field
p_expr = conditional((x > 0.3*domain_width),
                     conditional(x < 0.7*domain_width,
                                 conditional(z > 0.3*domain_height,
                                             conditional(z < 0.7*domain_height,
                                                         p_pert*exp(-((x-xc)**2 + (z-zc)**2)/d**2),
                                                         0), 0), 0), 0)


p_b = Function(Vp).interpolate(p_expr)
p0.assign(p_b)

u0.project(as_vector([Constant(0.0), Constant(0.0)]))

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
