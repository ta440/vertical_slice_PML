'''
A script to solve the linear acoustic buoyancy equations.
A pressure perturbation is applied to excite a radially
propagating acoustic wave.
Here, a sponge layer is used.
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
    KineticEnergy, VerticalKineticEnergy
)

#######################################################

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 20.e3    # height of model top, in m

# Set dx = dx = 500 m
ncolumns=200
nlayers=40

# Time parameters
dt=0.25
tmax=100.0
dumpfreq=4

# Pressure perturbation
pert = 10              # Amplitude of perturbation
d = 1.e3                 # Gaussian half-width for the pressure perturbation
xc = 0.5*domain_width   # x location of the perturbation
zc = 0.5*domain_height  # z location of the perturbation

# Other parameters
cs = 350                 # Speed of sound, m/s
sponge_depth = 2.e3     # 2000m, so the same width as the PML
mu_dt = 0.25               # strength of sponge layer, no units

########################################################

savename = f'acoustic_buoyancy_sponge_runA_mudt_{mu_dt}'

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
sponge = SpongeLayerParameters(
        mesh, H=domain_height, z_level=domain_height-sponge_depth, mubar=mu_dt/dt
    )

eqns = LinearAcousticBuoyancyEquations(
    domain=domain, parameters=parameters, sponge_options=sponge
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )

diagnostic_fields = [Perturbation('b'), ZComponent('u'), XComponent('u'), KineticEnergy(), VerticalKineticEnergy()]

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
# Initial conditions. A Gaussian perturbation on p.
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
p0 = stepper.fields("p")
b0 = stepper.fields("b")

# spaces
Vb = b0.function_space()
Vp = p0.function_space()

# Define a Gaussian perturbation

x, z = SpatialCoordinate(mesh)
expr = conditional((x > 0.3*domain_width),
                     conditional(x < 0.7*domain_width,
                                 conditional(z > 0.3*domain_height,
                                             conditional(z < 0.7*domain_height,
                                                         pert*exp(-((x-xc)**2 + (z-zc)**2)/d**2),
                                                         0), 0), 0), 0)

p_b = Function(Vp).interpolate(expr)
b0.assign(Constant(0.0))
p0.assign(p_b)
b_b = Function(Vb).interpolate(b0)

# No initial wind.
u0.project(as_vector([Constant(0.0), Constant(0.0)]))

# set the background buoyancy
stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
