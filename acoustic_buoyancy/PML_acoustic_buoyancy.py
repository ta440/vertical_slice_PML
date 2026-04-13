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
    Domain, CompressibleParameters, logger, PMLParameters,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, SUPGOptions, TrapeziumRule, MaxKernel, MinKernel,
    CompressibleEulerEquations, SubcyclingOptions, RungeKuttaFormulation,
    Timestepper, RK4, XComponent,ForwardEuler, BoussinesqEquations, BoussinesqParameters,
    boussinesq_hydrostatic_balance, LinearAcousticBuoyancyEquations,
    PMLParameters, KineticEnergy, VerticalKineticEnergy, time_derivative, transport,
    BousInternalEnergy, BousPotentialEnergy, BousPMLEnergy
    
)

#################################

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

# PML parameters
gamma0 = 0.45

savename = f'PML_acoustic_buoyancy_gamma0_{gamma0}'

# ------------------------------------------------------------------------ #
# Our settings for this set up
# ------------------------------------------------------------------------ #

spinup_steps = 5  # Not necessary but helps balance initial conditions
alpha = 0.51      # Necessary to absorb grid scale waves
element_order = 1

base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
mesh = ExtrudedMesh(
    base_mesh, layers=nlayers, layer_height=domain_height/nlayers
)
Vc = VectorFunctionSpace(mesh, "DG", 2)

domain = Domain(mesh, dt, "CG", element_order)

# Equation
parameters = BoussinesqParameters(mesh, cs=cs)
PML_pars = PMLParameters(mesh, H=domain_height, gamma0=gamma0)

eqns = LinearAcousticBuoyancyEquations(
    domain=domain, parameters=parameters, PML_options=PML_pars
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [Perturbation('b'), ZComponent('u'), XComponent('u'), ZComponent('q_u'), XComponent('q_u'), 
                     KineticEnergy(), VerticalKineticEnergy(), BousInternalEnergy(cs=cs), BousPotentialEnergy(N=parameters.N),
                     BousPMLEnergy(sigma=eqns.sigma, cs=cs)]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

suboptions = {}
suboptions.update({'b': [time_derivative, transport]})
b_opts = SUPGOptions(suboptions=suboptions)

stepper = Timestepper(
    eqns, RK4(domain, options=b_opts), io
)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
p0 = stepper.fields("p")
b0 = stepper.fields("b")
q_u0 = stepper.fields("q_u")
q_p0 = stepper.fields("q_p")

# spaces
Vb = b0.function_space()
Vp = p0.function_space()

x, z = SpatialCoordinate(mesh)

# Define a Gaussian pertubration
expr = conditional((x > 0.3*domain_width),
                     conditional(x < 0.7*domain_width,
                                 conditional(z > 0.3*domain_height,
                                             conditional(z < 0.7*domain_height,
                                                         pert*exp(-((x-xc)**2 + (z-zc)**2)/d**2),
                                                         0), 0), 0), 0)

# Apply the perturbation to the buoyancy
#b_b = Function(Vb).interpolate(b_expr)
#b0.interpolate(b_b)

#p_b = Function(Vp)
#boussinesq_hydrostatic_balance(eqns, b_b, p_b)
#p0.assign(p_b)

p_b = Function(Vp).interpolate(expr)
b0.assign(Constant(0.0))
p0.assign(p_b)
b_b = Function(Vb).interpolate(b0)

u0.project(as_vector([Constant(0.0), Constant(0.0)]))

# PML variables all set to zero
q_u0.project(as_vector([Constant(0.0), Constant(0.0)]))
q_p0.assign(Constant(0.0))

# set the background buoyancy
stepper.set_reference_profiles([('p', p_b), ('b', b_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
