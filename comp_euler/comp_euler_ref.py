'''
Test for the compressible Euler equations,
in the presence
of a Gaussian mountain.

Only create a reference solution for the acoustic configuration

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
    Timestepper, RK4, XComponent,ForwardEuler,KineticEnergy, VerticalKineticEnergy
)

domain_width = 100.e3    # width of domain in x direction, in m
domain_height = 35.e3    # height of model top, in m
z_base = 20.e3           # height of the main domain, in m

# Set dx = dx = 500 m
ncolumns=200
nlayers=70

# Time parameters.
tmax = 150.
dt = 0.25
dumpfreq=4
#dt = 0.1
#dumpfreq=10

# Mountain parameters
a = 10.e3                # scale width of mountain profile, in m
hm = 1000.               # height of mountain, in m

# Other parameters
initial_wind = 10.0      # initial horizontal wind, in m/s
cs = 350                 # Speed of sound, m/s
Tsurf = 300.             # temperature of surface, in K

savename = f'euler_acoustic_ref_{domain_height}'

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
parameters = CompressibleParameters(mesh, T_0=Tsurf)

eqns = CompressibleEulerEquations(
    domain, parameters, sponge_options=None, u_transport_option=u_eqn_type
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )

diagnostic_fields = [
    Exner(parameters), XComponent('u'), ZComponent('u'), Perturbation('theta'),
    Perturbation('rho'), KineticEnergy(), VerticalKineticEnergy()
]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)
theta_opts = SUPGOptions()
transported_fields = [
    TrapeziumRule(domain, "u", subcycling_options=subcycling_opts),
    SSPRK3(
        domain, "rho", rk_formulation=RungeKuttaFormulation.predictor,
        subcycling_options=subcycling_opts
    ),
    SSPRK3(
        domain, "theta", options=theta_opts,
        subcycling_options=subcycling_opts
    )
]

transport_methods = [
    DGUpwind(eqns, "u"),
    DGUpwind(eqns, "rho"),
    DGUpwind(eqns, "theta", ibp=theta_opts.ibp)
]

stepper = Timestepper(
    eqns, RK4(domain), io, transport_methods, physics_parametrisations=None
)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = u0.function_space()
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

x, z = SpatialCoordinate(mesh)

# Try an analytical initial condition
# Necessary quantities:
N = parameters.N
N2 = N**2
g = parameters.g
p0 = parameters.p_0
cp = parameters.cp
kappa = parameters.kappa
Rd = parameters.R_d
kappa_fact = (1-kappa)/kappa

# Analytical ICs
thetab = Tsurf*exp(N2*z/g)
exnerb = (g**2/(cp*N2*Tsurf))*(exp(-N2*z/g) - 1) + 1
rhob = p0/(Rd*thetab)*(exnerb**(kappa_fact))

# Interpolate these!
theta_b = Function(Vt).interpolate(thetab)
exner = Function(Vr).interpolate(exnerb)
rho_b = Function(Vr).interpolate(rhob)

theta0.assign(theta_b)
rho0.assign(rho_b)

u_b = Function(Vu).project(as_vector([initial_wind, 0.0]))
u0.assign(u_b)

# set the background buoyancy
stepper.set_reference_profiles([('u', u_b), ('rho', rho_b), ('theta', theta_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
