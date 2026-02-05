'''
Test for the compressible Euler equations,
in the presence
of a Gaussian mountain.

There are two confiugrations to examine:
1) A short timescale for acoustic waves.
2) A longer timescale for orographic gravity waves.

This script applies no damping at the model top.

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
    Timestepper, RK4, XComponent,ForwardEuler
)

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
initial_wind = 10.0      # initial horizontal wind, in m/s
g = 9.810616             # acceleration due to gravity, in m/s^2
cs = 350                 # Speed of sound, m/s
Tsurf = 300.             # temperature of surface, in K
cp = 1004.5              # specific heat capacity at constant pressure
exner_surf = 1.0         # maximum value of Exner pressure at surface
max_iterations = 20      # maximum number of hydrostatic balance iterations
tolerance = 1e-8         # tolerance for hydrostatic balance iteration

savename = f'euler_no_damp_{variant}'

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
parameters = CompressibleParameters(mesh, g=g, cp=cp, T_0=Tsurf)

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
    Perturbation('rho')
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

# Choose timestepper depending on simulation time
if variant == 'acoustic':
    # For the acoustic waves, run for a short time,
    # use an explicit timestepper and small dt
    stepper = Timestepper(
        eqns, RK4(domain), io, transport_methods, physics_parametrisations=None
    )
elif variant == 'gravity_wave':
    # If resolving the gravity waves, use an implicit method, theta=0.5.
    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)   
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
    stepper = Timestepper(
        eqns, TrapeziumRule(domain), io, transport_methods, physics_parametrisations=None
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

N = parameters.N
x, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
x, z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic exner
exner = Function(Vr)
rho_b = Function(Vr)

# Set up kernels to evaluate global minima and maxima of fields
min_kernel = MinKernel()
max_kernel = MaxKernel()

# First solve hydrostatic balance that gives Exner = 1 at bottom boundary
# This gives us a guess for the top boundary condition
bottom_boundary = Constant(exner_surf, domain=mesh)
logger.info(f'Solving hydrostatic with bottom Exner of {exner_surf}')
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=False, exner_boundary=bottom_boundary,
    solve_for_rho=True
)

# Solve hydrostatic balance again, but now use minimum value from first
# solve as the *top* boundary condition for Exner
top_value = min_kernel.apply(exner)
top_boundary = Constant(top_value, domain=mesh)
logger.info(f'Solving hydrostatic with top Exner of {top_value}')
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
)
max_bottom_value = max_kernel.apply(exner)

# Now we iterate, adjusting the top boundary condition, until this gives
# a maximum value of 1.0 at the surface
lower_top_guess = 0.9*top_value
upper_top_guess = 1.2*top_value
for i in range(max_iterations):
    # If max bottom Exner value is equal to desired value, stop iteration
    if abs(max_bottom_value - exner_surf) < tolerance:
        break

    # Make new guess by average of previous guesses
    top_guess = 0.5*(lower_top_guess + upper_top_guess)
    top_boundary.assign(top_guess)

    logger.info(
        f'Solving hydrostatic balance iteration {i}, with top Exner value '
        + f'of {top_guess}'
    )

    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary
    )

    max_bottom_value = max_kernel.apply(exner)

    # Adjust guesses based on new value
    if max_bottom_value < exner_surf:
        lower_top_guess = top_guess
    else:
        upper_top_guess = top_guess

logger.info(f'Final max bottom Exner value of {max_bottom_value}')

# Perform a final solve to obtain hydrostatically balanced rho
compressible_hydrostatic_balance(
    eqns, theta_b, rho_b, exner, top=True, exner_boundary=top_boundary,
    solve_for_rho=True
)

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
