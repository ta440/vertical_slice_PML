'''
A script to solve the comppressible Euler equations with the addition
of topography. This is part of experiments to determine if I 
can make a simple system that contains
orographically-driven gravity waves.
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
    CompressibleEulerEquations, SubcyclingOptions, RungeKuttaFormulation
)

savename = 'cs2023_nonhydrostatic_gaussian_h1000_looong'

ncolumns=100
nlayers=50
dt=5.0
tmax=20000.0
dumpfreq=10

domain_width = 144.e3   # width of domain in x direction, in m
domain_height = 35.e3   # height of model top, in m
a = 10.e3                # scale width of mountain profile, in m
hm = 1000.               # height of mountain, in m
Tsurf = 300.             # temperature of surface, in K
initial_wind = 10.0      # initial horizontal wind, in m/s
sponge_depth = 10000.0   # depth of sponge layer, in m
g = 9.810616             # acceleration due to gravity, in m/s^2
cp = 1004.5              # specific heat capacity at constant pressure
mu_dt = 0.15             # strength of sponge layer, no units
exner_surf = 1.0         # maximum value of Exner pressure at surface
max_iterations = 20      # maximum number of hydrostatic balance iterations
tolerance = 1e-8         # tolerance for hydrostatic balance iteration


# ------------------------------------------------------------------------ #
# Our settings for this set up
# ------------------------------------------------------------------------ #

spinup_steps = 5  # Not necessary but helps balance initial conditions
alpha = 0.51      # Necessary to absorb grid scale waves
element_order = 1
u_eqn_type = 'vector_invariant_form'

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

# Witch of Agensi:
#zs = a**2/((x-xc)**2 + a**2)


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
print(parameters.Omega)
sponge = SpongeLayerParameters(
    mesh, H=domain_height, z_level=domain_height-sponge_depth, mubar=mu_dt/dt
)
eqns = CompressibleEulerEquations(
    domain, parameters, sponge_options=sponge, u_transport_option=u_eqn_type
)

# I/O
output = OutputParameters(
        dirname=savename, 
        dumpfreq=dumpfreq, 
        dump_vtus=False, 
        dump_nc=True,
    )


diagnostic_fields = [
    Exner(parameters), ZComponent('u'), Perturbation('theta'),
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
    DGUpwind(eqns, "rho", advective_then_flux=True),
    DGUpwind(eqns, "theta", ibp=theta_opts.ibp)
]

# Linear solver
tau_values = {'rho': 1.0, 'theta': 1.0}
linear_solver = CompressibleSolver(eqns, alpha, tau_values=tau_values)

# Time stepper
stepper = SemiImplicitQuasiNewton(
    eqns, io, transported_fields, transport_methods,
    linear_solver=linear_solver, alpha=alpha, spinup_steps=spinup_steps
)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

# Thermodynamic constants required for setting initial conditions
# and reference profiles
N = parameters.N

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
u0.project(as_vector([initial_wind, 0.0]))

stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
