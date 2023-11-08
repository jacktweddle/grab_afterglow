import numpy as np
import emcee
import time
import faulthandler
from multiprocessing import Pool
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
faulthandler.enable()


# Define function to plug into f_solve to find a root
def eta_func(eta, p, epsilon_e, epsilon_B, gamma_c, gamma_m):
    return eta**(1/(2-p)) + eta**(1/(2*(2-p)))*np.sqrt(epsilon_e/epsilon_B) - (gamma_c/gamma_m)


# Function to produce array of circumburst density  values, array of constants in case of ISM (k=0)
def circumburst_density(t, rho_ref, gamma):
    rho_1_vals = (rho_ref*R_ref**k) / (2**(2*k)*gamma**(2*k)*c**k*t**k)

    return rho_1_vals


# Define a function to perform the RK4 algorithm, allowing for the possibility that the timestep changes
# which does occur due to SR effects when converting between engine and observer frames

def RK4(R, F, t_vals, h, E, rho_ref):
    # Set initial value and define empty list within which to store values of R
    R0 = R
    R_vals = []

    # Loop over all time values, calculating vectors of k1-k4 coefficients at each
    # time step t, and use these to update R0 to the next vector R1 at time t+h
    # using the RK4 method
    for i in range(0, len(t_vals)-1):
        k1 = h[i]*F(R0, t_vals[i], E, rho_ref)
        k2 = h[i]*F(R0 + k1/2, t_vals[i] + h[i]/2, E, rho_ref)
        k3 = h[i]*F(R0 + k2/2, t_vals[i] + h[i]/2, E, rho_ref)
        k4 = h[i]*F(R0 + k3, t_vals[i] + h[i], E, rho_ref)
        R1 = R0 + (k1 + 2*k2 + 2*k3 + k4)/6

        # Store and reset for next time step
        R_vals.append(R1)
        R0 = R1

    # Convert list of R vectors into array of R vectors; easier to select
    # the elements of R to plot
    R_vals = np.array(R_vals)
    return R_vals


# Define a function to calculate the vector F = dR/dt = (d(gamma)/dt, dr/dt) in OBSERVER FRAME
# Accounts for the case of a stellar wind
def F_obs(R, t, E, rho_ref):
    dgammadt = ((k-3)/(8-(2*k))) * ((9*E*2**k)/(1024*np.pi*rho_ref*R_ref**k*c**(5-k)))**(1/(8-(2*k)))*t**(((3*k)-11)/(8-(2*k)))
    dRdt = 4*R[0]*c*((2*t*dgammadt)+R[0])

    return np.array([dgammadt, dRdt])


def calc_spectrum(t, Y, gamma, gamma_m, gamma_c, nu_c, nu_m, nu_max, F_max, nu_c_ic, nu_m_ic, nu_max_ic, F_max_ic, nu_c_p, nu_m_p, nu_max_p, F_max_p, params):
    if nu_c < nu_m:
        low_freq_vals = np.geomspace(nu_c/1000, nu_c, 10000)
        middle_freq_vals = np.geomspace(nu_c, nu_m, 10000)
        high_freq_vals = np.geomspace(nu_m, nu_max, 10000)

        low_freq_fluxes = np.array((low_freq_vals/nu_c)**(1/3)*F_max)
        middle_freq_fluxes = np.array((middle_freq_vals/nu_c)**(-1/2)*F_max)
        high_freq_fluxes = np.array((nu_m/nu_c)**(-1/2)*(high_freq_vals/nu_m)**(-params[2]/2)*F_max)

        # If IC is relevant, define that part of the spectrum
        if 10**params[3] > 10**params[4]:
            nu_kn = (2*gamma*m_e*c**2) / (gamma_c*h)  # Frequency of the KN cutoff in the observer frame in the fast cooling regime

            low_freq_ic_vals = np.geomspace((nu_c/1000)*4*gamma_c**2, nu_c_ic, 10000)
            middle_freq_ic_vals = np.geomspace(nu_c_ic, nu_m_ic, 10000)
            high_freq_ic_vals = np.geomspace(nu_m_ic, nu_max_ic, 10000)

            low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_c_ic)**(1/3)*F_max_ic)
            middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_c_ic)**(-1/2)*F_max_ic)
            high_freq_ic_fluxes = np.array((nu_m_ic/nu_c_ic)**(-1/2)*(high_freq_ic_vals/nu_m_ic)**(-params[2]/2)*F_max_ic)

            # Determine the corresponding frequency of the KN cutoff in the IC spectrum
            if nu_kn < np.max(middle_freq_vals):
                nu_kn_ic = 4 * gamma_c**2 * nu_kn

            elif nu_kn > np.max(middle_freq_vals):
                nu_kn_ic = 4 * gamma_m**2 * nu_kn

    # If in slow cooling regime
    elif nu_c > nu_m:
        low_freq_vals = np.geomspace(nu_m/1000, nu_m, 10000)
        middle_freq_vals = np.geomspace(nu_m, nu_c, 10000)
        high_freq_vals = np.geomspace(nu_c, nu_max, 10000)

        low_freq_fluxes = np.array((low_freq_vals/nu_m)**(1/3)*F_max)
        middle_freq_fluxes = np.array((middle_freq_vals/nu_m)**((1-params[2])/2)*F_max)
        high_freq_fluxes = np.array((nu_c/nu_m)**((1-params[2])/2)*(high_freq_vals/nu_c)**(-params[2]/2)*F_max)

        # If IC is relevant, define that part of the spectrum
        if 10**params[3] > 10**params[4]:
            if Y >= 1:
                nu_kn = (2*gamma*m_e*c**2) / (gamma_m*h)  # Frequency of the KN cutoff in the observer frame in the slow cooling regime

                low_freq_ic_vals = np.geomspace((nu_m/1000)*4*gamma_m**2, nu_m_ic, 10000)
                middle_freq_ic_vals = np.geomspace(nu_m_ic, nu_c_ic, 10000)
                high_freq_ic_vals = np.geomspace(nu_c_ic, nu_max_ic, 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_m_ic)**(1/3)*F_max_ic)
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_m_ic)**((1-params[2])/2)*F_max_ic)
                high_freq_ic_fluxes = np.array((nu_c_ic/nu_m_ic)**((1-params[2])/2)*(high_freq_ic_vals/nu_c_ic)**(-params[2]/2)*F_max_ic)

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c**2 * nu_kn

    freqs = np.concatenate([low_freq_vals[:-1], middle_freq_vals[:-1], high_freq_vals])
    fluxes = np.concatenate([low_freq_fluxes[:-1], middle_freq_fluxes[:-1], high_freq_fluxes])

    fluxes = fluxes[freqs < nu_max]
    freqs = freqs[freqs < nu_max]

    nu_Fnu = freqs * fluxes

    low_freq_p_vals = np.geomspace(nu_m_p/1000, nu_m_p, 10000)
    middle_freq_p_vals = np.geomspace(nu_m_p, nu_c_p, 10000)
    high_freq_p_vals = np.geomspace(nu_c_p, nu_max_p, 10000)

    low_freq_p_fluxes = np.array((low_freq_p_vals/nu_m_p)**(1/3)*F_max_p)
    middle_freq_p_fluxes = np.array((middle_freq_p_vals/nu_m_p)**((1-params[2])/2)*F_max_p)
    high_freq_p_fluxes = np.array((nu_c_p/nu_m_p)**((1-params[2])/2)*(high_freq_p_vals/nu_c_p)**(-params[2]/2)*F_max_p)

    proton_freqs = np.concatenate([low_freq_p_vals[:-1], middle_freq_p_vals[:-1], high_freq_p_vals])
    proton_fluxes = np.concatenate([low_freq_p_fluxes[:-1], middle_freq_p_fluxes[:-1], high_freq_p_fluxes])

    proton_fluxes = proton_fluxes[proton_freqs < nu_max_p]
    proton_freqs = proton_freqs[proton_freqs < nu_max_p]

    p_flux_interp = interp1d(np.log10(proton_freqs), np.log10(proton_fluxes))

    for j in range(0, len(freqs)):
        if (freqs[j] > proton_freqs[0]) and (freqs[j] < proton_freqs[-1]):
            nu_Fnu[j] += (10**p_flux_interp(np.log10(freqs[j])) * freqs[j])

    total_freqs = np.concatenate([proton_freqs[proton_freqs < np.min(freqs)], freqs, proton_freqs[proton_freqs > np.max(freqs)]])
    total_nu_Fnu = np.concatenate([(proton_freqs[proton_freqs < np.min(freqs)]*proton_fluxes[proton_freqs < np.min(freqs)]), nu_Fnu, (proton_freqs[proton_freqs > np.max(freqs)]*proton_fluxes[proton_freqs > np.max(freqs)])])

    if 10**params[3] > 10**params[4]:
        if Y >= 1:
            ic_freqs = np.concatenate([low_freq_ic_vals[:-1], middle_freq_ic_vals[:-1], high_freq_ic_vals])
            ic_fluxes = np.concatenate([low_freq_ic_fluxes[:-1], middle_freq_ic_fluxes[:-1], high_freq_ic_fluxes])

            ic_fluxes = ic_fluxes[ic_freqs < nu_kn_ic]
            ic_freqs = ic_freqs[ic_freqs < nu_kn_ic]

            ic_flux_interp = interp1d(np.log10(ic_freqs), np.log10(ic_fluxes))

            for k in range(0, len(total_freqs)):
                if (total_freqs[k] > ic_freqs[0]) and (total_freqs[k] < ic_freqs[-1]):
                    total_nu_Fnu[k] += (10**ic_flux_interp(np.log10(total_freqs[k])) * total_freqs[k])

            total_nu_Fnu = np.concatenate([total_nu_Fnu, (ic_freqs[ic_freqs > np.max(total_freqs)]*ic_fluxes[ic_freqs > np.max(total_freqs)])])
            total_freqs = np.concatenate([total_freqs, ic_freqs[ic_freqs > np.max(total_freqs)]])

    return total_freqs, total_nu_Fnu


def ln_likelihood(params):

    E = 10**params[0]
    rho_ref = 10**params[1]
    p = params[2]
    epsilon_e = 10**params[3]
    epsilon_B = 10**params[4]
    epsilon_p = 1 - epsilon_e - epsilon_B

    # Initial mass and gamma in OBSERVER FRAME (Initial conditions)
    gamma0_obs = ((9*E*2**k)/(1024*np.pi*rho_ref*R_ref**k*c**(5-k)))**(1/(8-(2*k)))*t_obs_min**((k-3)/(8-(2*k)))
    r0_obs = 4*gamma0_obs**2*c*t_obs_min

    R0_obs = np.array([gamma0_obs, r0_obs])  # Initial vector of Lorentz factor and radius of shocked fluid in OBSERVER FRAME

    # Calculate timestep size in different frames

    t_obs_vals = np.geomspace(t_obs_min, t_obs_max, N+1)  # Observer timestep is constant
    h_obs_vals = np.diff(t_obs_vals)

    R_rk4_obs = RK4(R0_obs, F_obs, t_obs_vals, h_obs_vals, E, rho_ref)  # Solve in observer frame
    t_obs_vals = t_obs_vals[:-1]

    gamma_vals = R_rk4_obs[:, 0]  # gamma_{2,1} of shocked fluid at all observer times

    if np.any(gamma_vals < 1):
        return [-np.inf, -np.inf]

    radius_vals = R_rk4_obs[:, 1]  # Radius of shell at all engine times
    rho_1_vals = circumburst_density(t_obs_vals, rho_ref, gamma_vals)  # Density values of circumburst medium at all times

    # OBSERVER FRAME SPECTRUM CALCULATION

    # Implement relativistic shock jump conditions
    rho_2_vals = 4 * gamma_vals * rho_1_vals  # Mass density
    n_2_vals = rho_2_vals / m_p  # Number density
    e_2_vals = 4 * gamma_vals * (gamma_vals-1) * rho_1_vals * c**2  # Energy density

    gamma_m_vals = (epsilon_e*e_2_vals*(p-2)) / (n_2_vals*m_e*c**2*(p-1))  # Minimum Lorentz factor of electrons generated through shock acceleration

    B_vals = np.sqrt(8*np.pi*epsilon_B*e_2_vals)  # Value of magnetic flux density

    nu_m_obs_vals = 2 * gamma_vals * gamma_m_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Synchrotron frequency of electrons with minimum Lorentz factor

    gamma_c_vals = (6*np.pi*m_e*c) / (sigma_t*B_vals**2*2*gamma_vals*t_obs_vals)  # Lorentz factor of fast cooling electrons
    nu_c_obs_vals = 2 * gamma_vals * gamma_c_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_c

    gamma_max_vals = np.sqrt((3*q_e)/(sigma_t*B_vals))  # Maximum Lorentz factor of the synchrotron spectrum
    nu_max_obs_vals = 2 * gamma_vals * gamma_max_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_max

    P_max_obs_vals = (m_e*c**2*sigma_t*2*gamma_vals*B_vals) / (3*q_e)  # Peak power emitted by a single electron
    N_e_obs_vals = (4/3)*np.pi*rho_1_vals*4**3*gamma_vals**6*c**3*t_obs_vals**3 / m_p  # Number of electrons
    F_max_obs_vals = (N_e_obs_vals*P_max_obs_vals) / (4*np.pi*D**2)  # Peak flux emitted by all electrons
    F_max_obs_in_mjy = F_max_obs_vals / 10**(-26)  # Convert to mJy

    # Determine if inverse Compton is going to be relevant and define the Compton Y-param in the
    # fast cooling regime
    if epsilon_e > epsilon_B:
        eta_vals = np.zeros(len(t_obs_vals))
        # Scale cooling break depending on if IC is important
        for i in range(0, len(t_obs_vals)):
            eta_vals[i] = (fsolve(eta_func, 0.5, args=(p, epsilon_e, epsilon_B, gamma_c_vals[i], gamma_m_vals[i]))[0])

        eta_vals[eta_vals > 1] = 1
        Y_vals = np.sqrt(eta_vals*epsilon_e/epsilon_B)
        Y_vals[Y_vals < 1] = 0

        gamma_c_vals = gamma_c_vals / (1+Y_vals)
        nu_c_obs_vals = nu_c_obs_vals / ((1+Y_vals)**2)

        nu_m_obs_ic_vals = 4 * gamma_m_vals**2 * nu_m_obs_vals
        nu_c_obs_ic_vals = 4 * gamma_c_vals**2 * nu_c_obs_vals
        nu_max_obs_ic_vals = 4 * gamma_max_vals**2 * nu_max_obs_vals
        F_max_obs_ic_in_mjy = F_max_obs_in_mjy * (1/3) * sigma_t * (rho_1_vals/m_p) * radius_vals

    elif epsilon_e < epsilon_B:
        eta_vals = np.zeros(len(t_obs_vals))
        Y_vals = np.zeros(len(t_obs_vals))
        nu_m_obs_ic_vals = np.zeros(len(t_obs_vals))
        nu_c_obs_ic_vals = np.zeros(len(t_obs_vals))
        nu_max_obs_ic_vals = np.zeros(len(t_obs_vals))
        F_max_obs_ic_in_mjy = np.zeros(len(t_obs_vals))

    # Proton synchrotron spectrum

    nu_m_p_obs_vals = nu_m_obs_vals * (epsilon_p/epsilon_e)**2 * (m_e/m_p)**3
    nu_c_p_obs_vals = nu_c_obs_vals * (m_p/m_e)**5
    nu_max_p_obs_vals = nu_max_obs_vals * (m_p/m_e)**3
    F_max_p_obs_in_mjy = F_max_obs_in_mjy * (m_e/m_p)

    freq_vals, flux_vals = calc_spectrum(t, Y_vals[t], gamma_vals[t], gamma_m_vals[t], gamma_c_vals[t], nu_c_obs_vals[t],
                                         nu_m_obs_vals[t], nu_max_obs_vals[t], F_max_obs_in_mjy[t], nu_c_obs_ic_vals[t],
                                         nu_m_obs_ic_vals[t], nu_max_obs_ic_vals[t], F_max_obs_ic_in_mjy[t], nu_c_p_obs_vals[t],
                                         nu_m_p_obs_vals[t], nu_max_p_obs_vals[t], F_max_p_obs_in_mjy[t], params)

    # IF TEV EMISSION PRODUCED THEN RETURN A PROB OF 1 AND THE MAX FLUX OF THE TEV EMISSION
    if (np.any(freq_vals > 2.4E26)) and (np.min(flux_vals[freq_vals > 2.4E26]) > 1E15):
        print(params)
        return [0., np.max(flux_vals[freq_vals > 2.4E26]), np.min(flux_vals[freq_vals > 2.4E26])]

    return [-np.inf, -np.inf, -np.inf]


# Prior distribution, just uniform for now
def ln_prior(params):
    if np.min(E_range) < params[0] < np.max(E_range) and np.min(rho_range) < params[1] < np.max(rho_range) and np.min(p_range) < params[2] < np.max(p_range) and np.min(epsilon_e_range) < params[3] < np.max(epsilon_e_range) and np.min(epsilon_B_range) < params[4] < np.max(epsilon_B_range):
        return 0.5
    return -np.inf


# Calculate the posterior distribution using the priors and likelihoods
def ln_posterior(params):
    ln_prior_val = ln_prior(params)
    if not np.isfinite(ln_prior_val):
        return -np.inf, -np.inf, -np.inf
    ln_likelihood_val = ln_likelihood(params)
    if not np.isfinite(ln_likelihood_val[0]):
        return -np.inf, -np.inf, -np.inf
    return ln_prior_val + ln_likelihood_val[0], ln_likelihood_val[1], ln_likelihood_val[2]


N = 1000  # Number of timesteps
t = N - 1  # Timestep of interest
t_obs_min = 0.001  # ENGINE FRAME time at start of simulation in s
# t_obs_max = 10000  # ENGINE FRAME time at end of simulation in s
t_obs_max = 10  # The desired time at which to check the strength of TeV emission

E = 51  # Energy injected into GRB in erg
rho_ref = -23.78  # Reference circumburst medium density in g/cm3
R_ref = 1E19  # Reference radius used to set density profile
k = 0  # 0 for ISM, 2 for stellar wind, other values invalid
p = 2.2  # Slope of electron population spectrum
epsilon_e = -1  # Ignorance parameter governing energy fraction put into electrons
epsilon_B = -2  # Ignorance parameter governing energy fraction put into magnetic field
epsilon_p = 1 - epsilon_e - epsilon_B  # Ignorance parameter governing energy fraction put into protons
m_e = 9.11E-28  # Electron mass in g
q_e = 4.803E-10  # Electron charge in Fr
sigma_t = 6.6525E-25  # Thomson scattering cross section in cm2
h = 6.63E-27  # Planck's constant in erg s
c = 2.998E10  # Speed of light in cm/s
m_p = 1.67E-24  # Proton mass in g/cm3
D = 1E28  # Distance to GRB (approx 1 Gly in cm)

initial_params = np.array([E, rho_ref, p, epsilon_e, epsilon_B])  # E, rho_ref, p, epsilon_e, epsilon_B

# Define physically valid ranges of parameters to be varied
n = 100
E_range = np.linspace(45, 57, n)
rho_range = np.linspace(-25.78, -21.78, n)  # Number density of 1E-2 to 1E2
p_range = np.linspace(2, 5, n)
epsilon_e_range = np.linspace(-5, 0, n)
epsilon_B_range = np.linspace(-5, 0, n)

param_ranges = np.array([np.max(E_range)-np.min(E_range), np.max(rho_range)-np.min(rho_range),
                         np.max(p_range)-np.min(p_range), np.max(epsilon_e_range)-np.min(epsilon_e_range),
                         np.max(epsilon_B_range)-np.min(epsilon_B_range)])

ndim = 5
nwalkers = 32
nsteps = 10000

initial_positions = [initial_params + (1e-2 * np.random.randn(ndim) * param_ranges) for i in range(nwalkers)]

# Setup the backend to ensure data is consistently logged
# Good backup to have in case the code crashes halfway through for example
filename = "tev_emission_10s.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool(32) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, backend=backend, pool=pool)
    sampler.run_mcmc(initial_positions, nsteps, store = True, **{'skip_initial_state_check':True})





















