# TOMORROW (27/10) - LOOK AT EIC + PROTON SYNC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import chisquare

# Data for attenuation due to pair production in the IGM due to EBL
ebl_data = pd.read_csv('ebl_optical_depth.csv', delimiter=',', skiprows = 1, usecols = [0, 5], names=['E', 'tau'])
ebl_energies = ebl_data['E']
ebl_freqs = ebl_energies * (1.602E-7 / 6.63E-34)
ebl_optical_depths = ebl_data['tau']
ebl_atten_spl = CubicSpline(ebl_freqs, ebl_optical_depths)

# Define constants and initial conditions

N = 1000  # Number of timesteps
t_obs_min = 0.001  # ENGINE FRAME time at start of simulation in s
t_obs_max = 60*60  # ENGINE FRAME time at end of simulation in s

t = N - 1

E = 1E52  # Energy injected into GRB in erg
rho_ref = 1.67E-24  # Reference circumburst medium density in g/cm3
R_ref = 1E19  # Reference radius used to set density profile
k = 0  # 0 for ISM, 2 for stellar wind, other values invalid
p = 2.2  # Slope of electron population spectrum
epsilon_e = 0.3  # Ignorance parameter governing energy fraction put into electrons
epsilon_B = 0.0001  # Ignorance parameter governing energy fraction put into magnetic field
epsilon_p = 1 - epsilon_e - epsilon_B  # Ignorance parameter governing energy fraction put into protons
m_e = 9.11E-28  # Electron mass in g
q_e = 4.803E-10  # Electron charge in Fr
sigma_t = 6.6525E-25  # Thomson scattering cross section in cm2
h = 6.63E-27  # Planck's constant in erg s
c = 2.998E10  # Speed of light in cm/s
m_p = 1.67E-24  # Proton mass in g/cm3
D = 1E28  # Distance to GRB (approx 1 Gly in cm)
A_V = 0.3  # V-band extinction in magnitudes, average

# Extinction parameters for different galaxy types - a, lambda, b, n
mwg = np.array([[165, 0.047, 90, 2], [14, 0.08, 4, 6.5], [0.045, 0.22, -1.95, 2], [0.002, 9.7, -1.95, 2], [0.002, 18, -1.8, 2], [0.012, 25, 0, 2]])
lmc = np.array([[175, 0.046, 90, 2], [19, 0.08, 5.5, 4.5], [0.023, 0.22, -1.95, 2], [0.005, 9.7, -1.95, 2], [0.006, 18, -1.8, 2], [0.02, 25, 0, 2]])
smc = np.array([[185, 0.042, 90, 2], [27, 0.08, 5.5, 4], [0.005, 0.22, -1.95, 2], [0.01, 9.7, -1.95, 2], [0.012, 18, -1.8, 2], [0.03, 25, 0, 2]])

grond_obs = np.array([1.379E14, 1.775E14, 2.143E14, 3.571E14, 3.896E14, 4.762E14, 5.505E14])
bat_obs = np.arange(15E3, 150E3, 7E3) * (1.602E-19/6.63E-34)
xrt_obs = np.arange(0.2E3, 10E3+1, 0.2E3) * (1.602E-19/6.63E-34)
uvot_obs = np.array([5.486E14, 6.831E14, 8.658E14, 1.154E15, 1.336E15, 1.556E15])
gbm_obs = np.concatenate([np.arange(8E3, 1E6, 15E3), np.arange(1E6, 40E6+1, 100E3)]) * (1.602E-19/6.63E-34)
lat_obs = np.concatenate([np.arange(20E6, 1E9+1, 120E6), np.arange(1E9, 10E9+1, 0.85E9), np.arange(10E9, 300E9, 39E9)]) * (1.602E-19/6.63E-34)
hess_obs = np.arange(0.03E12, 100E12, 15E12) * (1.602E-19/6.63E-34)
magic_obs = np.arange(25E9, 30E12, 3E12) * (1.602E-19/6.63E-34)

observer_freqs = np.concatenate([grond_obs, bat_obs, xrt_obs, uvot_obs, gbm_obs, lat_obs, hess_obs, magic_obs])
observer_freqs = np.sort(observer_freqs)


def calc_chi_square(obs, exp):
    total = 0
    for t in range(0, len(obs)):
        total += (obs[t]-exp[t])**2 / exp[t]

    return total


def calc_extinction(galaxy, wavelengths):
    total = 0
    for i in range(0, 6):
        total += galaxy[i, 0] / ((wavelengths/galaxy[i, 1])**galaxy[i, 3] + (galaxy[i, 1]/wavelengths)**galaxy[i, 3] + galaxy[i, 2])

    return total


# Colour fader for plotting
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


c1 = 'blue'
c2 = 'red'


def calc_spectrum(t, Y, gamma, gamma_m, gamma_c, nu_c, nu_m, nu_max, F_max, nu_c_ic, nu_m_ic, nu_max_ic, F_max_ic, nu_c_p, nu_m_p, nu_max_p, F_max_p):

    if nu_c < nu_m:
        low_freq_vals = np.geomspace(nu_c/1000, nu_c, 10000)
        middle_freq_vals = np.geomspace(nu_c, nu_m, 10000)
        high_freq_vals = np.geomspace(nu_m, nu_max, 10000)

        low_freq_fluxes = np.array((low_freq_vals/nu_c)**(1/3)*F_max)
        middle_freq_fluxes = np.array((middle_freq_vals/nu_c)**(-1/2)*F_max)
        high_freq_fluxes = np.array((nu_m/nu_c)**(-1/2)*(high_freq_vals/nu_m)**(-p/2)*F_max)

        # If IC is relevant, define that part of the spectrum
        if epsilon_e > epsilon_B:
            nu_kn = (2*gamma*m_e*c**2) / (gamma_c*h)  # Frequency of the KN cutoff in the observer frame in the fast cooling regime

            low_freq_ic_vals = np.geomspace((nu_c/1000)*4*gamma_c**2, nu_c_ic, 10000)
            middle_freq_ic_vals = np.geomspace(nu_c_ic, nu_m_ic, 10000)
            high_freq_ic_vals = np.geomspace(nu_m_ic, nu_max_ic, 10000)

            low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_c_ic)**(1/3)*F_max_ic)
            middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_c_ic)**(-1/2)*F_max_ic)
            high_freq_ic_fluxes = np.array((nu_m_ic/nu_c_ic)**(-1/2)*(high_freq_ic_vals/nu_m_ic)**(-p/2)*F_max_ic)

            # Determine the corresponding frequency of the KN cutoff in the IC spectrum
            if nu_kn < np.max(middle_freq_vals):
                nu_kn_ic = 4 * gamma_c**2 * nu_kn

            elif nu_kn > np.max(middle_freq_vals):
                nu_kn_ic = 4 * gamma_m**2 * nu_kn

    # If in slow cooling regime
    if nu_c > nu_m:
        low_freq_vals = np.geomspace(nu_m/1000, nu_m, 10000)
        middle_freq_vals = np.geomspace(nu_m, nu_c, 10000)
        high_freq_vals = np.geomspace(nu_c, nu_max, 10000)

        low_freq_fluxes = np.array((low_freq_vals/nu_m)**(1/3)*F_max)
        middle_freq_fluxes = np.array((middle_freq_vals/nu_m)**((1-p)/2)*F_max)
        high_freq_fluxes = np.array((nu_c/nu_m)**((1-p)/2)*(high_freq_vals/nu_c)**(-p/2)*F_max)

        # If IC is relevant, define that part of the spectrum
        if epsilon_e > epsilon_B:
            if Y >= 1:
                nu_kn = (2*gamma*m_e*c**2) / (gamma_m*h)  # Frequency of the KN cutoff in the observer frame in the slow cooling regime

                low_freq_ic_vals = np.geomspace((nu_m/1000)*4*gamma_m**2, nu_m_ic, 10000)
                middle_freq_ic_vals = np.geomspace(nu_m_ic, nu_c_ic, 10000)
                high_freq_ic_vals = np.geomspace(nu_c_ic, nu_max_ic, 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_m_ic)**(1/3)*F_max_ic)
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_m_ic)**((1-p)/2)*F_max_ic)
                high_freq_ic_fluxes = np.array((nu_c_ic/nu_m_ic)**((1-p)/2)*(high_freq_ic_vals/nu_c_ic)**(-p/2)*F_max_ic)

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c**2 * nu_kn

    freqs = np.concatenate([low_freq_vals[:-1], middle_freq_vals[:-1], high_freq_vals])
    fluxes = np.concatenate([low_freq_fluxes[:-1], middle_freq_fluxes[:-1], high_freq_fluxes])

    total_fluxes = fluxes[freqs < nu_max]
    total_freqs = freqs[freqs < nu_max]

    total_nu_Fnu = total_freqs * total_fluxes

    '''
    low_freq_p_vals = np.geomspace(nu_m_p/1000, nu_m_p, 10000)
    middle_freq_p_vals = np.geomspace(nu_m_p, nu_c_p, 10000)
    high_freq_p_vals = np.geomspace(nu_c_p, nu_max_p, 10000)

    low_freq_p_fluxes = np.array((low_freq_p_vals/nu_m_p)**(1/3)*F_max_p)
    middle_freq_p_fluxes = np.array((middle_freq_p_vals/nu_m_p)**((1-p)/2)*F_max_p)
    high_freq_p_fluxes = np.array((nu_c_p/nu_m_p)**((1-p)/2)*(high_freq_p_vals/nu_c_p)**(-p/2)*F_max_p)

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
    '''

    if epsilon_e > epsilon_B:
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


# Define function to plug into f_solve to find a root
def eta_func(eta, gamma_c, gamma_m):
    return eta**(1/(2-p)) + eta**(1/(2*(2-p)))*np.sqrt(epsilon_e/epsilon_B) - (gamma_c/gamma_m)


# Function to produce the shocked electron population according to power law
def electron_population(C, gamma_m, gamma_max):
    gamma_e_vals = np.geomspace(gamma_m, 10*gamma_max, N)
    n_e_vals = C * gamma_e_vals**(-p)
    return gamma_e_vals, n_e_vals


# Function to produce array of circumburst density  values, array of constants in case of ISM (k=0)
def circumburst_density(t):
    rho_1_vals = (rho_ref*R_ref**k) / (c**k*t**k)

    return rho_1_vals


# Define a function to calculate the vector F = dR/dt = (d(gamma)/dt, dr/dt) in OBSERVER FRAME
# Accounts for the case of a stellar wind

def F_obs(R, t):
    dgammadt = ((k-3)/(8-(2*k))) * ((9*E*2**k)/(1024*np.pi*rho_ref*R_ref**k*c**(5-k)))**(1/(8-(2*k)))*t**(((3*k)-11)/(8-(2*k)))
    dRdt = 4*R[0]*c*((2*t*dgammadt)+R[0])

    return np.array([dgammadt, dRdt])


# Define a function to perform the RK4 algorithm, allowing for the possibility that the timestep changes
# which does occur due to SR effects when converting between engine and observer frames

def RK4(R, F, t_vals, h):
    # Set initial value and define empty list within which to store values of R
    R0 = R
    R_vals = []

    # Loop over all time values, calculating vectors of k1-k4 coefficients at each
    # time step t, and use these to update R0 to the next vector R1 at time t+h
    # using the RK4 method
    for i in range(0, len(t_vals)-1):
        k1 = h[i]*F(R0, t_vals[i])
        k2 = h[i]*F(R0 + k1/2, t_vals[i] + h[i]/2)
        k3 = h[i]*F(R0 + k2/2, t_vals[i] + h[i]/2)
        k4 = h[i]*F(R0 + k3, t_vals[i] + h[i])
        R1 = R0 + (k1 + 2*k2 + 2*k3 + k4)/6

        # Store and reset for next time step
        R_vals.append(R1)
        R0 = R1

    # Convert list of R vectors into array of R vectors; easier to select
    # the elements of R to plot
    R_vals = np.array(R_vals)
    return R_vals


# Initial mass and gamma in OBSERVER FRAME (Initial conditions)
gamma0_obs = ((9*E*2**k)/(1024*np.pi*rho_ref*R_ref**k*c**(5-k)))**(1/(8-(2*k)))*t_obs_min**((k-3)/(8-(2*k)))
r0_obs = 4*gamma0_obs**2*c*t_obs_min

R0_obs = np.array([gamma0_obs, r0_obs])  # Initial vector of Lorentz factor and radius of shocked fluid in OBSERVER FRAME

# Calculate timestep size in different frames

t_obs_vals = np.geomspace(t_obs_min, t_obs_max, N+1)  # Observer timestep is constant
h_obs_vals = np.diff(t_obs_vals)

R_rk4_obs = RK4(R0_obs, F_obs, t_obs_vals, h_obs_vals)  # Solve in observer frame
t_obs_vals = t_obs_vals[:-1]

gamma_vals = R_rk4_obs[:, 0]  # gamma_{2,1} of shocked fluid at all observer times

radius_vals = R_rk4_obs[:, 1]  # Radius of shell at all engine times
rho_1_vals = circumburst_density(t_obs_vals)  # Density values of circumburst medium at all times

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
        eta_vals[i] = (fsolve(eta_func, 0.5, args=(gamma_c_vals[i], gamma_m_vals[i]))[0])

    eta_vals[eta_vals > 1] = 1
    Y_vals = np.sqrt(eta_vals*epsilon_e/epsilon_B)
    Y_vals[Y_vals < 1] = 0

    gamma_c_vals = gamma_c_vals / (1+Y_vals)
    nu_c_obs_vals = nu_c_obs_vals / ((1+Y_vals)**2)

    nu_m_obs_ic_vals = 4 * gamma_m_vals**2 * nu_m_obs_vals
    nu_c_obs_ic_vals = 4 * gamma_c_vals**2 * nu_c_obs_vals
    nu_max_obs_ic_vals = 4 * gamma_max_vals**2 * nu_max_obs_vals
    F_max_obs_ic_in_mjy = F_max_obs_in_mjy * (1/3) * sigma_t * (rho_1_vals/m_p) * radius_vals

    nu_m_obs_ic_spl = CubicSpline(t_obs_vals, nu_m_obs_ic_vals)
    nu_c_obs_ic_spl = CubicSpline(t_obs_vals, nu_c_obs_ic_vals)
    nu_max_obs_ic_spl = CubicSpline(t_obs_vals, nu_max_obs_ic_vals)

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

nu_m_obs_spl = CubicSpline(t_obs_vals, nu_m_obs_vals)
nu_c_obs_spl = CubicSpline(t_obs_vals, nu_c_obs_vals)
nu_max_obs_spl = CubicSpline(t_obs_vals, nu_max_obs_vals)

nu_m_p_obs_spl = CubicSpline(t_obs_vals, nu_m_p_obs_vals)
nu_c_p_obs_spl = CubicSpline(t_obs_vals, nu_c_p_obs_vals)
nu_max_p_obs_spl = CubicSpline(t_obs_vals, nu_max_p_obs_vals)

freq_vals, flux_vals = calc_spectrum(t, Y_vals[t], gamma_vals[t], gamma_m_vals[t], gamma_c_vals[t], nu_c_obs_vals[t],
                                     nu_m_obs_vals[t], nu_max_obs_vals[t], F_max_obs_in_mjy[t], nu_c_obs_ic_vals[t],
                                     nu_m_obs_ic_vals[t], nu_max_obs_ic_vals[t], F_max_obs_ic_in_mjy[t], nu_c_p_obs_vals[t],
                                     nu_m_p_obs_vals[t], nu_max_p_obs_vals[t], F_max_p_obs_in_mjy[t])

wavelength_vals = c / freq_vals
v_band_extinction = calc_extinction(smc, 0.551)
extinction_func = calc_extinction(smc, wavelength_vals)

extinction_in_mag = (extinction_func/v_band_extinction) * A_V
optical_depth = extinction_in_mag/1.086
flux_obs_vals = flux_vals * np.e**-optical_depth

flux_obs_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))] = flux_obs_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))] * np.e**(-ebl_atten_spl(freq_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))]))
#flux_obs_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))] = flux_obs_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))] * ((1-np.e**(-ebl_atten_spl(freq_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))])))/ebl_atten_spl(freq_vals[(freq_vals>min(ebl_freqs)) & (freq_vals<max(ebl_freqs))]))
flux_obs_vals[freq_vals>max(ebl_freqs)] = 0

flux_obs_interp = CubicSpline(freq_vals, flux_obs_vals*1E-26)

observer_interp = CubicSpline(freq_vals, flux_obs_vals*1E-26)
observer_freqs = observer_freqs[observer_freqs < np.max(freq_vals)]
observer_fluxes = observer_interp(observer_freqs)
observer_flux_errors = 0.1 * observer_fluxes
shifts = np.random.normal(0, observer_flux_errors)
observer_fluxes += shifts

observed_flux = flux_obs_interp(observer_freqs)

chi_square_test_statistic = calc_chi_square(observed_flux, observer_fluxes)
print(chi_square_test_statistic)

plt.plot(freq_vals, flux_vals*1E-26, label='Ideal', color='green')
plt.plot(freq_vals, flux_obs_vals*1E-26, label='Observed')
plt.errorbar(observer_freqs, observer_fluxes, yerr=observer_flux_errors, linestyle='none', marker='o', color='red', markersize=1, capsize=5)

plt.vlines(min(bat_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'blue', '--')
plt.vlines(max(bat_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'blue', '--')
plt.annotate('BAT', (min(bat_obs)*1.2, max(flux_vals*1E-26)*1.5), color='blue')

plt.vlines(min(xrt_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'green', '--')
plt.vlines(max(xrt_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'green', '--')
plt.annotate('XRT', (min(xrt_obs)*1.2, max(flux_vals*1E-26)*1.5), color='green')

plt.vlines(min(uvot_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'black', '--')
plt.vlines(max(uvot_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'black', '--')
plt.annotate('UVOT', (max(uvot_obs)*1.2, max(flux_vals*1E-26)*1.5))

plt.vlines(min(grond_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'red', '--')
plt.vlines(max(grond_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'red', '--')
plt.annotate('GROND', (min(grond_obs)/10, max(flux_vals*1E-26)*1.5), color='red')

plt.vlines(min(gbm_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'pink', '--')
plt.vlines(max(gbm_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'pink', '--')
plt.annotate('GBM', (max(gbm_obs)/10, max(flux_vals*1E-26)*1.5), color='pink')

plt.vlines(min(lat_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'purple', '--')
plt.vlines(max(lat_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'purple', '--')
plt.annotate('LAT', (max(lat_obs)/8, max(flux_vals*1E-26)*1.5), color='purple')

plt.vlines(min(hess_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'grey', '--')
plt.vlines(max(hess_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'grey', '--')
plt.annotate('HESS', (max(hess_obs)*1.2, max(flux_vals*1E-26)*1.5), color='grey')

plt.vlines(min(magic_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'brown', '--')
plt.vlines(max(magic_obs), min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10, 'brown', '--')
plt.annotate('MAGIC', (max(magic_obs)/10, max(flux_vals*1E-26)*1.5), color='brown')

plt.xscale('log')
plt.yscale('log')
plt.ylim(min(flux_vals*1E-26)/10, max(flux_vals*1E-26)*10)
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel(r'$\nu$F$_\nu$ [erg cm$^{-2}$ s$^{-1}$]')
plt.legend(loc='upper left')



