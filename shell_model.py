# TOMORROW (27/10) - LOOK AT EIC + PROTON SYNC
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

writergif = animation.PillowWriter(fps=60)

# Define constants and initial conditions

N = 1000  # Number of timesteps
t_engine_min = 24*60*60  # ENGINE FRAME time at start of simulation in s
t_engine_max = 30*24*60*60  # ENGINE FRAME time at end of simulation in s

E = 8.45931341e+56  # Energy injected into GRB in erg
rho_ref = 1.08739229e-22  # Reference circumburst medium density in g/cm3
R_ref = 1E19  # Reference radius used to set density profile
k = 0  # 0 for ISM, 2 for stellar wind, other values invalid
p = 2.99484197e+00  # Slope of electron population spectrum
epsilon_e = 6.19891030e-01  # Ignorance parameter governing energy fraction put into electrons
epsilon_B = 5.28411826e-01  # Ignorance parameter governing energy fraction put into magnetic field
epsilon_p = 1 - epsilon_e - epsilon_B  # Ignorance parameter governing energy fraction put into protons
m_e = 9.11E-28  # Electron mass in g
q_e = 4.803E-10  # Electron charge in Fr
sigma_t = 6.6525E-25  # Thomson scattering cross section in cm2
h = 6.63E-27  # Planck's constant in erg s
c = 2.998E10  # Speed of light in cm/s
m_p = 1.67E-24  # Proton mass in g/cm3
D = 1E28  # Distance to GRB (approx 1 Gly in cm)


def plot_spectrum(t, Y, gamma, gamma_m, gamma_c, nu_c, nu_m, nu_max, F_max, nu_c_ic, nu_m_ic, nu_max_ic, F_max_ic, nu_c_p, nu_m_p, nu_max_p, F_max_p):
    plt.clf()
    start = time.time()

    print(gamma, gamma_m, gamma_c, nu_m, nu_c)

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

    fluxes = fluxes[freqs < nu_max]
    freqs = freqs[freqs < nu_max]

    nu_Fnu = freqs * fluxes

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

    print(time.time()-start)

    plt.plot(total_freqs, total_nu_Fnu, color='blue', label='Total')
    plt.plot(freqs, freqs*fluxes, '--', color='red', label='Electron sync')
    plt.plot(proton_freqs, proton_freqs*proton_fluxes, '--', color='green', label='Proton sync')

    if epsilon_e > epsilon_B:
        if Y >= 1:
            plt.plot(ic_freqs, ic_freqs*ic_fluxes, '--', color='purple', label='SSC')

    plt.title(f't = {t_obs_vals[t]:.0f} s')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\nu$ [Hz]')
    plt.ylabel(r'$\nu$F$_\nu$ [mJy Hz]')
    plt.legend()

    return plt


def animate(t):
    plot_spectrum(t, Y_vals[t], gamma_vals[t], gamma_m_vals[t], gamma_c_vals[t], nu_c_obs_vals[t],
                  nu_m_obs_vals[t], nu_max_obs_vals[t], F_max_obs_in_mjy[t], nu_c_obs_ic_vals[t],
                  nu_m_obs_ic_vals[t], nu_max_obs_ic_vals[t], F_max_obs_ic_in_mjy[t], nu_c_p_obs_vals[t],
                  nu_m_p_obs_vals[t], nu_max_p_obs_vals[t], F_max_p_obs_in_mjy[t])


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


# Colour fader for plotting
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


c1 = 'blue'
c2 = 'red'


# Define a function to calculate the vector F = dR/dt = (dM/dt, d(gamma)/dt) in ENGINE FRAME
# Accounts for the case of a stellar wind

def F_engine(R, t):
    dgammadt = ((k-3)/2) * np.sqrt((9*E)/(16*np.pi*rho_ref*R_ref**k*c**(5-k)))*t**((k-5)/2)
    dMdt = (4*np.pi*(3-k)*rho_ref*R_ref**k*c**(3-k)*t**(2-k)) / 3
    dRdt = c

    return np.array([dgammadt, dMdt, dRdt])


# Define a function to perform the RK4 algorithm, allowing for the possibility that the timestep changes
# which does occur due to SR effects when converting between engine and observer frames

def RK4(R, F, t_vals, h):
    # Set initial value and define empty list within which to store values of R
    R0 = R
    R_vals = []

    # Loop over all time values, calculating vectors of k1-k4 coefficients at each
    # time step t, and use these to update R0 to the next vector R1 at time t+h
    # using the RK4 method
    for i in range(0, len(t_vals)):
        k1 = h*F(R0, t_vals[i])
        k2 = h*F(R0 + k1/2, t_vals[i] + h/2)
        k3 = h*F(R0 + k2/2, t_vals[i] + h/2)
        k4 = h*F(R0 + k3, t_vals[i] + h)
        R1 = R0 + (k1 + 2*k2 + 2*k3 + k4)/6

        # Store and reset for next time step
        R_vals.append(R1)
        R0 = R1

    # Convert list of R vectors into array of R vectors; easier to select
    # the elements of R to plot
    R_vals = np.array(R_vals)
    return R_vals


# Initial mass and gamma in ENGINE FRAME (Initial conditions)
gamma0_engine = np.sqrt((9*E)/(16*np.pi*rho_ref*R_ref**k*c**(5-k)))*t_engine_min**((k-3)/2)
m0_engine = (4/3)*np.pi*(3-k)*rho_ref*R_ref**k*c**(3-k)*t_engine_min**(3-k)
r0_engine = c*t_engine_min

R0_engine = np.array([gamma0_engine, m0_engine, r0_engine])  # Initial vector of mass and Lorentz factor of shocked fluid in ENGINE FRAME

# Calculate timestep size in different frames

t_engine_vals = np.linspace(t_engine_min, t_engine_max, N)  # Engine timestep is constant
h_engine = (t_engine_max - t_engine_min) / N

R_rk4_engine = RK4(R0_engine, F_engine, t_engine_vals, h_engine)  # Solve in engine frame

gamma_vals = R_rk4_engine[:, 0]  # gamma_{2,1} of shocked fluid at all engine times
mass_vals = R_rk4_engine[:, 1]  # Swept up mass contained in shell at all engine times
radius_vals = R_rk4_engine[:, 2]  # Radius of shell at all engine times
rho_1_vals = circumburst_density(t_engine_vals)  # Density values of circumburst medium at all times

t_obs_vals = t_engine_vals / (2*gamma_vals**2)  # Work out observer frame times that correspond to engine frame times

# Define linear set of times in observer frame to illustrate time contraction phenomenon when plotting
t_obs_lin_vals = np.linspace(t_obs_vals[0], t_obs_vals[-1], N)

# ENGINE FRAME SPECTRUM CALCULATION

# Implement relativistic shock jump conditions
rho_2_vals = 4 * gamma_vals * rho_1_vals  # Mass density
n_2_vals = rho_2_vals / m_p  # Number density
e_2_vals = 4 * gamma_vals * (gamma_vals-1) * rho_1_vals * c**2  # Energy density

gamma_m_vals = (epsilon_e*e_2_vals*(p-2)) / (n_2_vals*m_e*c**2*(p-1))  # Minimum Lorentz factor of electrons generated through shock acceleration
c_vals = n_2_vals * (p-1) * gamma_m_vals**(p-1)  # Normalisation constant of electron power-law distribution

B_vals = np.sqrt(8*np.pi*epsilon_B*e_2_vals)  # Value of magnetic flux density

nu_m_engine_vals = gamma_m_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Synchrotron frequency of electrons with minimum Lorentz factor

gamma_c_vals = (6*np.pi*m_e*c) / (sigma_t*B_vals**2*2*gamma_vals*t_obs_vals)  # Lorentz factor of fast cooling electrons
nu_c_engine_vals = gamma_c_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_c

gamma_max_vals = np.sqrt((3*q_e)/(sigma_t*B_vals))  # Maximum Lorentz factor of the synchrotron spectrum
nu_max_engine_vals = gamma_max_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_max

# OBSERVER FRAME SPECTRUM CALCULATION

nu_m_obs_vals = 2 * gamma_vals * nu_m_engine_vals  # Synchrotron frequency of electrons with minimum Lorentz factor
nu_c_obs_vals = 2 * gamma_vals * nu_c_engine_vals  # Frequency corresponding to gamma_c
nu_max_obs_vals = 2 * gamma_vals * nu_max_engine_vals  # Frequency corresponding to gamma_max

P_max_obs_vals = (m_e*c**2*sigma_t*2*gamma_vals*B_vals) / (3*q_e)  # Peak power emitted by a single electron
N_e_obs_vals = (4/3)*np.pi*rho_1_vals*c**3*t_engine_vals**3 / m_p  # Number of electrons
F_max_obs_vals = (N_e_obs_vals*P_max_obs_vals) / (4*np.pi*D**2)  # Peak flux emitted by all electrons
F_max_obs_in_mjy = F_max_obs_vals / 10**(-26)  # Convert to mJy

nu_m_obs_spl = CubicSpline(t_obs_vals, nu_m_obs_vals)
nu_c_obs_spl = CubicSpline(t_obs_vals, nu_c_obs_vals)
nu_max_obs_spl = CubicSpline(t_obs_vals, nu_max_obs_vals)

# Determine if inverse Compton is going to be relevant and define the Compton Y-param in the
# fast cooling regime
if epsilon_e > epsilon_B:
    Y_fast = np.sqrt(epsilon_e/epsilon_B)
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

# Proton synchrotron spectrum

nu_m_p_obs_vals = nu_m_obs_vals * (epsilon_p/epsilon_e)**2 * (m_e/m_p)**3
nu_c_p_obs_vals = nu_c_obs_vals * (m_p/m_e)**5
nu_max_p_obs_vals = nu_max_obs_vals * (m_p/m_e)**3
F_max_p_obs_in_mjy = F_max_obs_in_mjy * (m_e/m_p)

nu_m_p_obs_spl = CubicSpline(t_obs_vals, nu_m_p_obs_vals)
nu_c_p_obs_spl = CubicSpline(t_obs_vals, nu_c_p_obs_vals)
nu_max_p_obs_spl = CubicSpline(t_obs_vals, nu_max_p_obs_vals)


# Plotting

'''
# Plot electron power-law spectrum
plt.figure(1)
for i in range(0, len(t_engine_vals)):
    if i % 100000 == 0:
        gamma_e_vals, n_e_vals = electron_population(c_vals[i], gamma_m_vals[i], gamma_max_vals[i])
        plt.plot(gamma_e_vals, n_e_vals, label=f't$_*$ = {t_engine_vals[i]/(24*60*60):.1f} days')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma_e$')
plt.ylabel('n$_e$($\gamma_e$)')
plt.legend()
plt.show()

c_obs_spl = CubicSpline(t_obs_vals, c_vals)
gamma_m_obs_spl = CubicSpline(t_obs_vals, gamma_m_vals)
gamma_max_obs_spl = CubicSpline(t_obs_vals, gamma_max_vals)
plt.figure(2)
for i in range(0, len(t_obs_lin_vals)):
    if i % 100000 == 0:
        gamma_e_vals, n_e_vals = electron_population(c_obs_spl(t_obs_lin_vals[i]), gamma_m_obs_spl(t_obs_lin_vals[i]), gamma_max_obs_spl(t_obs_lin_vals[i]))
        plt.plot(gamma_e_vals, n_e_vals, label=f't = {t_obs_lin_vals[i]/(60):.0f} min')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma_e$')
plt.ylabel('n$_e$($\gamma_e$)')
plt.legend(loc='upper right')
plt.show()
'''

'''
# TEST CODE TO PRODUCE A SPECTRUM AT A PRESCRIBED OBSERVER TIME
plt.figure(3)
low_freq_vals = np.geomspace(1, nu_m_obs_spl(10*60), 10000)
middle_freq_vals = np.geomspace(nu_m_obs_spl(10*60), nu_c_obs_spl(10*60), 10000)
high_freq_vals = np.geomspace(nu_c_obs_spl(10*60), 1E20, 10000)
plt.plot(low_freq_vals, (low_freq_vals/nu_m_obs_spl(10*60))**(1/3)*F_max_obs_in_mjy[-1], label=f't = {60*60/(60*60):.0f} hours')
plt.plot(middle_freq_vals, (middle_freq_vals/nu_m_obs_spl(10*60))**((1-p)/2)*F_max_obs_in_mjy[-1])
plt.plot(high_freq_vals, (nu_c_obs_spl(10*60)/nu_m_obs_spl(10*60))**((1-p)/2)*(high_freq_vals/nu_c_obs_spl(10*60))**(-p/2)*F_max_obs_in_mjy[-1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Flux [mJy]')
plt.show()

with open('spectrum_at_10_minutes.txt', 'w') as f:
    f.write('Frequency [Hz], Flux [mJy]')
    f.write('\n')
    for i in range(0, len(low_freq_vals)):
        f.write('{}, {}'.format(low_freq_vals[i], (low_freq_vals[i]/nu_m_obs_spl(10*60))**(1/3)*F_max_obs_in_mjy[-1]))
        f.write('\n')
    for i in range(0, len(middle_freq_vals)):
        f.write('{}, {}'.format(middle_freq_vals[i], (middle_freq_vals[i]/nu_m_obs_spl(10*60))**((1-p)/2)*F_max_obs_in_mjy[-1]))
        f.write('\n')
    for i in range(0, len(high_freq_vals)):
        f.write('{}, {}'.format(high_freq_vals[i], (nu_c_obs_spl(10*60)/nu_m_obs_spl(10*60))**((1-p)/2)*(high_freq_vals[i]/nu_c_obs_spl(10*60))**(-p/2)*F_max_obs_in_mjy[-1]))
        f.write('\n')
'''

'''
# Plot synchrotron spectrum as function of time

# ENGINE FRAME

plt.figure(4)
for i in range(0, len(t_engine_vals)):
    if i % 50000 == 0:
        # If in fast cooling regime
        if nu_c_engine_vals[i] < nu_m_engine_vals[i]:
            low_freq_vals = np.geomspace(1, nu_c_engine_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_c_engine_vals[i], nu_m_engine_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_m_engine_vals[i], 1E23, 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_c_engine_vals[i])**(1/3))
            middle_freq_fluxes = np.array((middle_freq_vals/nu_c_engine_vals[i])**(-1/2))
            high_freq_fluxes = np.array((nu_m_engine_vals[i]/nu_c_engine_vals[i])**(-1/2)*(high_freq_vals/nu_m_engine_vals[i])**(-p/2))

        # If in slow cooling regime
        if nu_c_engine_vals[i] > nu_m_engine_vals[i]:
            low_freq_vals = np.geomspace(1, nu_m_engine_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_m_engine_vals[i], nu_c_engine_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_c_engine_vals[i], 1E23, 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_m_engine_vals[i])**(1/3))
            middle_freq_fluxes = np.array((middle_freq_vals/nu_m_engine_vals[i])**((1-p)/2))
            high_freq_fluxes = np.array((nu_c_engine_vals[i]/nu_m_engine_vals[i])**((1-p)/2)*(high_freq_vals/nu_c_engine_vals[i])**(-p/2))

        freqs = np.concatenate([low_freq_vals[:-1], middle_freq_vals[:-1], high_freq_vals])
        fluxes = np.concatenate([low_freq_fluxes[:-1], middle_freq_fluxes[:-1], high_freq_fluxes])

        plt.plot(freqs, fluxes, color=colorFader(c1, c2, i/N), label=f't = {t_engine_vals[i]/(24*60*60):.1f} days')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Normalised flux')
plt.legend(loc='lower left')
plt.show()
'''

'''
# OBSERVER FRAME
start = time.time()
plt.figure(5)
for i in range(0, len(t_obs_vals)):
    if i % 10000 == 0:
        # If in fast cooling regime
        if nu_c_obs_vals[i] < nu_m_obs_vals[i]:
            low_freq_vals = np.geomspace(1E10, nu_c_obs_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_c_obs_vals[i], nu_m_obs_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_m_obs_vals[i], nu_max_obs_vals[i], 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_c_obs_vals[i])**(1/3)*F_max_obs_in_mjy[i])
            middle_freq_fluxes = np.array((middle_freq_vals/nu_c_obs_vals[i])**(-1/2)*F_max_obs_in_mjy[i])
            high_freq_fluxes = np.array((nu_m_obs_vals[i]/nu_c_obs_vals[i])**(-1/2)*(high_freq_vals/nu_m_obs_vals[i])**(-p/2)*F_max_obs_in_mjy[i])

            # If IC is relevant, define that part of the spectrum
            if epsilon_e > epsilon_B:
                nu_kn = (2*gamma_vals[i]*m_e*c**2) / (gamma_c_vals[i]*h)  # Frequency of the KN cutoff in the observer frame in the fast cooling regime

                low_freq_ic_vals = np.geomspace(1E10*4*gamma_c_vals[i]**2, nu_c_obs_ic_vals[i], 10000)
                middle_freq_ic_vals = np.geomspace(nu_c_obs_ic_vals[i], nu_m_obs_ic_vals[i], 10000)
                high_freq_ic_vals = np.geomspace(nu_m_obs_ic_vals[i], nu_max_obs_ic_vals[i], 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_c_obs_ic_vals[i])**(1/3)*F_max_obs_ic_in_mjy[i])
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_c_obs_ic_vals[i])**(-1/2)*F_max_obs_ic_in_mjy[i])
                high_freq_ic_fluxes = np.array((nu_m_obs_ic_vals[i]/nu_c_obs_ic_vals[i])**(-1/2)*(high_freq_ic_vals/nu_m_obs_ic_vals[i])**(-p/2)*F_max_obs_ic_in_mjy[i])

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c_vals[i]**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m_vals[i]**2 * nu_kn

        # If in slow cooling regime
        if nu_c_obs_vals[i] > nu_m_obs_vals[i]:
            low_freq_vals = np.geomspace(1E10, nu_m_obs_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_m_obs_vals[i], nu_c_obs_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_c_obs_vals[i], nu_max_obs_vals[i], 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_m_obs_vals[i])**(1/3)*F_max_obs_in_mjy[i])
            middle_freq_fluxes = np.array((middle_freq_vals/nu_m_obs_vals[i])**((1-p)/2)*F_max_obs_in_mjy[i])
            high_freq_fluxes = np.array((nu_c_obs_vals[i]/nu_m_obs_vals[i])**((1-p)/2)*(high_freq_vals/nu_c_obs_vals[i])**(-p/2)*F_max_obs_in_mjy[i])

            # If IC is relevant, define that part of the spectrum
            if Y_vals[i] >= 1:
                nu_kn = (2*gamma_vals[i]*m_e*c**2) / (gamma_m_vals[i]*h)  # Frequency of the KN cutoff in the observer frame in the slow cooling regime

                low_freq_ic_vals = np.geomspace(1E10*4*gamma_m_vals[i]**2, nu_m_obs_ic_vals[i], 10000)
                middle_freq_ic_vals = np.geomspace(nu_m_obs_ic_vals[i], nu_c_obs_ic_vals[i], 10000)
                high_freq_ic_vals = np.geomspace(nu_c_obs_ic_vals[i], nu_max_obs_ic_vals[i], 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_m_obs_ic_vals[i])**(1/3)*F_max_obs_ic_in_mjy[i])
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_m_obs_ic_vals[i])**((1-p)/2)*F_max_obs_ic_in_mjy[i])
                high_freq_ic_fluxes = np.array((nu_c_obs_ic_vals[i]/nu_m_obs_ic_vals[i])**((1-p)/2)*(high_freq_ic_vals/nu_c_obs_ic_vals[i])**(-p/2)*F_max_obs_ic_in_mjy[i])

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m_vals[i]**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c_vals[i]**2 * nu_kn

        freqs = np.concatenate([low_freq_vals[:-1], middle_freq_vals[:-1], high_freq_vals])
        fluxes = np.concatenate([low_freq_fluxes[:-1], middle_freq_fluxes[:-1], high_freq_fluxes])
        nu_Fnu = freqs * fluxes

        if Y_vals[i] >= 1:
            ic_freqs = np.concatenate([low_freq_ic_vals[:-1], middle_freq_ic_vals[:-1], high_freq_ic_vals])
            ic_fluxes = np.concatenate([low_freq_ic_fluxes[:-1], middle_freq_ic_fluxes[:-1], high_freq_ic_fluxes])

            ic_fluxes = ic_fluxes[ic_freqs < nu_kn_ic]
            ic_freqs = ic_freqs[ic_freqs < nu_kn_ic]

            ic_nu_Fnu = ic_freqs * ic_fluxes

            flux_interp = interp1d(np.log10(ic_freqs), np.log10(ic_fluxes))

            for j in range(0, len(freqs)):
                if freqs[j] > ic_freqs[0]:
                    nu_Fnu[j] += (10**flux_interp(np.log10(freqs[j])) * freqs[j])

            total_freqs = np.concatenate([freqs, ic_freqs[ic_freqs>np.max(freqs)]])
            total_nu_Fnu = np.concatenate([nu_Fnu, (ic_freqs[ic_freqs>np.max(freqs)]*ic_fluxes[ic_freqs>np.max(freqs)])])

            plt.plot(total_freqs, total_nu_Fnu, '--', color=colorFader(c1, c2, i/N), label=f't = {t_obs_vals[i]:.0f} s; IC')

        #plt.plot(freqs, fluxes, color=colorFader(c1, c2, i/N), label=f't = {t_obs_vals[i]:.0f} s; Sync')
        #plt.plot(ic_freqs, ic_fluxes, '--', color=colorFader(c1, c2, i/N), label=f't = {t_obs_vals[i]:.0f} s; IC')

        #plt.plot(freqs, fluxes, color=colorFader(c1, c2, i/N), label=f't = {t_obs_vals[i]/(60):.0f} minutes; Sync')


plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Flux [mJy]')
#plt.ylabel(r'Normalised $\nu$F$_\nu$')
#plt.ylabel(r'$\nu$F$_\nu$ [mJy Hz]')
#plt.ylabel(r'$\nu$F$_{\nu}$ / $\nu_c$F$_{\nu_c}$')
plt.legend(loc='lower left')
plt.show()
print(time.time()-start)
'''
'''
# LIGHTCURVES - SAME GRADIENTS REGARDLESS OF F OR NU*F EXCEPT WHEN SYNC AND SSC ARE ADDED TOGETHER

lightcurve_freq = 1E24
lightcurve_times = []
lightcurve_fluxes = []

plt.figure(6)
for i in range(0, len(t_obs_vals)):
    if i % 100 == 0:
        if nu_c_obs_vals[i] < nu_m_obs_vals[i]:
            low_freq_vals = np.geomspace(1E10, nu_c_obs_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_c_obs_vals[i], nu_m_obs_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_m_obs_vals[i], nu_max_obs_vals[i], 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_c_obs_vals[i])**(1/3)*F_max_obs_in_mjy[i])
            middle_freq_fluxes = np.array((middle_freq_vals/nu_c_obs_vals[i])**(-1/2)*F_max_obs_in_mjy[i])
            high_freq_fluxes = np.array((nu_m_obs_vals[i]/nu_c_obs_vals[i])**(-1/2)*(high_freq_vals/nu_m_obs_vals[i])**(-p/2)*F_max_obs_in_mjy[i])

            # If IC is relevant, define that part of the spectrum
            if epsilon_e > epsilon_B:
                nu_kn = (2*gamma_vals[i]*m_e*c**2) / (gamma_c_vals[i]*h)  # Frequency of the KN cutoff in the observer frame in the fast cooling regime

                low_freq_ic_vals = np.geomspace(1E10*4*gamma_c_vals[i]**2, nu_c_obs_ic_vals[i], 10000)
                middle_freq_ic_vals = np.geomspace(nu_c_obs_ic_vals[i], nu_m_obs_ic_vals[i], 10000)
                high_freq_ic_vals = np.geomspace(nu_m_obs_ic_vals[i], nu_max_obs_ic_vals[i], 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_c_obs_ic_vals[i])**(1/3)*F_max_obs_ic_in_mjy[i])
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_c_obs_ic_vals[i])**(-1/2)*F_max_obs_ic_in_mjy[i])
                high_freq_ic_fluxes = np.array((nu_m_obs_ic_vals[i]/nu_c_obs_ic_vals[i])**(-1/2)*(high_freq_ic_vals/nu_m_obs_ic_vals[i])**(-p/2)*F_max_obs_ic_in_mjy[i])

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c_vals[i]**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m_vals[i]**2 * nu_kn

        # If in slow cooling regime
        if nu_c_obs_vals[i] > nu_m_obs_vals[i]:
            low_freq_vals = np.geomspace(1E10, nu_m_obs_vals[i], 10000)
            middle_freq_vals = np.geomspace(nu_m_obs_vals[i], nu_c_obs_vals[i], 10000)
            high_freq_vals = np.geomspace(nu_c_obs_vals[i], nu_max_obs_vals[i], 10000)

            low_freq_fluxes = np.array((low_freq_vals/nu_m_obs_vals[i])**(1/3)*F_max_obs_in_mjy[i])
            middle_freq_fluxes = np.array((middle_freq_vals/nu_m_obs_vals[i])**((1-p)/2)*F_max_obs_in_mjy[i])
            high_freq_fluxes = np.array((nu_c_obs_vals[i]/nu_m_obs_vals[i])**((1-p)/2)*(high_freq_vals/nu_c_obs_vals[i])**(-p/2)*F_max_obs_in_mjy[i])

            # If IC is relevant, define that part of the spectrum
            if Y_vals[i] >= 1:
                nu_kn = (2*gamma_vals[i]*m_e*c**2) / (gamma_m_vals[i]*h)  # Frequency of the KN cutoff in the observer frame in the slow cooling regime

                low_freq_ic_vals = np.geomspace(1E10*4*gamma_m_vals[i]**2, nu_m_obs_ic_vals[i], 10000)
                middle_freq_ic_vals = np.geomspace(nu_m_obs_ic_vals[i], nu_c_obs_ic_vals[i], 10000)
                high_freq_ic_vals = np.geomspace(nu_c_obs_ic_vals[i], nu_max_obs_ic_vals[i], 10000)

                low_freq_ic_fluxes = np.array((low_freq_ic_vals/nu_m_obs_ic_vals[i])**(1/3)*F_max_obs_ic_in_mjy[i])
                middle_freq_ic_fluxes = np.array((middle_freq_ic_vals/nu_m_obs_ic_vals[i])**((1-p)/2)*F_max_obs_ic_in_mjy[i])
                high_freq_ic_fluxes = np.array((nu_c_obs_ic_vals[i]/nu_m_obs_ic_vals[i])**((1-p)/2)*(high_freq_ic_vals/nu_c_obs_ic_vals[i])**(-p/2)*F_max_obs_ic_in_mjy[i])

                # Determine the corresponding frequency of the KN cutoff in the IC spectrum
                if nu_kn < np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_m_vals[i]**2 * nu_kn

                elif nu_kn > np.max(middle_freq_vals):
                    nu_kn_ic = 4 * gamma_c_vals[i]**2 * nu_kn

        freqs = np.concatenate([low_freq_vals[:-1], middle_freq_vals[:-1], high_freq_vals])
        fluxes = np.concatenate([low_freq_fluxes[:-1], middle_freq_fluxes[:-1], high_freq_fluxes])
        nu_Fnu = freqs * fluxes

        if Y_vals[i] >= 1:
            ic_freqs = np.concatenate([low_freq_ic_vals[:-1], middle_freq_ic_vals[:-1], high_freq_ic_vals])
            ic_fluxes = np.concatenate([low_freq_ic_fluxes[:-1], middle_freq_ic_fluxes[:-1], high_freq_ic_fluxes])

            ic_fluxes = ic_fluxes[ic_freqs < nu_kn_ic]
            ic_freqs = ic_freqs[ic_freqs < nu_kn_ic]

            ic_nu_Fnu = ic_freqs * ic_fluxes

            flux_interp = interp1d(np.log10(ic_freqs), np.log10(ic_fluxes))

            for j in range(0, len(freqs)):
                if freqs[j] > ic_freqs[0]:
                    nu_Fnu[j] += (10**flux_interp(np.log10(freqs[j])) * freqs[j])

            total_freqs = np.concatenate([freqs, ic_freqs[ic_freqs>np.max(freqs)]])
            total_nu_Fnu = np.concatenate([nu_Fnu, (ic_freqs[ic_freqs>np.max(freqs)]*ic_fluxes[ic_freqs>np.max(freqs)])])

        nu_Fnu_interp = interp1d(np.log10(total_freqs), np.log10(total_nu_Fnu))

        lightcurve_flux_at_t = flux_interp(np.log10(lightcurve_freq))

        lightcurve_times.append(t_obs_vals[i])
        lightcurve_fluxes.append(10**lightcurve_flux_at_t)

times = np.array(lightcurve_times)
fluxes = np.array(lightcurve_fluxes)

plt.plot(times, fluxes, color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t [s]')
#plt.ylabel(r'$\nu$F$_\nu$($\nu$=10$^{25}$ Hz) [mJy Hz]')
plt.ylabel(r'F$_\nu$($\nu$=10$^{24}$ Hz) [mJy]')
plt.show()

plt.figure(7)
time_diffs = np.array([j-i for i, j in zip(np.log10(times[:-1]), np.log10(times[1:]))])
flux_diffs = np.array([j-i for i, j in zip(np.log10(fluxes[:-1]), np.log10(fluxes[1:]))])
gradients = flux_diffs / time_diffs
plt.plot(times[:-1], gradients)
plt.xscale('log')
plt.xlabel('t [s]')
plt.ylabel('Lightcurve gradient')
plt.show()
'''
'''
with open('lightcurve_1e15.txt', 'w') as f:
    f.write('Observer time [s], Flux [mJy]')
    f.write('\n')
    for i in range(0, len(times)):
        f.write('{}, {}'.format(times[i], fluxes[i]))
        f.write('\n')
'''

'''
fig, ax1 = plt.subplots()
lns1 = ax1.plot(t_engine_vals/(24*60*60), R_rk4_engine[:, 0], label='Engine frame', color='blue')
ax1.set_xlabel('Engine time [days]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('$\gamma$')
ax2 = ax1.twiny()
lns2 = ax2.plot(t_obs_vals/(60), R_rk4_engine[:, 0], label='Observer frame', color='red')
ax2.set_xlabel('Observer time [minutes]')
ax2.set_xscale('log')
ax2.set_yscale('log')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots()
lns1 = ax1.plot(t_engine_vals/(24*60*60), R_rk4_engine[:, 1]/1E27, label='Engine frame', color='blue')
ax1.set_xlabel('Engine time [days]')
ax1.set_ylabel('M [10$^{27}$ g]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2 = ax1.twiny()
lns2 = ax2.plot(t_obs_vals/(60), R_rk4_engine[:, 1]/1E27, label='Observer frame', color='red')
ax2.set_xlabel('Observer time [minutes]')
ax2.set_xscale('log')
ax2.set_yscale('log')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')
fig.tight_layout()
plt.show()
'''

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=len(t_obs_vals), repeat=False)

anim.save("test.gif", writer=writergif)
