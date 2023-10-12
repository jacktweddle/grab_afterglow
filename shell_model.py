# TOMORROW (12/10) START WITH GETTING RID OF THE SECOND SOLVE AND MAKING SURE EVERYTHING ELSE LINES UP
# THEN GET LIGHTCURVES AND PLOT THE ACTUAL SLOPES; COMPARE WITH AFTERGLOWPY
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline

# Define constants and initial conditions

N = 500000  # Number of timesteps
t_engine_min = 24*60*60  # ENGINE FRAME time at start of simulation
t_engine_max = 30*24*60*60  # ENGINE FRAME time at end of simulation

E = 1E51  # Energy injected into GRB in erg
rho_1 = 1.67E-24  # Density of ISM in g/cm3 // Also used as proton mass
p = 2.2  # Slope of electron population spectrum
epsilon_e = 0.1  # Ignorance parameter governing energy fraction put into electrons
epsilon_B = 0.01  # Ignorance parameter governing energy fraction put into magnetic field
m_e = 9.11E-28  # Electron mass in g
q_e = 4.803E-10  # Electron charge in Fr
sigma_t = 6.6525E-25  # Thomson scattering cross section in cm2
c = 3E10  # Speed of light in cm/s
D = 1E18  # Distance to GRB (approx 1 Gly in cm)


# Function to produce the shocked electron population according to power law
def electron_population(C, gamma_m):
    gamma_e_vals = np.linspace(gamma_m, 10*gamma_m, N)
    n_e_vals = C * gamma_e_vals**(-p)
    return gamma_e_vals, n_e_vals


# Colour fader for plotting
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


c1 = 'blue'
c2 = 'red'


# Define a function to calculate the vector F = dR/dt = (dM/dt, d(gamma)/dt) in ENGINE FRAME
# CURRENTLY ASSUMING NO INITIAL EJECTED MASS AND NO STELLAR WIND

def F_engine(R, t):
    dgammadt = -(9/8)*np.sqrt(E/(rho_1*np.pi*c**5))*t**(-5/2)
    dMdt = 4*np.pi*rho_1*c**3*t**2

    return np.array([dgammadt, dMdt])


# Define a function to calculate the vector F = dR/dt = (dM/dt, d(gamma)/dt) in OBSERVER FRAME
# CURRENTLY ASSUMING NO INITIAL EJECTED MASS AND NO STELLAR WIND
# IS ANY APPEARANCE OF GAMMA IN THESE EQUATIONS OBSERVER FRAME GAMMA OR ENGINE FRAME GAMMA?

def F_obs(R, t):
    dgammadt = -(3/8)*((9*E)/(16*np.pi*rho_1*c**5))**(1/8)*t**(-11/8)
    dMdt = 4*np.pi*rho_1*R[0]**6*c**3*t**2 + 8*np.pi*rho_1*R[0]**5*c**3*t**3*dgammadt

    return np.array([dgammadt, dMdt])


# Define a function to perform the RK4 algorithm, allowing for the possibility that the timestep changes
# which does occur due to SR effects when converting between engine and observer frames

def RK4(R, F, t_vals, h_vals):
    # Set initial value and define empty list within which to store values of R
    R0 = R
    R_vals = []

    # Loop over all time values, calculating vectors of k1-k4 coefficients at each
    # time step t, and use these to update R0 to the next vector R1 at time t+h
    # using the RK4 method
    for i in range(0, len(t_vals)):
        k1 = h_vals[i]*F(R0, t_vals[i])
        k2 = h_vals[i]*F(R0 + k1/2, t_vals[i] + h_vals[i]/2)
        k3 = h_vals[i]*F(R0 + k2/2, t_vals[i] + h_vals[i]/2)
        k4 = h_vals[i]*F(R0 + k3, t_vals[i] + h_vals[i])
        R1 = R0 + (k1 + 2*k2 + 2*k3 + k4)/6

        # Store and reset for next time step
        R_vals.append(R1)
        R0 = R1

    # Convert list of R vectors into array of R vectors; easier to select
    # the elements of R to plot
    R_vals = np.array(R_vals)
    return R_vals


# Initial mass and gamma in ENGINE FRAME
gamma0_engine = (3/4)*np.sqrt(E/(rho_1*np.pi*c**5))*t_engine_min**(-3/2)
m0_engine = (4/3)*np.pi*rho_1*c**3*t_engine_min**3

t_obs_min = t_engine_min / (gamma0_engine**2)  # OBSERVER FRAME time at start

# Initial mass and gamma in OBSERVER FRAME
gamma0_obs = ((9*E)/(16*np.pi*rho_1*c**5))**(1/8)*t_obs_min**(-3/8)  # SHOULD BE SAME VALUE AS ENGINE FRAME
m0_obs = (4/3)*np.pi*rho_1*gamma0_obs**6*c**3*t_obs_min**3  # SHOULD BE SAME VALUE AS ENGINE FRAME

R0_engine = np.array([gamma0_engine, m0_engine])  # Initial vector of mass and Lorentz factor of shocked fluid in ENGINE FRAME
R0_obs = np.array([gamma0_obs, m0_obs])  # Initial vector of mass and Lorentz factor of shocked fluid in OBSERVER FRAME

# AGAIN ENGINE AND OBSERVER R0 SHOULD BE THE SAME

# Calculate timestep size in different frames

t_engine_vals = np.linspace(t_engine_min, t_engine_max, N)  # Engine timestep is constant
h_engine = (t_engine_max - t_engine_min) / N
h_engine_vals = len(t_engine_vals)*[h_engine]

R_rk4_engine = RK4(R0_engine, F_engine, t_engine_vals, h_engine_vals)  # Solve in engine frame

gamma_engine_vals = R_rk4_engine[:, 0]  # gamma_{2,1} of shocked fluid at all engine times
mass_engine_vals = R_rk4_engine[:, 1]  # Swept up mass contained in shell at all engine times

t_obs_vals = t_engine_vals / (gamma_engine_vals**2)  # Work out observer frame times that correspond to engine frame times
h_obs_vals = [j-i for i, j in zip(t_obs_vals[:-1], t_obs_vals[1:])]  # Work out RK4 timestep - changes with every timestep
# For solver to not crash, work out extra timestep
t_obs_extra_val = (t_engine_vals[-1]+h_engine) / (gamma_engine_vals[-1] - (9/8)*np.sqrt(E/(rho_1*np.pi*c**5))*(t_engine_vals[-1]+h_engine)**(-5/2)*h_engine)**2
h_obs_vals.append(t_obs_extra_val-t_obs_vals[-1])

# Define linear set of times in observer frame to illustrate time contraction phenomenon when plotting
t_obs_lin_vals = np.linspace(t_obs_vals[0], t_obs_vals[-1], N)

R_rk4_obs = RK4(R0_obs, F_obs, t_obs_vals, h_obs_vals)  # Solve in observer frame

gamma_obs_vals = R_rk4_obs[:, 0]  # gamma_{2,1} of shocked fluid at all observer times
mass_obs_vals = R_rk4_obs[:, 1]  # Swept up mass contained in shell at all observer times

# THESE SHOULD BE THE SAME AS THE ENGINE FRAME VALUES TO WITHIN NUMERICAL ERROR
# SINCE THE TIMES HAVE BEEN APPROPRIATELY SCALED

# ENGINE FRAME SPECTRUM CALCULATION

# Implement relativistic shock jump conditions
rho_2_engine_vals = 4 * gamma_engine_vals * rho_1  # Mass density
n_2_engine_vals = rho_2_engine_vals / rho_1  # Number density
e_2_engine_vals = 4 * gamma_engine_vals * (gamma_engine_vals-1) * rho_1 * c**2  # Energy density

gamma_m_engine_vals = (epsilon_e*e_2_engine_vals*(p-2)) / (n_2_engine_vals*m_e*c**2*(p-1))  # Minimum Lorentz factor of electrons generated through shock acceleration
c_engine_vals = n_2_engine_vals * (p-1) * gamma_m_engine_vals**(p-1)  # Normalisation constant of electron power-law distribution

B_engine_vals = np.sqrt(8*np.pi*epsilon_B*e_2_engine_vals)  # Value of magnetic flux density

nu_m_engine_vals = gamma_m_engine_vals**2 * (q_e*B_engine_vals)/(2*np.pi*m_e*c)  # Synchrotron frequency of electrons with minimum Lorentz factor

gamma_c_engine_vals = (6*np.pi*m_e*c*gamma_engine_vals) / (sigma_t*B_engine_vals**2*t_engine_vals)  # Lorentz factor of fast cooling electrons
nu_c_engine_vals = gamma_c_engine_vals**2 * (q_e*B_engine_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_c

# OBSERVER FRAME SPECTRUM CALCULATION

# Implement relativistic shock jump conditions
rho_2_obs_vals = 4 * gamma_obs_vals * rho_1  # Mass density
n_2_obs_vals = rho_2_obs_vals / rho_1  # Number density
e_2_obs_vals = 4 * gamma_obs_vals * (gamma_obs_vals-1) * rho_1 * c**2  # Energy density

gamma_m_obs_vals = (epsilon_e*e_2_obs_vals*(p-2)) / (n_2_obs_vals*m_e*c**2*(p-1))  # Minimum Lorentz factor of electrons generated through shock acceleration
c_obs_vals = n_2_obs_vals * (p-1) * gamma_m_obs_vals**(p-1)  # Normalisation constant of electron power-law distribution

B_obs_vals = np.sqrt(8*np.pi*epsilon_B*e_2_obs_vals)  # Value of magnetic flux density

nu_m_obs_vals = gamma_obs_vals * gamma_m_obs_vals**2 * (q_e*B_obs_vals)/(2*np.pi*m_e*c)  # Synchrotron frequency of electrons with minimum Lorentz factor

gamma_c_obs_vals = (6*np.pi*m_e*c) / (sigma_t*B_obs_vals**2*gamma_obs_vals*t_obs_vals)  # Lorentz factor of fast cooling electrons
nu_c_obs_vals = gamma_obs_vals * gamma_c_obs_vals**2 * (q_e*B_obs_vals)/(2*np.pi*m_e*c)  # Frequency corresponding to gamma_c

P_max_obs_vals = (m_e*c**2*sigma_t*gamma_obs_vals*B_obs_vals) / (3*q_e)
N_e_obs_vals = (4/3)*np.pi*gamma_obs_vals**6*c**3*t_obs_vals**3
F_max_obs_vals = (N_e_obs_vals*P_max_obs_vals) / (4*np.pi*D**2)

# Plotting
'''
# Plot electron power-law spectrum
plt.figure(1)
for i in range(0, len(t_engine_vals)):
    if i % 100000 == 0:
        gamma_e_engine_vals, n_e_engine_vals = electron_population(c_engine_vals[i], gamma_m_engine_vals[i])
        plt.plot(gamma_e_engine_vals, n_e_engine_vals, label=f't$_*$ = {t_engine_vals[i]/(24*60*60):.1f} days')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma_e$')
plt.ylabel('n$_e$($\gamma_e$)')
plt.legend()
plt.show()

c_obs_spl = CubicSpline(t_obs_vals, c_obs_vals)
gamma_m_obs_spl = CubicSpline(t_obs_vals, gamma_m_obs_vals)
plt.figure(2)
for i in range(0, len(t_obs_lin_vals)):
    if i % 100000 == 0:
        gamma_e_obs_vals, n_e_obs_vals = electron_population(c_obs_spl(t_obs_lin_vals[i]), gamma_m_obs_spl(t_obs_lin_vals[i]))
        plt.plot(gamma_e_obs_vals, n_e_obs_vals, label=f't = {t_obs_lin_vals[i]/(60):.0f} min')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma_e$')
plt.ylabel('n$_e$($\gamma_e$)')
plt.legend(loc='upper right')
plt.show()
'''

# FIGURE OUT HOW TO PLOT THESE NICELY AS TIME EVOLVES

plt.figure(3)
for i in range(0, len(t_engine_vals)):
    if i % 50000 == 0:
        # If in fast cooling regime
        if nu_c_engine_vals[i] < nu_m_engine_vals[i]:
            low_freq_vals = np.linspace(nu_c_engine_vals[i]/10, nu_c_engine_vals[i], 10000)
            middle_freq_vals = np.linspace(nu_c_engine_vals[i], nu_m_engine_vals[i], 10000)
            high_freq_vals = np.linspace(nu_m_engine_vals[i], 10*nu_m_engine_vals[i], 10000)

            plt.plot(low_freq_vals, (low_freq_vals/nu_c_engine_vals[i])**(1/3), color=colorFader(c1, c2, i/N), label=f't$_*$ = {t_engine_vals[i]/(24*60*60):.1f} days')
            plt.plot(middle_freq_vals, (middle_freq_vals/nu_c_engine_vals[i])**(-1/2), color=colorFader(c1, c2, i/N))
            plt.plot(high_freq_vals, (nu_m_engine_vals[i]/nu_c_engine_vals[i])**(-1/2)*(high_freq_vals/nu_m_engine_vals[i])**(-p/2), color=colorFader(c1, c2, i/N))

        # If in slow cooling regime
        if nu_c_engine_vals[i] > nu_m_engine_vals[i]:
            low_freq_vals = np.linspace(nu_m_engine_vals[i]/10, nu_m_engine_vals[i], 10000)
            middle_freq_vals = np.linspace(nu_m_engine_vals[i], nu_c_engine_vals[i], 10000)
            high_freq_vals = np.linspace(nu_c_engine_vals[i], 10*nu_c_engine_vals[i], 10000)

            plt.plot(low_freq_vals, (low_freq_vals/nu_m_engine_vals[i])**(1/3), color=colorFader(c1, c2, i/N), label=f't$_*$ = {t_engine_vals[i]/(24*60*60):.1f} days')
            plt.plot(middle_freq_vals, (middle_freq_vals/nu_m_engine_vals[i])**((1-p)/2), color=colorFader(c1, c2, i/N))
            plt.plot(high_freq_vals, (nu_c_engine_vals[i]/nu_m_engine_vals[i])**((1-p)/2)*(high_freq_vals/nu_c_engine_vals[i])**(-p/2), color=colorFader(c1, c2, i/N))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Normalised flux')
plt.legend(loc='lower left')
plt.show()

nu_m_obs_spl = CubicSpline(t_obs_vals, nu_m_obs_vals)
nu_c_obs_spl = CubicSpline(t_obs_vals, nu_c_obs_vals)
plt.figure(4)
for i in range(0, len(t_obs_lin_vals)):
    if i % 50000 == 0:
        # If in fast cooling regime
        if nu_c_obs_spl(t_obs_lin_vals[i]) < nu_m_obs_spl(t_obs_lin_vals[i]):
            low_freq_vals = np.linspace(nu_c_obs_spl(t_obs_lin_vals[i])/10, nu_c_obs_spl(t_obs_lin_vals[i]), 10000)
            middle_freq_vals = np.linspace(nu_c_obs_spl(t_obs_lin_vals[i]), nu_m_obs_spl(t_obs_lin_vals[i]), 10000)
            high_freq_vals = np.linspace(nu_m_obs_spl(t_obs_lin_vals[i]), 10*nu_m_obs_spl(t_obs_lin_vals[i]), 10000)

            plt.plot(low_freq_vals, (low_freq_vals/nu_c_obs_spl(t_obs_lin_vals[i]))**(1/3), color=colorFader(c1, c2, i/N), label=f't = {t_obs_lin_vals[i]/(60):.0f} minutes')
            plt.plot(middle_freq_vals, (middle_freq_vals/nu_c_obs_spl(t_obs_lin_vals[i]))**(-1/2), color=colorFader(c1, c2, i/N))
            plt.plot(high_freq_vals, (nu_m_obs_spl(t_obs_lin_vals[i])/nu_c_obs_spl(t_obs_lin_vals[i]))**(-1/2)*(high_freq_vals/nu_m_obs_spl(t_obs_lin_vals[i]))**(-p/2), color=colorFader(c1, c2, i/N))

        # If in slow cooling regime
        if nu_c_obs_spl(t_obs_lin_vals[i]) > nu_m_obs_spl(t_obs_lin_vals[i]):
            low_freq_vals = np.linspace(nu_m_obs_spl(t_obs_lin_vals[i])/10, nu_m_obs_spl(t_obs_lin_vals[i]), 10000)
            middle_freq_vals = np.linspace(nu_m_obs_spl(t_obs_lin_vals[i]), nu_c_obs_spl(t_obs_lin_vals[i]), 10000)
            high_freq_vals = np.linspace(nu_c_obs_spl(t_obs_lin_vals[i]), 10*nu_c_obs_spl(t_obs_lin_vals[i]), 10000)

            plt.plot(low_freq_vals, (low_freq_vals/nu_m_obs_spl(t_obs_lin_vals[i]))**(1/3), color=colorFader(c1, c2, i/N), label=f't = {t_obs_lin_vals[i]/(60):.0f} minutes')
            plt.plot(middle_freq_vals, (middle_freq_vals/nu_m_obs_spl(t_obs_lin_vals[i]))**((1-p)/2), color=colorFader(c1, c2, i/N))
            plt.plot(high_freq_vals, (nu_c_obs_spl(t_obs_lin_vals[i])/nu_m_obs_spl(t_obs_lin_vals[i]))**((1-p)/2)*(high_freq_vals/nu_c_obs_spl(t_obs_lin_vals[i]))**(-p/2), color=colorFader(c1, c2, i/N))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Normalised flux')
plt.legend(loc='lower left')
plt.show()


'''
fig, ax1 = plt.subplots()
lns1 = ax1.plot(t_engine_vals/(24*60*60), R_rk4_engine[:, 0], label='Engine frame', color='blue')
ax1.set_xlabel('Engine time [days]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('$\gamma$')
ax2 = ax1.twiny()
lns2 = ax2.plot(t_obs_vals/(60), R_rk4_obs[:, 0], label='Observer frame', color='blue')
ax2.set_xlabel('Observer time [minutes]')
ax2.set_xscale('log')
ax2.set_yscale('log')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs)
fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots()
lns1 = ax1.plot(t_engine_vals/(24*60*60), R_rk4_engine[:, 1]/1E27, label='Engine frame', color='blue')
ax1.set_xlabel('Engine time [days]')
ax1.set_ylabel('M [10$^{27}$ g]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2 = ax1.twiny()
lns2 = ax2.plot(t_obs_vals/(60), R_rk4_obs[:, 1]/1E27, label='Observer frame', color='blue')
ax2.set_xlabel('Observer time [minutes]')
ax2.set_xscale('log')
ax2.set_yscale('log')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc='upper left')
fig.tight_layout()
plt.show()
'''
