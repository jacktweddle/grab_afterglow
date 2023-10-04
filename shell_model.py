import numpy as np
import matplotlib.pyplot as plt
import time

# Define constants and initial conditions

N = 500000  # Number of timesteps
t_min = 60*60  # Time at start of simulation
t_max = 100*24*60*60  # Time at end of simulation
E = 1E51  # Energy injected into GRB in erg
rho_1 = 1.67E-24  # Density of ISM in g/cm3 // Also used as proton mass
p = 2.2  # Slope of electron population spectrum
epsilon_e = 0.1  # Ignorance parameter governing energy fraction put into electrons
epsilon_B = 0.01  # Ignorance parameter governing energy fraction put into magnetic field
m_e = 9.11E-28  # Electron mass in g
q_e = 4.803E-10  # Electron charge in Fr
sigma_t = 6.6525E-25  # Thomson scattering cross section in cm2
c = 3E10  # Speed of light in cm/s
tolerance = 1E-7  # Target relative tolerance for each calculation
R0 = np.array([(4/3)*np.pi*rho_1*c**3*t_min**3, 0.75*np.sqrt(E/(rho_1*np.pi*c**5))*(1/(t_min**1.5))])  # Initial vector of mass and Lorentz factor

# Calculate initial timestep size

h = (t_max - t_min) / N
t_vals = np.linspace(t_min, t_max, N)


def electron_population(C, gamma_m):
    gamma_e_vals = np.linspace(gamma_m, 10*gamma_m, N)
    n_e_vals = C * gamma_e_vals**(-p)
    return gamma_e_vals, n_e_vals


# Define a function to calculate the vector F = dR/dt = (dM/dt, d(gamma)/dt)

def F(R, t):
    dMdt = 4*np.pi*rho_1*c**3*t**2
    dgammadt = -(9/8)*np.sqrt(E/(rho_1*np.pi*c**5))*(1/(t**2.5))
    return np.array([dMdt, dgammadt])


# Define a function to perform the RK4 algorithm

def RK4(R):
    # Set initial value and define empty list within which to store values of R
    R0 = R
    R_vals = []

    # Loop over all time values, calculating vectors of k1-k4 coefficients at each
    # time step t, and use these to update R0 to the next vector R1 at time t+h
    # using the RK4 method
    for i in t_vals:
        k1 = h*F(R0, i)
        k2 = h*F(R0 + k1/2, i + h/2)
        k3 = h*F(R0 + k2/2, i + h/2)
        k4 = h*F(R0 + k3, i + h)
        R1 = R0 + (k1 + 2*k2 + 2*k3 + k4)/6

        # Store and reset for next time step
        R_vals.append(R1)
        R0 = R1

    # Convert list of R vectors into array of R vectors; easier to select
    # the elements of R to plot
    R_vals = np.array(R_vals)
    return R_vals


R_rk4 = RK4(R0)
mass_vals = R_rk4[:, 0]
gamma_vals = R_rk4[:, 1]

rho_2_vals = 4 * gamma_vals * rho_1
n_2_vals = rho_2_vals / rho_1
e_2_vals = 4 * gamma_vals * (gamma_vals-1) * rho_1 * c**2

gamma_m_vals = (epsilon_e*e_2_vals*(p-2)) / (n_2_vals*m_e*c**2*(p-1))
c_vals = n_2_vals * (p-1) * gamma_m_vals**(p-1)

B_vals = np.sqrt(8*np.pi*epsilon_B*e_2_vals)

nu_m_vals = gamma_m_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)

gamma_c_vals = (6*np.pi*m_e*c) / (sigma_t*B_vals**2*t_vals)
nu_c_vals = gamma_c_vals**2 * (q_e*B_vals)/(2*np.pi*m_e*c)

# Plotting

plt.figure(1)
for i in range(0, len(t_vals)):
    if i % 100000 == 0:
        gamma_e_vals, n_e_vals = electron_population(c_vals[i], gamma_m_vals[i])
        plt.plot(gamma_e_vals, n_e_vals)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\gamma_e$')
plt.ylabel('n$_e$($\gamma_e$)')
plt.show()

low_freq_vals = np.linspace(1E8, nu_m_vals[-1], 10000)
middle_freq_vals = np.linspace(nu_m_vals[-1], nu_c_vals[-1], 10000)
high_freq_vals = np.linspace(nu_c_vals[-1], 1E18, 10000)

plt.figure(2)
plt.plot(low_freq_vals, low_freq_vals**(1/3)/max(low_freq_vals)**(1/3))
plt.plot(middle_freq_vals, middle_freq_vals**(1-p/2)/min(middle_freq_vals)**(1-p/2))
plt.plot(high_freq_vals, high_freq_vals**(-p/2)/min(high_freq_vals)**(-p/2)*middle_freq_vals[-1]**(1-p/2)/min(middle_freq_vals)**(1-p/2))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('Normalised flux')
plt.show()

'''
plt.figure(1)
plt.plot(t_vals, R_rk4[:, 0])
plt.xlabel('t [s]')
plt.ylabel('M [g]')
plt.show()

plt.figure(2)
plt.plot(t_vals, R_rk4[:, 1])
plt.xlabel('t [s]')
plt.ylabel('$\gamma$')
plt.show()
'''