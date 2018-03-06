#!python3

# imports
import sys
from math import exp, log
import argparse
import random
from numba import jit
import numpy as np
from scipy import integrate
import sdeint
import matplotlib.pyplot as plt

####################
# Model definition #
####################

# Logistic growth solution
def log_grow(K, N0, r, t):
    return((K*N0*exp(r*t)) / (K + N0*(exp(r*t) - 1)))

log_grow_vec = np.vectorize(log_grow)

# Model for populations
def dN_dt(N, t, K, r_res, r_chal, a_RC, a_CR):
    return np.array([ N[0]*(r_res/K)*(K-N[0]-a_RC*N[1]),
                   N[1]*(r_chal/K)*(K-N[1]-a_CR*N[0])])

# Closure, for use with sdeint
def dN_dt_stochatic(K, r_res, r_chal, a_RC, a_CR):
    def f(N, t):
        dN = np.array([ N[0]*(r_res/K)*(K-N[0]-a_RC*N[1]),
                        N[1]*(r_chal/K)*(K-N[1]-a_CR*N[0])])
        return dN
    return f

# Jacobian (deterministic model only)
def d2N_dt2(N, t, K, r_res, r_chal, a_RC, a_CR):
    return np.array([[(r_res/K)*(K-2*N[0]-a_RC*N[1]), -(r_res/K)*N[0]*a_RC           ],
                  [-(r_chal/K)*N[1]*a_CR         , (r_chal/K)*(K-2*N[1]-a_CR*N[0])]])

# Brownian motion. No off-diagonal terms, as N, C covariance ignored in finite pop
def brownian(K, r_res, r_chal, a_RC, a_CR):
    def G(N, t):
        B = np.sqrt(np.diag([max(0, N[0]*(r_res/K)*(K+N[0]+a_RC*N[1])),
                             max(0, N[1]*(r_chal/K)*(K+N[1]+a_CR*N[0]))]))
        return B
    return G


###################
# Run integration #
###################

# Integration points
def t_range(start, end, resolution):
    assert end > start
    points = int(round(resolution/(end-start)))
    return np.linspace(start, end, points)

# Wrapper for Gillespie algorithm, which is JIT compiled by numba
def gillespie(t_init, t_max, R_init, C_init, K, r_res, r_chal, a_RC, a_CR):
    ta, Ra, Ca = gillespie_jit(t_init, t_max, R_init, C_init, K, r_res, r_chal, a_RC, a_CR)
    return(np.array(ta), np.column_stack((Ra, Ca)))

# CTMC stochastic algorithm
@jit(nopython=True)
def gillespie_jit(t_init, t_max, R_init, C_init, K, r_res, r_chal, a_RC, a_CR):
    ta = []
    Ra = []
    Ca = []

    t = 0
    R = R_init
    C = C_init
    while (t < t_max - t_init):
        # Previous step
        ta.append(t)
        Ra.append(R)
        Ca.append(C)

        # Rates
        B_R = r_res * R
        B_C = r_chal * C
        D_R = r_res/K * R * (R + a_RC * C)
        D_C = r_chal/K * C * (C + a_CR * R)

        # Choose time interval based on total intensity
        R_intensity = B_R + B_C + D_R + D_C
        if (R_intensity == 0):
            break
        u1 = random.random()
        t += -log(u1)/R_intensity

        u2 = random.random()
        if (u2 < B_R/R_intensity):
            R += 1
        elif (u2 > B_R/R_intensity and u2 < (B_R + D_R)/R_intensity):
            R -= 1
        elif (u2 > (B_R + D_R)/R_intensity and u2 < (B_R + D_R + B_C)/R_intensity):
            C += 1
        else:
            C -= 1

    ta = [t + t_init for t in ta]
    return(ta, Ra, Ca)

# Alternative CTMC algorithm (should be faster, with appropriate tau)
@jit(nopython=True)
def tau_leaping_jit(t_init, t_max, R_init, C_init, K, r_res, r_chal, a_RC, a_CR, tau = 0.001):
    ta = []
    Ra = []
    Ca = []

    t = 0
    R = R_init
    C = C_init
    while (t < t_max - t_init):
        # Previous step
        ta.append(t)
        Ra.append(R)
        Ca.append(C)

        # Rates
        B_R = r_res * R
        B_C = r_chal * C
        D_R = r_res/K * R * (R + a_RC * C)
        D_C = r_chal/K * C * (C + a_CR * R)

        # Constant step sizes. Choose to be small
        t += tau

        # Choose tau based on Cao et al 2005. Or use max rate from theory (rK/4)
        # tau = epsilon / max(B_R, B_C, D_R, D_C)

        # Rates, Poisson d0.0001istributed. First term is 'birth', second is 'death'. cf ODEs
        R += (np.random.poisson(B_R * tau, 1) - np.random.poisson(D_R * tau, 1))[0]
        C += (np.random.poisson(B_C * tau, 1) - np.random.poisson(D_C * tau, 1))[0]

        if R < 0:
            R = 0
        if C < 0:
            C = 0

    ta = [t + t_init for t in ta]
    return(ta, Ra, Ca)

# Solve R and C as a function of t
def solve_integral(K, r_res, r_chal, gamma_res_chal, gamma_chal_res, beta, resolution,
        t_com, t_chal, t_end, C_size, R_size, mode = 'ode'):

    # integration is in three pieces
    if t_chal > 0:
        t0 = t_range(0, t_chal, resolution)
        if mode == 'ode':
            N0 = np.vstack((log_grow_vec(K, R_size, r_res, t0), np.zeros(t0.shape[0]))).T
            N0_end = np.array([log_grow(K, R_size, r_res, t_chal), C_size]) # initial conditions
        else:
            t0, N0 = integrate_piece(t0, np.array([R_size, 0]), K, r_res, r_chal, 0, 0, mode)
            N0_end = np.array([N0[-1, 0], C_size])

        a_RC = gamma_res_chal
        a_CR = gamma_chal_res
        if (t_chal < t_com):
            # From arrival of challenger to development of competence
            t1 = t_range(t_chal, t_com, resolution)
            t1, N1 = integrate_piece(t1, N0_end, K, r_res, r_chal, a_RC, a_CR, mode)

            N1_end = N1[-1,:]
            t2 = t_range(t_com, t_com + t_chal, resolution)
        else:
            # case where t_chal > t_com (only two pieces)
            N1_end = N0_end
            t2 = t_range(t_chal, t_com + t_chal, resolution)

        # From development of competence in resident to development of competence in challenger
        a_CR = gamma_chal_res + beta
        t2, N2 = integrate_piece(t2, N1_end, K, r_res, r_chal, a_RC, a_CR, mode)

        # From development of competence in resident to development of competence in challenger
        t3 = t_range(t_com + t_chal, t_com + t_chal + t_end, resolution)
        N2_end = N2[-1,:]
        a_CR = gamma_chal_res
        t3, N3 = integrate_piece(t3, N2_end, K, r_res, r_chal, a_RC, a_CR, mode)

        if (t_chal < t_com):
            t_series = np.concatenate((t0, t1, t2, t3))
            N_series = np.concatenate((N0, N1, N2, N3))
        else:
            t_series = np.concatenate((t0, t2, t3))
            N_series = np.concatenate((N0, N2, N3))

    # If equal arrival, simple LV for all t
    else:
        t_series = t_range(0, t_end, resolution)
        t_series, N_series = integrate_piece(t_series, np.array([R_size, C_size]), K, r_res, r_chal, gamma_res_chal, gamma_chal_res, mode)

    return(t_series, N_series)

# Code to choose which integration method to use, given all model parameters
def integrate_piece(t, N_end, K, r_res, r_chal, a_RC, a_CR, mode = 'ode'):
    if mode == 'sde':
        f = dN_dt_stochatic(K, r_res, r_chal, a_RC, a_CR)
        G = brownian(K, r_res, r_chal, a_RC, a_CR)
        N = sdeint.itoint(f, G, N_end, t)
    elif mode == 'ctmc':
        t, N = gillespie(t[0], t[-1], N_end[0], N_end[1], K, r_res, r_chal, a_RC, a_CR)
    else:
        N = integrate.odeint(dN_dt, N_end, t, args=(K, r_res, r_chal, a_RC, a_CR), Dfun=d2N_dt2)

    return(t, N)


############
# Plotting #
############

def pop_plot(time, populations, output_file, title):
    f1 = plt.figure()

    plt.plot(time, populations[:,0], 'r-', label='Resident')
    plt.plot(time, populations[:,1], 'b-', label='Challenger')

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Time (hrs)')
    plt.ylabel('CFU')
    plt.title(title)

    f1.savefig(output_file)
    plt.show()

if __name__ == '__main__':

    ####################
    # Model parameters #
    ####################

    # see log_growth_fit.R for estimates
    K = 43615           # carrying capacity
    r_res = 1.032       # growth rate (resident)
    r_chal = r_res      # growth rate (challenger)

    # competition terms
    gamma_res_chal = 1  # competition (challenger on resident)
    gamma_chal_res = 1  # competition (resident on challenger)
    beta = 0.1          # strength of effect of competence

    # arrival times (in hours)
    t_com = 6.0           # time for competence to develop
    t_chal = 4.0          # time of arrival of challenger inoculum
    t_end = 10.0          # time to run integration in final step

    # starting parameters
    C_size = 1000      # size of challenger inoculum
    R_size = 10        # size of resident inoculum

    ###################
    # Numerical setup #
    ###################

    # Number of points per hour
    resolution = 1000

    #########
    # Input #
    #########

    description = 'Lotka–Volterra model'
    parser = argparse.ArgumentParser(description=description,
                                         prog='ode_int')
    parser.add_argument('--mode', default="deterministic", help='Model to use {ode, sde, ctmc}')
    parser.add_argument('--output-prefix', default="res_chal", help='Output prefix for plot')
    parser.add_argument('--resolution', default=resolution, type=int, help='Number of points per hour')

    growth = parser.add_argument_group('Growth parameters')
    growth.add_argument('--K', default=K, type=float, help='Carrying capacity')
    growth.add_argument('--r_res', default=r_res, type=float, help='Growth rate (resident)')
    growth.add_argument('--r_chal', default=r_chal, type=float, help='Growth rate (challenger)')

    competition = parser.add_argument_group('Competition terms')
    competition.add_argument('--g-RC', default=gamma_res_chal, type=float, help='competition (challenger on resident)')
    competition.add_argument('--g-CR', default=gamma_chal_res, type=float, help='competition (resident on challenger)')
    competition.add_argument('--beta', default=beta, type=float, help='strength of effect of competence')

    times = parser.add_argument_group('Arrival times (hrs)')
    times.add_argument('--t_com', default=t_com, type=float, help='time for competence to develop')
    times.add_argument('--t_chal', default=t_chal, type=float, help='time of arrival of challenger inoculum')
    times.add_argument('--t_end', default=t_end, type=float, help='time to run integration in final step')

    init = parser.add_argument_group('Starting sizes')
    init.add_argument('--C_size', default=C_size, type=float, help='size of challenger inoculum')
    init.add_argument('--R_size', default=R_size, type=float, help='size of resident inoculum')

    args = parser.parse_args()

    if args.mode != 'sde' and args.mode != 'ctmc':
        args.mode = 'ode'

    # do the integral
    times, populations = solve_integral(args.K, args.r_res, args.r_chal, args.g_RC, args.g_CR, args.beta, args.resolution,
                       args.t_com, args.t_chal, args.t_end, args.C_size, args.R_size, args.mode)

    ##########
    # Output #
    ##########

    # Draw plot
    if args.mode == 'sde':
        pop_plot(times, populations, args.output_prefix + '_stochastic.pdf', 'Resident vs. challenger (stochastic)')
    elif args.mode == 'ctmc':
        pop_plot(times, populations, args.output_prefix + '_ctmc.pdf', 'Resident vs. challenger (CTMC)')
    else:
        pop_plot(times, populations, args.output_prefix + '_deterministic.pdf', 'Resident vs. challenger (deterministic)')

    sys.exit(0)

