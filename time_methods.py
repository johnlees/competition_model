#!python3

# imports
import sys
import argparse
import numpy as np

import timeit

from ode_int import solve_integral

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
    resolution = 2000

    #########
    # Input #
    #########

    description = 'Lotka–Volterra model'
    parser = argparse.ArgumentParser(description=description,
                                         prog='ode_int')
    parser.add_argument('--output-prefix', default="res_chal", help='Output prefix for plot')
    parser.add_argument('--resolution', default=resolution, type=int, help='Number of points per hour')
    parser.add_argument('--mode', default="deterministic", help='Model to use {ode, sde, ctmc}')

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


    # do the integral
    print(timeit.timeit('solve_integral(args.K, args.r_res, args.r_chal, args.g_RC, args.g_CR, args.beta, args.resolution, args.t_com, args.t_chal, args.t_end, args.C_size, args.R_size, args.mode)',
        number=100, globals=globals()))

    sys.exit(0)

