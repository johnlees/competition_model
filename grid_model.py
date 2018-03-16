#!python3

# imports
import sys
import dill
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ode_int import solve_integral

if __name__ == '__main__':

    import argparse

    description = 'Calculate model regimes'
    parser = argparse.ArgumentParser(description=description,
                                         prog='elfi_simulator.py')
    parser.add_argument('--repeats', type=int, default = 100, help='Number of simulator runs [default: 100]')
    parser.add_argument('--mode', default='ctmc', help='Mode of integral')
    parser.add_argument('--output', type=str, required=True, help='Output prefix')
    parser.add_argument('--isogenic', action='store_true', default=False, help='Run isogenic simulations')
    parser.add_argument('--intergenic', action='store_true', default=False, help='Run intergenic simulations')
    parser.add_argument('--t-chal', default = 24, type=float, help='t_chal in intergenic simulations')
    parser.add_argument('--threshold', default = 1, type=float, help='Threshold for declaring co-existence')
    parser.add_argument('--grid-resolution', default = 50, type=int, help='Number of points to draw')
    parser.add_argument('--de-resolution', default = 2000, type=int, help='Number of points to integrate')
    args = parser.parse_args()

    # isogenic setup
    params = {'K': 43615,
              't_com': 3.76,
              'beta': 1.48,
              'r_res': 1.032,
              'r_chal': 1.032,
              'gamma_res_chal': 1,
              'gamma_chal_res': 1,
              'resolution': args.de_resolution,
              't_end': 36,
              'R_size': 10,
              'mode': args.mode}

    C_steps = np.power(10, np.linspace(0, 4, args.grid_resolution))
    t_steps = np.linspace(0, 24, args.grid_resolution)

    if args.isogenic:
        with open(args.output + '.isogenic_runs.txt', 'w') as outfile:
            outfile.write("\t".join(["C_size", "t_chal", "avg_R", "avg_C"]) + "\n")
            for C_size in C_steps:
                #sys.stderr.write("C_size: " + str(C_size) + "\n")
                for t_chal in t_steps:
                    #sys.stderr.write("t_chal: " + str(t_chal) + "\n")
                    final_C_pop = []
                    final_R_pop = []
                    for repeat in range(0, args.repeats):
                        times, populations = solve_integral(params['K'], params['r_res'], params['r_chal'],
                        params['gamma_res_chal'], params['gamma_chal_res'], params['beta'], params['resolution'],
                        params['t_com'], t_chal, params['t_end'], C_size, params['R_size'], mode = params['mode'])

                        final_R_pop.append(populations[-1,0])
                        final_C_pop.append(populations[-1,1])

                    avg_C = sum(final_C_pop)/float(len(final_C_pop))
                    avg_R = sum(final_R_pop)/float(len(final_R_pop))
                    outfile.write("\t".join([str(C_size), str(t_chal), str(avg_R), str(avg_C)]) + "\n")

    # intergenic setup
    params = {'K': 43615,
              't_com': 3.76,
              'beta': 1.48,
              'r_res': 1.032,
              'r_chal': 1.032,
              'resolution': args.de_resolution,
              't_end': 36,
              'R_size': 10,
              'C_size': 10,
              't_chal': args.t_chal,
              'mode': args.mode}

    gamma_res_chal_steps = np.power(10, np.linspace(-2, 2, args.grid_resolution))
    gamma_chal_res_steps = np.power(10, np.linspace(-2, 2, args.grid_resolution))

    if args.intergenic:
        with open(args.output + '.intergenic_runs.txt', 'w') as outfile:
            outfile.write("\t".join(["gamma_rc", "gamma_cr", "final_R", "final_C", "domain"]) + "\n")
            for gamma_rc in gamma_res_chal_steps:
                #sys.stderr.write("gamma: " + str(gamma) + "\n")
                for gamma_cr in gamma_chal_res_steps:
                    #sys.stderr.write("r: " + str(r) + "\n")
                    final_R_pop = []
                    final_C_pop = []
                    for repeat in range(0, args.repeats):
                        times, populations = solve_integral(params['K'], params['r_res'], params['r_chal'],
                        gamma_rc, gamma_cr, params['beta'], params['resolution'],
                        params['t_com'], params['t_chal'], params['t_end'], params['C_size'], params['R_size'], mode = params['mode'])

                        final_R_pop.append(populations[-1,0])
                        final_C_pop.append(populations[-1,1])

                    avg_R = sum(final_R_pop)/float(len(final_R_pop))
                    avg_C = sum(final_C_pop)/float(len(final_C_pop))
                    if avg_R > args.threshold and avg_C > args.threshold:
                        domain = 0.5 # co-existence
                    elif avg_R > args.threshold:
                        domain = 1 # resident wins
                    elif avg_C > args.threshold:
                        domain = 0 # challenger wins
                    else:
                        domain = -1 # both dead
                    outfile.write("\t".join([str(gamma_rc), str(gamma_cr), str(avg_R), str(avg_C), str(domain)]) + "\n")

    sys.exit(0)
