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
    args = parser.parse_args()

    # isogenic setup
    params = {'K': 43615,
              't_com': 3.76,
              'beta': 1.48,
              'r_res': 1.032,
              'r_chal': 1.032,
              'gamma_res_chal': 1,
              'gamma_chal_res': 1,
              'resolution': 100,
              't_end': 10,
              'R_size': 10,
              'mode': 'ctmc'}

    C_steps = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    t_steps = np.linspace(0, 10, 0.1)

    with open('isogenic_runs.txt', 'w') as outfile:
        outfile.write("\t".join(["C_size", "t_chal", "avg_C"]) + "\n")
        for C_size in C_steps:
            sys.stderr.write("C_size: " + str(C_size) + "\n")
            for t_chal in t_steps:
                sys.stderr.write("t_chal: " + str(t_chal) + "\n")
                final_C_pop = []
                for repeat in range(0, args.repeats):
                    times, populations = solve_integral(params['K'], params['r_res'], params['r_chal'],
                    params['gamma_res_chal'], params['gamma_chal_res'], params['beta'], params['resolution'],
                    params['t_com'], t_chal, params['t_end'], C_size, params['R_size'], mode = params['mode'])

                    final_C_pop.append(np.log10(populations[-1,1] + 1))

                avg_C = sum(final_C_pop)/float(len(final_C_pop))
                outfile.write("\t".join([str(C_size), str(t_chal), str(avg_C)]) + "\n")

    sys.exit(0)
