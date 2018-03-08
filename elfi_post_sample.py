#!python3

# imports
import sys
import dill
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.stats
import elfi

if __name__ == '__main__':

    import argparse

    description = 'Fitting model'
    parser = argparse.ArgumentParser(description=description,
                                         prog='elfi_simulator.py')
    parser.add_argument('--bolfi', required=True, help='Bolfi dill')
    parser.add_argument('--posterior', default = None, help='Posterior dill')
    args = parser.parse_args()

    # Save results
    with open(args.bolfi, 'rb') as bolfi_dill:
        bolfi = dill.load(bolfi_dill)
    if args.posterior is not None:
        with open(args.posterior, 'rb') as post_dill:
            post = dill.load(post_dill)

    # sample from BOLFI posterior
    sys.stderr.write("Sampling from BOLFI posterior\n")
    result_BOLFI = bolfi.sample(1000, info_freq=1000)
    print(result_BOLFI)

    result_BOLFI.plot_traces()
    plt.savefig("posterior_traces.pdf")
    plt.close()

    result_BOLFI.plot_marginals()
    plt.savefig("posterior_marginals.pdf")
    plt.close()



