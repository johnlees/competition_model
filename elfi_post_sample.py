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
    args = parser.parse_args()

    # Load results
    with open(args.bolfi, 'rb') as bolfi_dill:
        bolfi = dill.load(bolfi_dill)

    print(bolfi.target_model)

    # sample from BOLFI posterior
    sys.stderr.write("Sampling from BOLFI posterior\n")
    result_BOLFI = bolfi.sample(2000, info_freq=1000)

    print(result_BOLFI)
    np.savetxt("samples.txt", result_BOLFI.samples_array)

    result_BOLFI.plot_traces()
    plt.savefig("posterior_traces.pdf")
    plt.close()

    result_BOLFI.plot_pairs()
    plt.savefig("pair_traces.pdf")
    plt.close()

    result_BOLFI.plot_marginals()
    plt.savefig("posterior_marginals.pdf")
    plt.close()



