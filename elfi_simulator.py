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

from ode_int import solve_integral

def multi_integral(beta, t_com, experimental_conditions, params, batch_size=1, random_state=None):

    final_pop = []
    for experiment in experimental_conditions:
        (C_size, t_chal) = experiment
        #sys.stderr.write(str(C_size) + "," + str(t_chal) + "\n")
        times, populations = solve_integral(params['K'], params['r_res'], params['r_chal'],
                params['gamma_res_chal'], params['gamma_chal_res'], beta, params['resolution'],
                t_com, t_chal, params['t_end'], C_size, params['R_size'], mode = params['mode'])
        final_pop.append((populations[-1,0], populations[-1,1]))

    return(np.asarray(final_pop))

def log_destack(final_pops):
    x = final_pops.flatten().reshape(1, -1)
    return(np.log(np.fmax(x, np.zeros(x.shape)) + 1))

if __name__ == '__main__':

    import argparse

    description = 'Fitting model'
    parser = argparse.ArgumentParser(description=description,
                                         prog='elfi_simulator.py')
    parser.add_argument('--experiments', required=True, help='Experimental data')
    parser.add_argument('--simulated', default=False, action='store_true', help='Fit to simulated data')
    parser.add_argument('--beta', type=float, help='beta to simulate')
    parser.add_argument('--tcom', type=float, help='tcom to simulate')
    args = parser.parse_args()

    # experimental setup
    sys.stderr.write("Reading in setup\n")
    params = {'K': 43615,
              'r_res': 1.032,
              'r_chal': 1.032,
              'gamma_res_chal': 1,
              'gamma_chal_res': 1,
              'resolution': 1000,
              't_end': 10,
              'R_size': 10,
              'mode': 'sde'}

    experimental_conditions = []
    obs = []
    with open(args.experiments, 'r') as experiment_file:
        header = experiment_file.readline()
        for line in experiment_file:
            (C_size, t_chal, C_obs, R_obs) = line.rstrip().split("\t")
            experimental_conditions.append((float(C_size), float(t_chal)))
            obs.append((float(R_obs), float(C_obs)))

    real_obs = np.asarray(obs)

    # Simulate with known beta and t_com to check it works
    if args.simulated:
        sys.stderr.write("Simulating observations beta = " + str(args.beta) + "; t_com = " + str(args.tcom) + "\n")
        sim_obs = multi_integral(args.beta, args.tcom, experimental_conditions, params)
        np.savetxt("sim_obs.txt", sim_obs)

    # ELFI set-up
    # gamma params: a, loc, scale
    sys.stderr.write("Setting up ELFI\n")
    beta = elfi.Prior(scipy.stats.uniform, 0, 3)
    t_com = elfi.Prior(scipy.stats.uniform, 1, 5)

    vectorized_simulator = elfi.tools.vectorize(multi_integral, [2, 3])

    if args.simulated:
        Y = elfi.Simulator(vectorized_simulator, beta, t_com, experimental_conditions, params, observed=sim_obs)
    else:
        Y = elfi.Simulator(vectorized_simulator, beta, t_com, experimental_conditions, params, observed=real_obs)
    S = elfi.Summary(log_destack, Y)
    d = elfi.Distance('euclidean', S)
    log_d = elfi.Operation(np.log, d)
    #elfi.draw(d)

    # Fit w/ SMC ABC
    #sys.stderr.write("SMC inference\n")
    #smc = elfi.SMC(log_d, batch_size=10000, seed=1)
    #N = 1000
    #schedule = [0.7, 0.2, 0.05]
    #result_smc = smc.sample(N, schedule)
    #result_smc.summary(all=True)
    #result_smc.plot_marginals(all=True, bins=25, figsize=(8, 2), fontsize=12)

    # Run fit w/ BOLFI
    sys.stderr.write("BOLFI inference\n")
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10,
                   bounds={'beta':(0.001, 3), 't_com':(1, 6)}, acq_noise_var=[0.01, 0.01], seed=1)
    post = bolfi.fit(n_evidence=200)

    # Save results
    dill.dump(bolfi, open("bolfi.pkl", "wb"))
    dill.dump(post, open("posterior.pkl", "wb"))

    # plot results
    bolfi.plot_state()
    plt.savefig("bolfi_state.pdf")
    plt.close()

    bolfi.plot_discrepancy()
    plt.savefig("bolfi_discrepancy.pdf")
    plt.close()

    post.plot(logpdf=True)
    plt.savefig("posterior.pdf")
    plt.close()

    sys.exit(0)

    # sample from BOLFI posterior
    # this is now done separately
    #sys.stderr.write("Sampling from BOLFI posterior\n")
    #result_BOLFI = bolfi.sample(1000, info_freq=1000)
    #print(result_BOLFI)

    #result_BOLFI.plot_traces()
    #plt.savefig("posterior_traces.pdf")
    #plt.close()

    #result_BOLFI.plot_marginals()
    #plt.savefig("posterior_marginals.pdf")
    #plt.close()



