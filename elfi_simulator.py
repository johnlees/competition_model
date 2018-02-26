#!python3

# imports
import sys
import numpy as np
import elfi

from .ode_int import solve_integral

def batch_integral(beta, t_com, experimental_conditions, params, batch_size=1, random_state=None):

    batched_pops = []
    for i in range(0, batch_size):
        final_pop = []
        for experiment in experimental_conditions:
            (C_size, t_chal) = experiment

            times, populations = solve_integral(params['K'], params['r_res'], params['r_chal'],
                    params['gamma_res_chal'], params['gamma_chal_res'], beta, params['resolution'],
                    t_com, t_chal, params['t_end'], C_size, params['R_size'], mode = 'sde')
            final_pop.append((populations[-1,0], populations[-1,1]))
        batched_pops.append(np.asarray(final_pop))

    return(batched_pops)

if __name__ == '__main__':

    import argparse

    description = 'Fitting model'
    parser = argparse.ArgumentParser(description=description,
                                         prog='elfi_simulator.py')
    parser.add_argument('--experiments', help='Experimental data')
    args = parser.parse_args()

    # experimental setup
    params = {'K': 43615,
              'r_res': 1.032,
              'r_chal': 1.032,
              'gamma_res_chal': 1,
              'gamma_chal_res': 1,
              'resolution': 1000,
              't_end': 48,
              'R_size': 10}

    experimental_conditions = []
    obs = []
    with open(args.experiments, 'r') as experiment_file:
        for line in experiment_file:
            (C_size, t_chal, C_obs, R_obs) = line.rstrip().split("\t")
            experimental_conditions.append((C_size, t_chal))
            obs.append((R_obs, C_obs))

    obs = np.asarray(obs)

    # ELFI set-up
    beta = elfi.Prior(scipy.stats.gamma, a=2, scale=2)
    t_com = elfi.Prior(scipy.stats.gamma, a=4, scale=0.5)

    Y = elfi.Simulator(batch_integral, beta, t_com, experimental_conditions, params, observed=obs)
    d = elfi.Distance('euclidean', Y)
    log_d = elfi.Operation(np.log, 'd')

    # Fit w/ SMC ABC
    smc = elfi.SMC(log_d, batch_size=10000, seed=1)
    N = 1000
    schedule = [0.7, 0.2, 0.05]
    result_smc = smc.sample(N, schedule)
    result_smc.summary(all=True)
    result_smc.plot_marginals(all=True, bins=25, figsize=(8, 2), fontsize=12)

    # Run fit w/ BOLFI
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10,
                   bounds={'beta':(0, 5), 't_com':(0, 20)}, acq_noise_var=[0.1, 0.1], seed=seed)
    post = bolfi.fit(n_evidence=200)

    print(bolfi.target_model)
    bolfi.plot_state()
    bolfi.plot_discrepancy

    post.plot(logpdf=True)

    result_BOLFI = bolfi.sample(1000, info_freq=1000)
    print(result_BOLFI)
    result_BOLFI.plot_marginals()


