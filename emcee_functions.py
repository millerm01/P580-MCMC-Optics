import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from bayesian_functions import log_posterior

# A couple of functions used in the MCMC fitting process

def emcee_fit(x, y, sigma_y, initial_guess, uniform_lower_bounds, uniform_upper_bounds, 
              uniform_param_names, normal_param_avgs, normal_param_stdevs, normal_param_names, fit_func, nwalkers, nsteps):
    """
    performs mixture model fit using emcee on the provided data
    
    Parameters:
        x: ndarray of floats, x values for fit
        y: ndarray of floats, y values for fit
        sigma: float, uncertainty 
        initial_guess: ndarray of initial guess values to use in mixture model fit (uniform then normal)
        column_names: list of strings to replace column names with
        nwalkers: float, number of walkers to be used in algorithm
        nsteps: float, number of steps to be taken in algorithm
    
    kwargs:
        ind_bnds: [tuple] bounds for the uniform prior on index
            *if not specified, bounds are 1.0 and 1.6
        measured_t: [tuple] -mean and standard deviation of measured thickness
                            -used for normal prior on thickness 
            *if not specified, checks for t_bnds
        t_bnds: [tuple] bounds for the uniform prior on thickness
            *if not specified, returns an error
                you MUST specify either measured_t or t_bnds 
    
    Returns:
        df: pandas dataframe containing emcee mixture model fit results
    """

    ndim = len(initial_guess)
    gaussian_ball = 1e-4 * np.random.randn(nwalkers, ndim)
    starting_positions = (1 + gaussian_ball) * initial_guess
    sampler = emcee.EnsembleSampler
    first_argument = nwalkers
    moves_arg = None
    # moves_arg = [(emcee.moves.StretchMove(a=2), 0.7), (emcee.moves.WalkMove(), 0.3)]

    sampler = sampler(first_argument, ndim, log_posterior, moves = moves_arg, 
                      args=(x, y, sigma_y, uniform_lower_bounds, uniform_upper_bounds, normal_param_avgs, normal_param_stdevs, 
                      fit_func))
    start_time = datetime.now()
    sampler.run_mcmc(starting_positions, nsteps)
    
    df = pd.DataFrame(np.vstack(sampler.chain))
    df.index = pd.MultiIndex.from_product([range(nwalkers), range(nsteps)], names=['walker', 'step'])
    column_names = uniform_param_names + normal_param_names
    df.columns = column_names
    print("time: ", datetime.now() - start_time)
    return df

def plot_emcee_chain(samples, parameter_names, nchains=10):
    """
    generates and plots traces of samples for nchains chains.
    
    Parameters:
        samples: sampled values from emcee_mixture_fit
        parameter_names: list of strings with parameter names
        nchains: int, number of chains to plot
    Returns: 
        nothing; plots traces
    """
    fig, axes = plt.subplots(len(parameter_names), figsize=(15,15))
    for ax, name in zip(axes, parameter_names):
        ax.set(ylabel=name)
    for i in range(nchains):
        for ax, name in zip(axes, parameter_names):
            sns.lineplot(data=samples.loc[i], x=samples.loc[i].index, y=name, ax=ax)
    # fig.savefig("walkers.png", dpi=200)
    plt.show()