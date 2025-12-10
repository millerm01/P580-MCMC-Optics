"""
@author: Matthew Miller

This script can be used to perform MCMC fits of a transfer-matrix method
reflection model to VNA reflection measurements of an optical sample composed
of planar layers, providing estimates of the refractive indices and thicknesses
of the sample's layers. 

It does the following:
    - Reads in the VNA measurement data
    - Normalizes the sample measurements using the short measurements on a
    frequency-by-frequency basis
    - Calculates averages and standard deviations of the normalized data on a 
    frequency-by-frequency basis
    - Performs an MCMC fit based on a list of parameters, initial guesses,
    and priors defined by the user
    - Generates plots to visualize the results of the fit and project the
    reflected power of the sample over a desired frequency range

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner

from bayesian_functions import tmm
from emcee_functions import emcee_fit, plot_emcee_chain
#%%
"""
This section reads in the raw data and processes it so that a fit can be performed.
The code normalizes the data and calculates the average and standard deviation 
on a frequency-by-frequency basis. Several notes:
    - directory, sample_files, and short_files need to be filled in. It is assumed
    that there is one short file for each sample file.
    - It is assumed that there are multiple sample (and short) files. If there 
    is only one measurement, there is a commented-out line that can be used to 
    create a constant array of artificial standard deviations so that the fitting 
    code will work properly.
"""

directory = 'data/'   # PATH TO FOLDER CONTAINING VNA MEASUREMENT FILES; PLOTS WILL ALSO BE SAVED HERE
sample_files = ['20230816_20230801_1', '20230816_20230801_2', '20230816_20230801_3', '20230816_20230801_4']   # LIST OF NAMES OF SAMPLE MEASUREMENT FILES (WITHOUT .S1P SUFFIX)
short_files = ['20230816_20230801_ref1', '20230816_20230801_ref2', '20230816_20230801_ref3', '20230816_20230801_ref4']   # LIST OF NAMES OF SHORT MEASUREMENT FILES (WITHOUT.S1P SUFFIX)

# Read in and normalize the data, then append the resulting arrays to measurement_sets
measurement_sets = []
for i in range(len(sample_files)):
    sample_data = pd.read_csv(directory + sample_files[i] + ".S1P", header = 13, delim_whitespace = True, names = ['freq', 'amplitude', 'phase', 'dumb', 'dumb2', 'dumb3'])
    short_data = pd.read_csv(directory + short_files[i] + ".S1P", header = 13, delim_whitespace = True, names = ['freq', 'amplitude', 'phase', 'dumb', 'dumb2', 'dumb3'])
    measurement_sets.append(np.array(sample_data['amplitude']) / np.array(short_data['amplitude']))

# Frequencies at which the VNA recorded measurements, converted from GHz to Hz
freq_range = np.array(sample_data['freq']) * 1e9

# Find the average and standard deviation of reflection measurements, frequency by frequency
measured_reflection = np.mean(measurement_sets, axis=0)
measured_reflection_stdev = np.std(measurement_sets, axis=0)

# measured_reflection_stdev = np.ones(len(measured_reflection)) * 0.005  # fake standard deviations for the fitting code's sake, if needed

#%%
"""
This section sets up and performs the MCMC fit. This is where priors and other
information for the fit are defined. You should know the following:
    - The number of steps and number of walkers in the ensemble are defined
    using the variables nsteps and nwalkers.
    - As the code is currently set up, parameters can be given uniform or
    normal priors. The lists uniform_params and normal_params should contain, 
    respectively, the names of the parameters with uniform and normal priors.
    - Each parameter is either a refractive index or a thickness. The names of 
    thickness parameters should be placed in the thickness_params list. All 
    other parameters are assumed to be indices.
    - initial_guess contains the user's initial guesses of the parameter values.
    Here and in other locations that refer to the entire set of parameters, the
    order is defined by the order in the list param_names, which is the 
    contatenation uniform_params + normal_params.
    - lower and upper contain the lower and upper bounds of the uniform priors.
    - avgs and stdevs contain the averages and standard deviations that define
    the normal priors.
    - The function fit_func is what the MCMC code fits to the data. Its inputs
    are a frequency range and a list of parameters. The list of parameters
    should be in the same order as param_names. The function calculates and 
    returns the amplitude of the reflected wave over the frequency range for the 
    given parameters based on the transfer matrix method. If any thicknesses or 
    indices in the sample are not being used as fit parameters, they should be 
    hard coded in this function. If the sample is two-sided and assumed to be 
    symmetric, that is also expressed in this function. An example for a sample 
    coated on both sides where most thicknesses/indices are assumed to be the 
    same on both sides, one thickness may have different values on either side,
    and one index is hard coded rather than being determined by the fit:
    
    def fit_func(frequencies, params):
        dewal_n = params[0]
        ptfe_n = params[1]
        window_n = params[2]
        dewal_t1 = params[3]
        dewal_t2 = params[4]
        ptfe_t = params[5]
        ldpe_t = params[6]
        window_t = params[7]
        ldpe_n = 1.51
        return tmm(np.array([dewal_n, ldpe_n, ptfe_n, window_n, ptfe_n, ldpe_n, dewal_n]), np.array([dewal_t1, ldpe_t, ptfe_t, window_t, ptfe_t, ldpe_t, dewal_t2]), frequencies)
"""

nwalkers = 50
nsteps = 1000

uniform_params = ['dewal_n1', 'ldpe_n', 'dewal_n2', 'window_n', 'window_t']   # LIST OF PARAMETERS WITH UNIFORM PRIOR
normal_params = ['dewal_t1', 'ldpe_t', 'dewal_t2']   # LIST OF PARAMETERS WITH NORMAL PRIORS
param_names = uniform_params + normal_params

thickness_params = ['dewal_t1', 'ldpe_t', 'dewal_t2', 'window_t']   # LIST OF THICKNESS PARAMETERS
index_params = ['dewal_n1', 'ldpe_n', 'dewal_n2', 'window_n']   # LIST OF INDEX PARAMETERS

initial_guess = np.array([1.15, 1.51, 1.19, 1.51, 0.00125, 0.00028, 0.00009, 0.00005])   # ARRAY OF INITIAL GUESSES (IN ORDER OF param_names)

lower =  np.array([1.13, 1.50, 1.16, 1.49, 0.00120])   # ARRAY OF LOWER BOUNDS FOR UNIFORM PRIORS
upper = np.array([1.18, 1.52, 1.22, 1.53, 0.00130])   # ARRAY OF UPPER BOUNDS FOR UNIFORM PRIORS

avgs = np.array([0.000287, 0.00009, 0.00005])   # ARRAY OF AVERAGES FOR NORMAL PRIORS
stdevs = np.array([0.00001, 0.00001, 0.00001])   # ARRAY OF STANDARD DEVIATIONS FOR NORMAL PRIORS

def fit_func(frequencies, params):
    dewal_n1 = params[0]
    ldpe_n = params[1]
    dewal_n2 = params[2]
    window_n = params[3]
    window_t = params[4]
    dewal_t1 = params[5]
    ldpe_t = params[6]
    dewal_t2 = params[7]
    return tmm(np.array([dewal_n1, ldpe_n, dewal_n2, window_n, dewal_n2, ldpe_n, dewal_n1]), np.array([dewal_t1, ldpe_t, dewal_t2, window_t, dewal_t2, ldpe_t, dewal_t1]), frequencies)

emcee_samples = emcee_fit(freq_range, measured_reflection, measured_reflection_stdev, initial_guess, lower, upper, 
                          uniform_params, avgs, stdevs, normal_params, fit_func, nwalkers = nwalkers, nsteps = nsteps)

# Generate a plot of the trajectories taken by the walkers in the ensemble
plot_emcee_chain(emcee_samples, param_names, nchains=nwalkers)

#%%
"""
This section extracts the average values and uncertainties for the parameters.
It first isolates the end (burned-in) portion of the walkers' trajectories, then
prints the averages and uncertainties (+/- 1σ). Nothing needs to be filled in
in this section.
"""
burned_in = emcee_samples.xs(slice(500, 1000), level=1)
burned_in_indexed = burned_in.set_index(np.arange(0, len(burned_in)))

q = burned_in.quantile([0.16,0.50,0.84], axis=0)

param_avgs = []
for param_name in param_names:
    if param_name in thickness_params:
        print(param_name + " = {:.3f} + {:.5f} - {:.5f} mm".format(q[param_name][0.50]*1e3, 
                                                    q[param_name][0.84]*1e3-q[param_name][0.50]*1e3,
                                                    q[param_name][0.50]*1e3-q[param_name][0.16]*1e3))
        # Switch lengths from m to mm to make plotting more palatable
        burned_in_indexed['dewal_t1'] *= 1e3
    else:
        print(param_name + " = {:.3f} + {:.5f} - {:.5f}".format(q[param_name][0.50], 
                                                    q[param_name][0.84]-q[param_name][0.50],
                                                    q[param_name][0.50]-q[param_name][0.16]))
    # Build up a list of the average values obtained for the parameters, to be used in generating plots later
    param_avgs.append(q[param_name][0.50])



#%%
"""
This section takes the burned-in MCMC results and generates a corner plot using
the corner.py module. The generation of the corner plot itself is trivial; most
of the code here is meant to get the labels right (and I'm sure it is not the 
most efficient way of going about it). The following variables need to be filled
in:
    - descriptive_title_string and descriptive_file_string should contain
    identifying phrases to be inserted into the plot titles and file names of
    saved plots. This prevents you from having to go through and change each of
    them individually.
    - thickness_plot_labels and index_plot_labels hold the labels for the various 
    parameters to be used in the corner plot. For example, for an LDPE thickness
    parameter, you might put 'LDPE t (mm)'. Just make sure that they are in the 
    same order as in the thickness_params and index_params lists above.
"""
descriptive_title_string = '20230801, BA4 Window Redesign'   # PHRASE TO BE INSERTED INTO PLOT TITLES
descriptive_file_string = '20230801_BA4_window_redesign'   # PHRASE TO BE INSERTED INTO FILE NAMES OF SAVED PLOTS

thickness_plot_labels = ['DeWAL t1 (mm)', 'LDPE t (mm)', 'DeWAL t2 (mm)', 'Window t (mm)']   # THICKNESS PARAMETER LABELS FOR CORNER PLOT
index_plot_labels = ['DeWAL n1', 'LDPE n', 'DeWAL n2', 'Window n']   # INDEX PARAMETER LABELS FOR CORNER PLOT

# A dictionary linking the parameter names to their corner plot labels
label_dict = dict(zip(thickness_params + index_params, thickness_plot_labels + index_plot_labels))

# Build up a list of titles for the columns of the corner plot (e.g., 'n = 1.144 + 0.010 - 0.009')
column_titles = []
for param_name in param_names:
    if param_name in index_params:
        mean = q[param_name][0.50]
        upper = q[param_name][0.84]
        lower = q[param_name][0.16]
        column_titles.append(r"n = {:.3f} $^{{+{:.3f}}}_{{-{:.3f}}}$".format(mean, upper - mean, mean - lower))
    if param_name in thickness_params:
        mean = q[param_name][0.50] * 1e3
        upper = q[param_name][0.84] * 1e3
        lower = q[param_name][0.16] * 1e3
        column_titles.append(r"t = {:.3f} $^{{+{:.3f}}}_{{-{:.3f}}}$ mm".format(mean, upper - mean, mean - lower))

# Create the corner plot figure
plt.figure(dpi=250)
corner_plot = corner.corner(burned_in_indexed, bins=40, color='blue', plot_contours=False, 
                            labels=param_names, show_titles=True, quantiles = [0.16, 0.50, 0.84], plot_density=True, 
                            plot_datapoints=False, data_kwargs={'alpha':0.08})

corner_plot.suptitle(descriptive_title_string, fontsize=24)

# Go through each subplot and insert custom labels where appropriate
title_counter = 0
for i in range(len(param_names) ** 2):
    xlabel = corner_plot.axes[i].get_xlabel()
    ylabel = corner_plot.axes[i].get_ylabel()
    title = corner_plot.axes[i].get_title()
    if xlabel in thickness_params:
        corner_plot.axes[i].set_xlabel(label_dict[xlabel], fontsize=16)
    if ylabel in thickness_params:
        corner_plot.axes[i].set_ylabel(label_dict[ylabel], fontsize=16)
    if xlabel in index_params:
        corner_plot.axes[i].set_xlabel(label_dict[xlabel], fontsize=16)
    if ylabel in index_params:
        corner_plot.axes[i].set_ylabel(label_dict[ylabel], fontsize=16)
    if title:
        corner_plot.axes[i].set_title(column_titles[title_counter], fontsize=15)
        title_counter += 1

corner_plot.savefig(directory + descriptive_file_string + '_corner.png', dpi=200)
plt.show()
#%%
"""
This section generates two final plots: one displaying the VNA measurements
together with the fitted model, and one displaying the projected in-band 
(assumed to be 200 to 300 GHz) reflectance based on the results of the fit.
The fit was performed in terms of electromagnetic field amplitude (r) since that
is what the VNA reports, but that is converted into reflected power (r^2 = R) here.
"""
# Generate data vs. model plot
plt.figure(dpi=200)
plt.errorbar(freq_range/1e9, measured_reflection ** 2, yerr = 2 * measured_reflection * measured_reflection_stdev, fmt='.', ms=3, label='normalized data')
plt.plot(freq_range/1e9, fit_func(freq_range, param_avgs) ** 2, label='model')
plt.title("VNA Reflection Measurements: " + descriptive_title_string, size=8)
plt.ylim(0, 0.1)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Reflected Power (0-1)')
plt.legend()
plt.grid()
plt.savefig(directory + descriptive_file_string + '_data_vs_model', dpi=200)
plt.show()

# Generate projected in-band reflectance plot
plt.figure(dpi=200)
inband_freqs = np.linspace(200, 300, 200)*1e9
reflected_power_model = fit_func(inband_freqs, param_avgs) ** 2
plt.plot(inband_freqs/1e9, reflected_power_model, label='optical sample', color='blue', ls='-.')

print('avg. reflected power, model: ', np.mean(reflected_power_model))

plt.ylim(0, 0.15)
plt.legend()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Reflected Power (0-1)")
plt.title("Modeled In-Band Reflected Power: " + descriptive_title_string, size=10)
plt.annotate(r'R$_{avg}$ = '+str(np.round(np.mean(reflected_power_model), 3)*100)+'%', xy=(205,0.08), color='blue')

plt.savefig(directory + descriptive_file_string + '_inband_reflectance', dpi=200)
plt.show()
