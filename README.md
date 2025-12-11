# P580-MCMC-Optics

## Summary
This is the repository for my Physics 580 final project. The project uses Markov chain Monte Carlo methods to determine the thicknesses and refractive indices of the layers of a multi-layer microwave-band anti-reflection coating. It fits a transfer-matrix reflectance model to frequency-swept reflectance data using the emcee Python package.

For illustration purposes, the script is currently configured to be used with one particular AR coating sample; parts of the code are hard-coded for that sample. However, it can easily be adapted to other samples. This only requires several changes to the setup portion of the main script (to define the particular parameters, priors, etc. for that sample).

## Repository Contents

### Python scripts

- ``MCMC-main-script.py``: **Run this script to perform the fit.** It reads in the measurement files, cleans up the data, defines relevant fitting parameters, performs the fit, and produces several plots to visualize the results.
- ``emcee_functions.py``: Contains a couple of functions used by the main script to run the MCMC walker ensemble and plot the results.
- ``bayesian_functions.py``: Contains several Bayesian statistical functions used by the other scripts, as well as the generic transfer matrix method script used to define the particular fitting function for a sample.

### Data files

The ``data`` directory contains a sample set of raw measurement files as a fitting target. The data files are frequency-swept reflectance measurements taken in the microwave band using a vector network analyzer. There are four separate measurements of the sample (taken at different angles) and four blank measurements with a reflective metal plate. The main script averages the sample measurements and normalizes the result using the blank measurements to obtain the data for fitting.

## Required Python packages

- numpy
- matplotlib.pyplot
- pandas
- emcee
- corner
- seaborn

The project was developed using Python 3.14.
