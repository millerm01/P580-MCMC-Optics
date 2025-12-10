import numpy as np

def tmm(n, thicknesses, frequencies):
    """    
    Calculate reflection (and transmission) using transfer matrix method.

    Parameters
    ----------
    n : ndarray
        1D array containing refractive index for each layer (assumes air/vacuum (n = 1) on either side)
    thicknesses : ndarray
        1D array containing thickness for each layer (assumes air/vacuum (infinite thickness) on either side)
    frequencies : ndarray
        1D array of frequencies to evaluate at

    Returns
    -------
    R : ndarray
        Reflected power fraction at each wavelength
    T : ndarray
        Transmitted power fraction at each wavelength (currently commented out)
    """
    #Add air/vacuum 'layers' to either side of the stack
    n = np.concatenate((np.array([1]), n, np.array([1])))
    thicknesses = np.concatenate((np.array([np.inf]), thicknesses, np.array([np.inf])))
    
    c = 2.998e8  # speed of light, m/s
    lambda_vac = c / frequencies # convert desired frequencies to wavelengths
    
    kz = 2 * np.pi * n / lambda_vac.repeat(n.size).reshape((-1, n.size))
    delta = thicknesses * kz

    # Single interface reflection / transmission
    n1 = n[:-1]  # Index array, excluding the last
    n2 = n[1:]   # Index array, excluding the first
    n1n2 = n1 + n2
    r = (n1 - n2) / n1n2
    t = 2 * n1 / n1n2
    
    # Calculate transfer matrix
    eid = np.exp(1j * delta[:, 1:-1])
    enid = eid.conj()
    M = np.array([[enid, enid * r[1:]], [eid * r[1:], eid]]) / t[1:]
    M = np.moveaxis(M, (0, 1, 2, 3), (2, 3, 0, 1))

    M_tilde = np.tile(np.array([1, r[0], r[0], 1]) / t[0], lambda_vac.size).reshape(
        (-1, 2, 2)
    )
    for i in range(n.size - 2):
        M_tilde = M_tilde @ M[:, i]

    # Full-stack reflection / transmission amplitude
    refl = M_tilde[:, 1, 0] / M_tilde[:, 0, 0]
    # trans = 1 / M_tilde[:, 0, 0]

    # Full-stack reflected / transmitted power
    # R = np.abs(refl) ** 2
    # T = np.abs(trans) ** 2 * n[-1] / n[0]
    
    return np.abs(refl)


def logUniformPrior(guess, lower, upper):
    """
    returns the log of an unnormalized uniform prior between two bounds.
    
    Parameters:
        guess: point to get the uniform prior value for
        lower, upper: lower and upper ends of the uniform prior bounds
    """
    if lower<guess<upper:
        return 0
    else:
        return -np.inf

def logNormalPrior(guess, mean, std):
    """
    returns the log of an unnormalized normal prior with a mean and standard deviation.
    
    Parameters:
        guess: point to get the uniform prior value for
        mean, std: mean and standard deviation of the normal distribution
    """
    normal = (1/(std))*np.exp(-((guess-mean)/std)**2)

    return np.log(normal)

def prior(theta, uniform_lower_bounds, uniform_upper_bounds, normal_param_avgs, normal_param_stdevs):
    """
    returns log of prior probability distribution
    
    Parameters:
        theta: [tuple] model parameters
        
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
        combo_prior: the combined prior of all the parameters specified, given the assumed or specified bounds
        
    To Do:
        Generalize this so that it is similar to scipy.curve_fit() in its inputs: instead of 
        these difficult ind_bnds vs. measured_t vs. t_bnds conditions, we will have an input
        vector of the model parameters (theta), a vector of the lower bounds of those parameters,
        and a vector of the lower bounds of those parameters. Maybe later I can make an 
        alternate version with measured values (mean/standard deviation), if that is better.
    """
    combo_prior = 0
    
    uniform_params_length = len(uniform_lower_bounds)
    normal_params_length = len(normal_param_avgs)
    
    for i in range(uniform_params_length):
        combo_prior += logUniformPrior(theta[i], uniform_lower_bounds[i], uniform_upper_bounds[i])
        
    for i in range(normal_params_length):
        combo_prior += logNormalPrior(theta[uniform_params_length + i], normal_param_avgs[i], normal_param_stdevs[i])
        
    #combo_prior += logNormalPrior(theta[uniform_params_length + normal_params_length], total_thickness_avg, total_thickness_stdev)
    
    return combo_prior        
        
    
    
def log_likelihood(theta, x, y, sigma_y, fit_func):
    """
    returns log of likelihood
    
    Parameters:
        theta: model parameters (specified as a tuple)
        x: independent data (array of length N)
        y: measurements (array of length N)
        sigma: uncertainties on y (array of length N)
        
    Returns:
        log_likelihood: [float64] an (unnormalized) likelihood given the model specified, the parameters and the data given
    """
    predicted = fit_func(x, theta)
    
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sigma_y)))
    
    residual   = (y - predicted)**2
    chi_square = np.sum(residual/(2*(sigma_y**2)))
    foreground = constant - chi_square
    
    return foreground
    

def log_posterior(theta, x, y, sigma_y, uniform_lower_bounds, uniform_upper_bounds, normal_param_avgs, normal_param_stdevs, fit_func):
    """
    returns log of posterior probability distribution
    
    Parameters:
        theta: model parameters (specified as a tuple)
        x: independent data (array of length N)
        y: measurements (array of length N)
        sigma_y: uncertainties on y (array of length N)
        
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
        log_posterior: [float64] the log posterior for the specified guess with the given parameter ranges
    """
    return prior(theta, uniform_lower_bounds, uniform_upper_bounds, normal_param_avgs, normal_param_stdevs) + log_likelihood(theta, x, y, sigma_y, fit_func)
