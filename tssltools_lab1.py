import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def acf(x, lags=None):
    """ Computes the empirical autocorralation function.

    :param x: array (n,), sequence of data points
    :param lags: int, maximum lag to compute the ACF for. If None, this is set to n-1. Default is None.
    :return gamma: array (lags,), values of the ACF at lags 0 to lags
    """

    gamma = np.correlate(x, x, mode='full')  # Size here is always 2*len(x)-1
    gamma = gamma[int((gamma.size - 1) / 2):]  # Keep only second half
    if lags is not None and lags < len(gamma):
        gamma = gamma[0:lags + 1]
    return gamma / gamma[0]
    
def acfplot(x, lags=None, conf=0.95):
    """Plots the empirical autocorralation function.

    :param x: array (n,), sequence of data points
    :param lags: int, maximum lag to compute the ACF for. If None, this is set to n-1. Default is None.
    :param conf: float, number in the interval [0,1] which specifies the confidence level (based on a central limit
                 theorem under a white noise assumption) for two dashed lines drawn in the plot. Default is 0.95.
    :return:
    """

    n = len(x)
    y = acf(x, lags)
    lags = len(y)
    
    lag_vec = np.arange(lags)
    
    c = norm.isf((1-conf)/2,loc=0,scale=1/np.sqrt(n)) # Use inverse survival function (=1-cdf) at half the confidence interval
    plt.plot(lag_vec,c*np.ones(lags),'k--',linewidth=1, label=f"{100*conf}% confidence")
    plt.plot(lag_vec,-c*np.ones(lags),'k--',linewidth=1)
    
    plt.stem(lag_vec, y, linefmt='-', markerfmt=' ', basefmt="k ", use_line_collection=True, label="Empirical ACF")
    plt.plot(lag_vec, 0*lag_vec, 'k-')
    plt.title(f"Empirical ACF")
    plt.legend() 