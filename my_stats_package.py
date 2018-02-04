import pandas as pd
import numpy as np

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, len(data)+1) / n

    return x, y

#variance is the mean of the squared distances of data points from the mean (nb: weird units)
def variance(data):
    """Compute the variance of a one-dimensional array"""
    differences = data - np.mean(data)
    diff_sq = differences ** 2
    variance = np.mean(diff_sq)

    return variance

#standard deviation is the squareroot of the variance (nb: same units as data)
#in numpy use np.std()
def standard_deviation(data):
    """Compute the standard deviation of a one-dimensional array"""

    return np.sqrt(variance(data))

def mean_squarred_error(data, predictions):
    mse = 1/len(data) * np.sum((data - predictions)**2)

    return mse

# The Pearson correlation coeffiecient is a dimensionless measurement of the covariance
def pearson_cov(x,y):
    return covariance(x, y) / (standard_deviation(x) * standard_deviation(y))

#Bernoulli trials are probabilistic trials with two outcomes: success and failure
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()


        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
"""
SEM
In fact, it can be shown theoretically that under not-too-restrictive conditions,
the value of the mean will always be Normally distributed. (This does not hold
in general, just for the mean and a few other statistics.) The standard deviation
of this distribution, called the standard error of the mean, or SEM, is given by
the standard deviation of the data divided by the square root of the number of
data points. I.e., for a data set, sem = np.std(data) / np.sqrt(len(data)).
Using hacker statistics, you get this same result without the need to derive it,
but you will verify this result from your bootstrap replicates.
"""

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps
