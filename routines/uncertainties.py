import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, newton
from itertools import combinations


def estimateErrors(mle, popt, args):
    """Determines 1-sigma parameter errors by finding parameter values to produce Delta ln(fun) = ln(fun(popt)) - ln(fun(p)) = 1/2
    
    See Lecture 8 of Scott Oser's PHYS509 slides starting on slide 18
    https://phas.ubc.ca/~oser/p509/Lec_08.pdf
    """
    
    # target = mle(popt, args) + 1/2
    target = mle(popt, args) + 7.04 / 2
    
    lower_errors = []
    upper_errors = []
    for i, p0 in enumerate(popt):
        print(i)
        # Make lambda
        p = lambda x: [*popt[:i], p0 + x, *popt[i+1:]]
        fun = lambda x: mle(p(x), args)
        
        if p0 == 0:
            X = np.linspace(-1, 1, 1000)
        elif i == 5:
            X = np.linspace(0, 2, 1000)
        else:
            X = np.linspace(-p0 / 2, p0 / 2, 1000)
        plt.plot(X + p0, [fun(x) - target for x in X])
        # plt.ylim(-1, 1)
        plt.vlines(p0, -3, 3)
        plt.show()
        
        # Obtain initial guess for Newton's method
        ub = 0
        # while np.abs(fun(ub) - target) > 1e-3:
        #     ub += 1e-3

        lb = 0
        # while np.abs(fun(lb) - target) > 1e-3:
        #     lb -= 1e-3

        ub = newton(fun, ub)
        lb = newton(fun, lb)
        
        lower_errors.append(lb)
        upper_errors.append(ub)
    
    return lower_errors, upper_errors


def computeInverseCorrelationMatrix(mle, p, args, eps):
    
    matrix = np.zeros(shape=(len(p), len(p)))
    
    for i, j in combinations(range(len(p)), 2):
        if i == j:
            if p[i] != 0:
                left = [*p[:i], p[i]*(1 - eps), *p[i+1:]]
                right = [*p[:i], p[i]*(1 + eps), *p[i+1:]]
                delta = p[i]*eps
            else:
                left = [*p[:i], p[i] - eps, *p[i+1:]]
                right = [*p[:i], p[i] + eps, *p[i+1:]]
                delta = eps
            
            partial = (mle(left, args) - 2 * mle(p, args) + mle(right, args)) / delta**2
        
        else:
            if p[i] != 0:
                left = [*p[:i], p[i]*(1 - eps), *p[i+1:]]
                right = [*p[:i], p[i]*(1 + eps), *p[i+1:]]
                delta = p[i]*eps
            else:
                left = [*p[:i], p[i] - eps, *p[i+1:]]
                right = [*p[:i], p[i] + eps, *p[i+1:]]
                delta = eps
            
            if p[j] != 0:
                left0 = [*left[:j], left[j]*(1 - eps), *left[j+1:]]
                left1 = [*left[:j], left[j]*(1 + eps), *left[j+1:]]
                right0 = [*right[:j], right[j]*(1 - eps), *right[j+1:]]
                right1 = [*right[:j], right[j]*(1 + eps), *right[j+1:]]
                delta *= p[j]*eps
            else:
                left0 = [*left[:j], left[j] - eps, *left[j+1:]]
                left1 = [*left[:j], left[j] + eps, *left[j+1:]]
                right0 = [*right[:j], right[j] - eps, *right[j+1:]]
                right1 = [*right[:j], right[j] + eps, *right[j+1:]]
                delta *= eps
            
            partial = (mle(left0, args) + mle(right1, args) - mle(left1, args) - mle(right0, args)) / 4 / delta
            
        matrix[i, j] = partial
        matrix[j, i] = partial
            
    return matrix


def estimateErrorsMonteCarlo(mle, popt, x, y, N, minimize_kwargs):
    """Estimate errors by simulating experiment.
    Experiemnts are simulated by drawing points from a dataset with replacement
    Each simulation will sample the same number of data points as the data.
    Each simulated experiment will be fitted using the model and the resulting variance in fit
    parameters is proportional to the error on the fit parameter.
    
    See 15.6.1 of Numerical Recipes by Press, Teukolsky and Vetterling
    """
    
    # 'Unpack' histogram
    samples = []
    for occurances, value in zip(y, x):
        samples += [value] * occurances
    
    # Simulate experiment N times, fit it and save parameters
    params = []
    for h in range(N):
        data = np.random.choice(samples, size=sum(y), replace=True)
        experiment = [0]*len(x)
        for i, v in enumerate(x):
            experiment[i] += sum(data == v)
        
        # Fit mle to experiment and save parameters
        params.append(minimize(mle, popt, [x, experiment], **minimize_kwargs).x)
        
        print(h)
    
    return np.asarray(params)