import numpy as np


def centralDiff(f, x, p, i, eps):
    left = np.copy(p)
    right = np.copy(p)
    
    left[i] = left[i] * (1 + eps)
    right[i] = right[i] * (1 - eps)
    
    return (f(x, *left) - f(x, *right)) / (2*p[i] * eps)


def makeVector(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros(M)
    
    for i in range(M):
        OUT[i] = - np.sum((data / f(x, *p) - 1) * centralDiff(f, x, p, i, eps))
    
    return OUT


def makeMatrix(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
            OUT[i, j] = - np.sum(data / (f(x, *p)**2) * centralDiff(f, x, p, i, eps) * centralDiff(f, x, p, j, eps))

    return OUT


def fit(f, x, y, p0, iter=200, weights=None):
    
    eps = 0.00001
    
    M = len(p0)
    N = len(x)
    dof = N - M
    
    if weights == None:
        weights = np.ones(N)
    elif len(weights) != len(x):
        print("Each point must have a weight!")
        return
    else:
        weights = np.asarray(weights)
    
    p0 = np.asarray(p0)
    p = np.copy(p0)
    
    for _ in range(iter):
        p += np.linalg.solve(makeMatrix(f, x, y, p, eps), makeVector(f, x, y, p, eps))
    
    # Compute Chi-Squared
    X2 = np.zeros(np.shape(y))
    X2[y != 0] = f(x[y != 0], *p) - y[y != 0] + y[y != 0] * np.log(y[y != 0] / f(x[y != 0], *p))
    # Checking where y == 0 since log(0) = -np.inf but I am enforcing 0 * inf = 0
    X2[y == 0] = f(x[y == 0], *p)
    X2 = 2*np.sum(X2)
    
    # Compute errors
    E1 = np.zeros(M)
    B_inv = np.linalg.inv(makeMatrix(f, x, y, p, eps))
    for i in range(M):
        # main diagonal of B_inv looks like its always negative?
        # Fixed by swapping the signs of the elements in b and B
        # Alternatively can also take absolute value
        E1[i] = np.sqrt(np.abs(B_inv[i,i]))
    
    E2 = np.zeros(M)
    for i in range(M):
        E2[i] = E1[i]*np.sqrt(np.sum(weights * (y - f(x, *p))**2) / dof)
    
    # Compute scaled error (standard deviation)
    sigma = E2 / np.sqrt(X2 / dof)
    
    return p, E1, E2, sigma, X2