import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LbfgsInvHessProduct


def centralDiff(f, x, p, i, eps):
    
    tol = 1e-16
    
    left = np.copy(p)
    right = np.copy(p)
    
    if -tol <= p[i] and p[i] <= tol:
        # When p[i] is close to zero cannot use relative stepsize as p[i]*eps = 0 and hence the result will be infinite
        # Instead use a very small finite step size
        # This is not in the PHYSICA manual
        left[i] = left[i] + tol
        right[i] = right[i] - tol
        
        return (f(x, *left) - f(x, *right)) / (2 * tol)
    else:
        left[i] = left[i] * (1 + eps)
        right[i] = right[i] * (1 - eps)
        
        return (f(x, *left) - f(x, *right)) / (2*p[i] * eps)


def makeVector(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros(M)
    
    for i in range(M):
        OUT[i] = np.sum((data / f(x, *p) - 1) * centralDiff(f, x, p, i, eps))
    
    return OUT


def makeHessian(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
            OUT[i, j] = np.sum(data / (f(x, *p)**2) * centralDiff(f, x, p, i, eps) * centralDiff(f, x, p, j, eps))

    return OUT


def fit_physica(f, x, y, p0, iter=200):
    # Follows the method presented in the PHYSICA manual
    
    eps = 0.00001
    
    M = len(p0)
    N = len(x)
    dof = N - M
    
    p0 = np.asarray(p0)
    p = np.copy(p0)
    
    for _ in range(iter):
        B = makeHessian(f, x, y, p, eps)
        b = makeVector(f, x, y, p, eps)
        dp = np.linalg.solve(B, b)
        p += dp
    
    # Compute Chi-Squared
    X2 = np.zeros(np.shape(y))
    # To avoid errors X2 is computed first for all points where y != 0 to avoid infinities
    # Where y = 0 enforce that 0*log(0) = 0 and not inf
    X2[y != 0] = f(x[y != 0], *p) - y[y != 0] + y[y != 0] * np.log(y[y != 0] / f(x[y != 0], *p))
    X2[y == 0] = f(x[y == 0], *p)
    X2 = 2*np.sum(X2)
    
    # Compute errors
    E1 = np.zeros(M)
    B_inv = np.linalg.inv(B)
    for i in range(M):
        # main diagonal of B_inv looks like its always negative?
        # Fixed by swapping the signs of the elements in b and B
        # Alternatively can also take absolute value
        E1[i] = np.sqrt(B_inv[i,i])
    
    E2 = np.zeros(M)
    for i in range(M):
        E2[i] = E1[i]*np.sqrt(np.sum((y - f(x, *p))**2) / dof)
    
    # Compute scaled error (standard deviation)
    sigma = E2 / np.sqrt(X2 / dof)
    
    return p, E1, E2, sigma, X2


def fit(f, x, y, p0, bounds):
    # Instead of using method outlined in PHYSICA manual use scipy.minimize this is more robust
    
    # Define log-likelihood function
    LL = lambda p : - np.sum(y * np.log(f(x, *p)) - f(x, *p))
    
    # Minimize using scipy.optimize.minimize
    res = minimize(LL, p0, bounds=bounds)
    
    p = res.x
    
    # Errors are obtained from the inverse of the Hessian matrix
    M = len(p)
    N = len(x)
    dof = N - M
    
    # Try and compute the Hessian and its inverse by hand
    try:
        B = makeHessian(f, x, y, p, 0.00001)
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = res.hess_inv
        # print("Use approximated Hessian")
    
    if type(B_inv) == LbfgsInvHessProduct:
        # Extract full matrix
        B_inv = B_inv.todense()
    
    # np.printoptions(suppress=True)
    # print(np.round(B_inv, 2))
    
    E1 = np.zeros(M)
    for i in range(len(p)):
        E1[i] = np.sqrt(B_inv[i, i])
    
    E2 = np.zeros(M)
    for i in range(M):
        E2[i] = E1[i]*np.sqrt(np.sum((y - f(x, *p))**2) / dof)
        
    # Compute Chi-Squared
    X2 = np.zeros(np.shape(y))
    # To avoid errors X2 is computed first for all points where y != 0 to avoid infinities
    # Where y = 0 enforce that 0*log(0) = 0 and not inf
    X2[y != 0] = f(x[y != 0], *p) - y[y != 0] + y[y != 0] * np.log(y[y != 0] / f(x[y != 0], *p))
    X2[y == 0] = f(x[y == 0], *p)
    X2 = 2*np.sum(X2)
    
    # Compute scaled error
    sigma = E2 / np.sqrt(X2 / dof)
    
    return p, E1, E2, sigma, X2