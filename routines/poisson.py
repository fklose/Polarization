import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LbfgsInvHessProduct


def centralDiff(f, x, p, i, eps):
    
    left = np.copy(p)
    right = np.copy(p)
    
    left[i] = left[i] * (1 + eps)
    right[i] = right[i] * (1 - eps)
    
    return (f(x, *left) - f(x, *right)) / (2*p[i] * eps)


def altCentralDiff(f, x, p, i, eps):
    # TODO look for more robust way of taking derivative when p[i] == 0 to incorporate into centralDiff()
    # I would like to keep a relative step size instead of an absolute which would run into trouble as the
    # magnitude of the parameters approaches the step size.
    left = np.copy(p)
    right = np.copy(p)
    
    left[i] = left[i] + eps
    right[i] = right[i] - eps
    
    return (f(x, *left) - f(x, *right)) / (2 * eps)


def makeVector(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros(M)
    
    for i in range(M):
        OUT[i] = np.sum((data / f(x, *p) - 1) * centralDiff(f, x, p, i, eps))
    
    return OUT


def makeMatrix(f, x, data, p, eps):
    
    M = len(p)
    
    OUT = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
            OUT[i, j] = np.sum(data / (f(x, *p)**2) * centralDiff(f, x, p, i, eps) * centralDiff(f, x, p, j, eps))

    return OUT


def makeHessian(f, x, data, p, eps):
    # TODO This is only here beacause it uses a slightly different method to compute the derivative
    M = len(p)
    
    OUT = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
            OUT[i, j] = np.sum(data / (f(x, *p)**2) * altCentralDiff(f, x, p, i, eps) * altCentralDiff(f, x, p, j, eps))

    return OUT


def fit(f, x, y, p0, iter=200):
    
    eps = 0.00001
    
    M = len(p0)
    N = len(x)
    dof = N - M
    
    p0 = np.asarray(p0)
    p = np.copy(p0)
    
    for _ in range(iter):
        B = makeMatrix(f, x, y, p, eps)
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


def altFit(f, x, y, p0, bounds):
    # Instead of using method outlined in PHYSICA manual use scipy.minimize
    
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
        B = makeMatrix(f, x, y, p, 0.00001)
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        B_inv = res.hess_inv
        print("Use approximated Hessian")
    
    if type(B_inv) == LbfgsInvHessProduct:
        # Extract full matrix
        B_inv = B_inv.todense()
    
    np.printoptions(suppress=True)
    print(np.round(B_inv, 2))
    
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