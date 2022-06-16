import numpy as np
from scipy.special import wofz

def voigt(x, x0, s, g):
    z = ((x - x0) + 1j*g) / np.sqrt(2) / s
    return np.real(wofz(z)) / s / np.sqrt(2*np.pi)