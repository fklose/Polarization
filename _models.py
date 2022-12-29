from _constants import mu_B
from numba_stats import voigt as _voigt
# from scipy.special import voigt_profile
import numpy as np


def voigt(x, x0, s, g):
    # return voigt_profile(x - x0, s, g)
    return _voigt._pdf(x, g, x0, s)


def F2m2(x, am2, x0, h, B, s, g):
    return 4*am2 * voigt(x, x0 - 2/3*mu_B*B, s, g)


def F2m1(x, am1, x0, h, B, s, g):
    return 3*am1 * voigt(x, x0 - h - 2/3*mu_B*B, s, g) + 1*am1 * voigt(x, x0 - 1/3*mu_B*B, s, g)


def F20 (x, a0 , x0, h, B, s, g):
    return 4*a0 * voigt(x, x0 - h, s, g)


def F2p1(x, ap1, x0, h, B, s, g):
    return 3*ap1 * voigt(x, x0 - h + 2/3*mu_B*B, s, g) + 1*ap1 * voigt(x, x0 + 1/3*mu_B*B, s, g)


def F2p2(x, ap2, x0, h, B, s, g):
    return 4*ap2 * voigt(x, x0 + 2/3*mu_B*B, s, g)


def sublevel_model(x, x0, h, am2, am1, a0, ap1, ap2, B, s, g):
    return F2m2(x, am2, x0, h, B, s, g) \
        + F2m1(x, am1, x0, h, B, s, g) \
        + F20 (x, a0 , x0, h, B, s, g) \
        + F2p1(x, ap1, x0, h, B, s, g) \
        + F2p2(x, ap2, x0, h, B, s, g)
        

def sublevel_model_pdf(x, x0, h, am2, am1, a0, ap1, ap2, B, s, g):
    normalization = 4 * (am2 + am1 + a0 + ap1 + ap2)
    return sublevel_model(x, x0, h, am2, am1, a0, ap1, ap2, B, s, g) / normalization