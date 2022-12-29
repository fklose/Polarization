from uncertainties.core import ufloat

def nuclear_polarization_41K_F2(pops, pops_err):
    # Uncertainties package propagates statistical errors when using the provided ufloat object
    am2 = ufloat(pops[0], pops_err[0])
    am1 = ufloat(pops[1], pops_err[1])
    a0  = ufloat(pops[2], pops_err[2])
    ap1 = ufloat(pops[3], pops_err[3])
    ap2 = ufloat(pops[4], pops_err[4])
    
    total = am2 + am1 + a0 + ap1 + ap2
    
    P = 1 / (3/2) * (3/4) * (2*ap2 + ap1 - am1 - 2*am2) / total
    
    return P.nominal_value, P.std_dev