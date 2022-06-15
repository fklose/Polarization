from functions.math import voigt
import numpy as np

def peaks(x, A, B, x0, h, s, g):
    return A*voigt(x, x0 - h, s, g) + B*voigt(x, x0, s, g)


def F2_pi_sublevels(x, am2, am1, a0, a1, a2, x0, h, s, g, B):
    
    mu_B = 1.399
    
    F1_ex_m1 = 3*am1*voigt(x, x0 - h + 2/3*mu_B*B, s, g)
    F1_ex_0 = 4*a0*voigt(x, x0 - h, s, g)
    F1_ex_1 = 3*a1*voigt(x, x0 - h - 2/3*mu_B*B, s, g)
    F1_ex = F1_ex_m1 + F1_ex_0 + F1_ex_1
    
    F2_ex_m2 = 4*am2*voigt(x, x0 + 2/3*mu_B*B, s, g)
    F2_ex_m1 = 1*am1*voigt(x, x0 + 1/3*mu_B*B, s, g)
    F2_ex_1 = 1*a1*voigt(x, x0 - 1/3*mu_B*B, s, g)
    F2_ex_2 = 4*a2*voigt(x, x0 - 2/3*mu_B*B, s, g)
    F2_ex = F2_ex_m2 + F2_ex_m1 + F2_ex_1 + F2_ex_2    
    
    return F1_ex + F2_ex_2