from functions.maths import voigt
import numpy as np
from functions.physics import ZeemanShift_41K_4s5p


def peaks(x, A, B, x0, h, s, g):
    return A*voigt(x, x0 - h, s, g) + B*voigt(x, x0, s, g)


def F2_pi_sublevels(x, am2, am1, a0, a1, a2, x0, h, s, g, B):
    
    ZeemanShiftF2F1 = lambda mF, B: ZeemanShift_41K_4s5p(2, mF, 1, mF, B)
    ZeemanShiftF2F2 = lambda mF, B: ZeemanShift_41K_4s5p(2, mF, 2, mF, B)
    
    F1_ex_m1 = 3*am1*voigt(x, x0 - h + ZeemanShiftF2F1(-1, B), s, g)
    F1_ex_0 = 4*a0*voigt(x, x0 - h + ZeemanShiftF2F1(0, B), s, g)
    F1_ex_1 = 3*a1*voigt(x, x0 - h + ZeemanShiftF2F1(1, B), s, g)
    F1_ex = F1_ex_m1 + F1_ex_0 + F1_ex_1
    
    F2_ex_m2 = 4*am2*voigt(x, x0 + ZeemanShiftF2F2(-2, B), s, g)
    F2_ex_m1 = 1*am1*voigt(x, x0 + ZeemanShiftF2F2(-1, B), s, g)
    F2_ex_1 = 1*a1*voigt(x, x0 + ZeemanShiftF2F2(1, B), s, g)
    F2_ex_2 = 4*a2*voigt(x, x0 + ZeemanShiftF2F2(2, B), s, g)
    F2_ex = F2_ex_m2 + F2_ex_m1 + F2_ex_1 + F2_ex_2    
    
    return F1_ex + F2_ex_2