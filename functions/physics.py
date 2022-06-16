from scipy.constants import physical_constants, h


def gJ(J, S, L):
    return 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))


def gF(F, J, I, S, L):
    return gJ(J, S, L) * (F*(F+1) + S*(S+1) - I*(I+1)) / (2*F*(F+1))


def ZeemanCoefficient(F_gs, mF_gs, F_ex, mF_ex, J_gs, J_ex, L_gs, L_ex, S_gs, S_ex, I):
    
    gs = mF_gs * gF(F_gs, J_gs, I, S_gs, L_gs)
    ex = mF_ex * gF(F_ex, J_ex, I, S_ex, L_ex)
    
    return ex - gs


def ZeemanShift_41K_4s5p(F_gs, mF_gs, F_ex, mF_ex, B):
    
    # For 41K
    I = 3/2
    
    # 4s groundstate
    L_gs = 0
    
    # 5p excited state
    L_ex = 1
    
    # Common
    S = 1/2
    J = 1/2
    
    # Import constants from 2018 CODATA database
    mu_B = physical_constants["Bohr magneton in Hz/T"][0] * 1e-6 * 1e-4
    
    return ZeemanCoefficient(F_gs, mF_gs, F_ex, mF_ex, J, J, L_gs, L_ex, S, S, I) * mu_B * B