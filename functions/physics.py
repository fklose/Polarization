from scipy.constants import physical_constants
import numpy as np
from sympy.physics.quantum.cg import CG


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


def starkShift_AC_41K_4s4p(Delta, I, F_gs, mF_gs, F_ex, mF_ex):
    
    # 41K, 4s -> 4p transition see (https://arxiv.org/pdf/1506.06651.pdf)
    I_sat = 1.71 # Saturation intensity [mW/cm^2]
    gamma = 2*np.pi * 5.956 # Natural linewidth [MHz]
    
    cg = CG(F_gs, mF_gs, 1, mF_ex - mF_gs, F_ex, mF_ex)
    cg = float(cg.doit())
    
    return gamma / 2 * (Delta / gamma - np.sqrt((Delta / gamma)**2 + I / 2 / I_sat)) * np.abs(cg)**2


def starkShift_AC_41K_4s5p(Delta, I, F_gs, mF_gs, F_ex, mF_ex):
    
    # 41K, 4s -> 5p transition see (https://arxiv.org/pdf/1506.06651.pdf)
    I_sat = 58.8 # Saturation intensity [mW/cm^2]
    gamma = 2*np.pi * 170.3e-3 # Natural linewidth [MHz]
    
    cg = CG(F_gs, mF_gs, 1, mF_ex - mF_gs, F_ex, mF_ex)
    cg = float(cg.doit())
    
    return gamma / 2 * (Delta / gamma - np.sqrt((Delta / gamma)**2 + I / 2 / I_sat)) * np.abs(cg)**2


def NuclearPolarizationF1_41K(F1_m1, F1_0, F1_1):
    
    I = 3/2
    
    total = np.abs(F1_m1) + np.abs(F1_0), np.abs(F1_1)
    
    return 1/I * 5/4 * (F1_1 - F1_m1) / total


def NuclearPolarizationF2_41K(F2_m2, F2_m1, F2_0, F2_1, F2_2):
    
    I = 3/2
    
    total = np.abs(F2_m2) + np.abs(F2_m1) + np.abs(F2_0) + np.abs(F2_1) + np.abs(F2_2)
    
    return 1/I * 3/4 * (2*F2_2 + F2_1 - F2_m1 - 2*F2_m2) / total


def NuclearPolarizationErrorF2_41K(F2_m2, F2_m1, F2_0, F2_1, F2_2, F2_m2_err, F2_m1_err, F2_0_err, F2_1_err, F2_2_err):
    
    I = 3/2
    mF = np.arange(-2, 3, 1)
    
    F2 = np.abs(np.array([F2_m2, F2_m1, F2_0, F2_1, F2_2]))
    F2_err = np.abs(np.array([F2_m2_err, F2_m1_err, F2_0_err, F2_1_err, F2_2_err]))
    
    total = np.sum(F2)
    total_err = np.sqrt(np.sum(F2_err**2))
    
    pops = F2 / total
    pops_err = pops * np.sqrt((F2_err / F2)**2 + (total_err / total)**2)
    
    return I * (3/4) * np.sqrt(np.sum((mF * pops_err)**2))


def NuclearPolarization_41K(F1_m1, F1_0, F1_1, F2_m2, F2_m1, F2_0, F2_1, F2_2):
    
    return NuclearPolarizationF1_41K(F1_m1, F1_0, F1_1) + NuclearPolarizationF2_41K(F2_m2, F2_m1, F2_0, F2_1, F2_2)