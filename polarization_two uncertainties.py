import numpy as np
import matplotlib.pyplot as plt
from functions.physics import NuclearPolarizationF2_41K, NuclearPolarizationErrorF2_41K
from routines.makeSpectrum import makeSpectrum
from routines.load import load
from routines.poisson import fit
from functions.models import peaks, F2_pi_sublevels
from scipy.constants import physical_constants
from tabulate import tabulate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import newton
from itertools import combinations

# Define plot styles
flip_ebar = {"capsize":3, "ls":"", "marker":".", "label":"$OP_{Flip}$"}
flip_style = {"color":"magenta"}
norm_ebar = {"capsize":3, "ls":"", "marker":"s", "label":"$OP_{Norm}$", "markersize":3}
norm_style = {"color":"purple"}
guess_style = {"ls":"dashed"}

# root_path = "/home/felix/fklose/Data/ROOT_Files/"
root_path = "./"
run_flip = "output03512.root"
run_norm = "output03514.root"

# Load data
data_flip = load(root_path + run_flip)
data_norm = load(root_path + run_norm)

x_flip, y_flip = makeSpectrum(run_flip, data_flip)
x_norm, y_norm = makeSpectrum(run_norm, data_norm)

# Fit two peaks to norm and flip data to obtain frequency of 2->2' transition and corresponding B-field
model = lambda x, A, B, x0, h, s, g: peaks(x, A, B, x0, h, s, g)

p0_flip = [80, 800, x_flip[np.argmax(y_flip)], 9, 1, 1]
p_flip, _, _, _, _ = fit(model, x_flip, y_flip, p0_flip, bounds=[(0, np.inf)]*6)

p0_norm = [80, 800, x_norm[np.argmax(y_norm)], 9, 1, 1]
p_norm, _, _, _, _ = fit(model, x_norm, y_norm, p0_norm, bounds=[(0, np.inf)]*6)

# Plot fits and spectra on the same plot
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fits = fig.add_subplot(gs[0, 0])
ress = fig.add_subplot(gs[1, 0])

# Plot OP_flip
fits.errorbar(x_flip, y_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)
fits.plot(x_flip, model(x_flip, *p_flip), **flip_style)

ress.errorbar(x_flip, y_flip - model(x_flip, *p_flip), **flip_style, **flip_ebar)

# Plot OP_norm
fits.errorbar(x_norm, y_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)
fits.plot(x_norm, model(x_norm, *p_norm), **norm_style)

ress.errorbar(x_norm, y_norm - model(x_norm, *p_norm), **norm_style, **norm_ebar)

# General plot stuff
fits.set_xticks([])
fits.legend()

fig.subplots_adjust(hspace=0, wspace=0)
plt.close()

# Using results from peak fit compute average of OP Flip and OP norm peaks
# This average should be close to the theoretical value of the 2->2' transition location
x0 = (p_flip[2] + p_norm[2]) / 2

# Assuming that we are well polarized meaning that most atoms will be in mF = +/- 2
# B will be proportional to the separation of the large peaks of OP norm and OP flip
# Based on past measurements we know that the sign of the B-field is (-)
mu_B = physical_constants["Bohr magneton in Hz/T"][0] * 1e-6 * 1e-4
B = - (np.abs(p_flip[2] - p_norm[2]) / 2) * (3/2) / mu_B

# We also know from theory that the splitting between F=2->2', mF=0 and F=2->1', mF=0 is 9.94 MHz 
# and that the linewidth of the 4s -> 5p transition in 41K is 1.1 Mhz
h = 9.94
g = 1.1

# We now move towards fitting the sublevel populations by defining the model
model = lambda x, am2, am1, a0, a1, a2, s: F2_pi_sublevels(x, am2, am1, a0, a1, a2, x0, h, s, g/2, B)

# Define Maximum Likelihood estimator
mle = lambda p, args: - np.sum(args[1] * np.log(model(args[0], *p)) - model(args[0], *p))

# Fit populations
p0_flip = [1, 1, 1, 1, 1, 1]
p0_flip, _ = curve_fit(model, x_flip, y_flip, p0=p0_flip)
res_flip = minimize(mle, p0_flip, args=[x_flip, y_flip], bounds=[(0, np.inf)]*5 + [(1, np.inf)], tol=1e-16)
p_flip = res_flip.x


def derivative1st():
    return

def derivative2nd():
    return

# Compute correlation matrix
def computeInverseCorrelationMatrix(mle, p, args, eps):
    
    matrix = np.zeros(shape=(len(p), len(p)))
    
    for i, j in combinations(range(len(p)), 2):
        if i == j:
            if p[i] != 0:
                left = [*p[:i], p[i]*(1 - eps), *p[i+1:]]
                right = [*p[:i], p[i]*(1 + eps), *p[i+1:]]
            else:
                left = [*p[:i], p[i] - eps, *p[i+1:]]
                right = [*p[:i], p[i] + eps, *p[i+1:]]
            
            matrix[i, i] = (mle(left, args) - 2 * mle(p, args) + mle(right, args)) / eps**2
        else:
            if p[i] != 0:
                left = [*p[:i], p[i]*(1 - eps), *p[i+1:]]
                right = [*p[:i], p[i]*(1 + eps), *p[i+1:]]
            else:
                left = [*p[:i], p[i] - eps, *p[i+1:]]
                right = [*p[:i], p[i] + eps, *p[i+1:]]
            
            if left[j] != 0:
                left = [*left[:j], left[j]*(1 - eps), *left[j+1:]]
            else:
                left = [*left[:j], left[j] - eps, *left[j+1:]]
                
            if right[j] != 0:
                right = [*right[:j], right[j]*(1 + eps), *right[j+1:]]
            else:
                right = [*right[:j], right[j] + eps, *right[j+1:]]
            
            partial = (mle(left, args) - 2 * mle(p, args) + mle(right, args)) / eps**2
            
            matrix[i, j] = partial
            matrix[j, i] = partial
    return matrix


m = computeInverseCorrelationMatrix(mle, p_flip, (x_flip, y_flip), 1e-10)                

print(np.round(np.linalg.inv(m), 3))

m_inv = np.linalg.inv(m)

for i in range(len(p_flip)):
    print(np.sqrt(m_inv[i, i]))
    

# def estimateErrorsMonteCarlo(mle, popt, args, x, y, N):
#     """Estimate errors by simulating experiment.
#     Experiemnts are simulated by drawing points from a dataset with replacement
#     Each simulation will sample the same number of data points as the data.
#     Each simulated experiment will be fitted using the model and the resulting variance in fit
#     parameters is proportional to the error on the fit parameter.
    
#     See 15.6.1 of Numerical Recipes by Press, Teukolsky and Vetterling
#     """
    
#     # 'Unpack' histogram
#     samples = []
#     for n, v in zip(y, x):
#         samples += [v] * n
    
#     # Simulate experiment N times, fit it and save parameters
#     params = []
#     for _ in range(N):
#         data = np.random.choice(samples, size=sum(y), replace=True)
#         experiment = [0]*len(x)
#         for i, v in enumerate(x):
#             experiment[i] += sum(data == v)
        
#         # Fit mle to experiment and save parameters
#         params.append(minimize(mle, popt, args, bounds=[(-np.inf, np.inf)]*5 + [(0.1, np.inf)], tol=1e-16).x)
    
#     return np.asarray(params)

# params = estimateErrorsMonteCarlo(mle, p_flip, [x_flip, y_flip], x_flip, y_flip, 100)

# table = []
# for i in range(len(p_flip)):
#     table.append([p_flip[i], np.std(params[:, i])])

# print(tabulate(table))



# def estimateErrors(mle, popt, args):
#     """Determines 1-sigma parameter errors by finding parameter values to produce Delta ln(fun) = ln(fun(popt)) - ln(fun(p)) = 1/2
    
#     See Lecture 8 of Scott Oser's PHYS509 slides starting on slide 18
#     https://phas.ubc.ca/~oser/p509/Lec_08.pdf
#     """
    
#     # target = mle(popt, args) + 1/2
#     target = mle(popt, args) + 7.04 / 2
    
#     lower_errors = []
#     upper_errors = []
#     for i, p0 in enumerate(popt):
#         print(i)
#         # Make lambda
#         p = lambda x: [*popt[:i], p0 + x, *popt[i+1:]]
#         fun = lambda x: mle(p(x), args)
        
#         if p0 == 0:
#             X = np.linspace(-1, 1, 1000)
#         elif i == 5:
#             X = np.linspace(0, 2, 1000)
#         else:
#             X = np.linspace(-p0 / 2, p0 / 2, 1000)
#         plt.plot(X + p0, [fun(x) - target for x in X])
#         # plt.ylim(-1, 1)
#         plt.vlines(p0, -3, 3)
#         plt.show()
        
#         # Obtain initial guess for Newton's method
#         ub = 0
#         # while np.abs(fun(ub) - target) > 1e-3:
#         #     ub += 1e-3

#         lb = 0
#         # while np.abs(fun(lb) - target) > 1e-3:
#         #     lb -= 1e-3

        
        
#         ub = newton(fun, ub)
#         lb = newton(fun, lb)
        
#         lower_errors.append(lb)
#         upper_errors.append(ub)
    
#     return lower_errors, upper_errors


# print(*estimateErrors(mle, p_flip, (x_flip, y_flip)))


# p0_norm = [1, 1, 1, 1, 1, 1]
# p0_norm, _ = curve_fit(model, x_norm, y_norm, p0=p0_norm)
# p_norm, _, _, err_norm, X2_norm = fit(model, x_norm, y_norm, p0_norm, bounds=[(-np.inf, np.inf)]*5 + [(0.01, np.inf)])

# # Enforce positive sign on parameters
# p_flip = np.abs(p_flip)
# p_norm = np.abs(p_norm)

# # Print nuclear polarization and population levels
# pnames = ["am2", "am1", "a0", "a1", "a2", "s", "P"]

# p_flip_list = list(p_flip)
# err_flip_list = list(err_flip)
# p_norm_list = list(p_norm)
# err_norm_list = list(err_norm)

# p_flip_list.append(np.round(NuclearPolarizationF2_41K(p_flip[0], p_flip[1], p_flip[2], p_flip[3], p_flip[4]), 4))
# err_flip_list.append(np.round(NuclearPolarizationErrorF2_41K(p_flip[0], p_flip[1], p_flip[2], p_flip[3], p_flip[4], err_flip[0], err_flip[1], err_flip[2], err_flip[3], err_flip[4]), 2))
# p_norm_list.append(np.round(NuclearPolarizationF2_41K(p_norm[0], p_norm[1], p_norm[2], p_norm[3], p_norm[4]), 2))
# err_norm_list.append(np.round(NuclearPolarizationErrorF2_41K(p_norm[0], p_norm[1], p_norm[2], p_norm[3], p_norm[4], err_norm[0], err_norm[1], err_norm[2], err_norm[3], err_norm[4]), 2))

# print(tabulate(zip(pnames, np.round(p_flip_list, 4), np.round(err_flip_list, 4), np.round(p_norm_list, 4), np.round(err_norm_list, 4)), headers=["Name", "Flip", "Error", "Norm", "Error"]))

# # Plot fits and spectra on the same plot
# fig = plt.figure(figsize=(18, 4))
# gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
# fits = fig.add_subplot(gs[0, 0])
# ress = fig.add_subplot(gs[1, 0])

# fit_flip = fig.add_subplot(gs[0, 1])
# ress_flip = fig.add_subplot(gs[1, 1])

# fit_norm = fig.add_subplot(gs[0, 2])
# ress_norm = fig.add_subplot(gs[1, 2])

# # Plot OP_flip
# fits.errorbar(x_flip, y_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)
# fits.plot(x_flip, model(x_flip, *p_flip), **flip_style)
# fits.plot(x_flip, model(x_flip, *p0_flip), **flip_style, **guess_style)

# res_flip = y_flip - model(x_flip, *p_flip)
# ress.errorbar(x_flip, res_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)

# fit_flip.errorbar(x_flip, y_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)
# fit_flip.plot(x_flip, model(x_flip, *p_flip), **flip_style)
# fit_flip.plot(x_flip, model(x_flip, *p0_flip), **flip_style, **guess_style)

# ress_flip.errorbar(x_flip, res_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)

# # Plot OP_norm
# fits.errorbar(x_norm, y_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)
# fits.plot(x_norm, model(x_norm, *p_norm), **norm_style)
# fits.plot(x_norm, model(x_norm, *p0_norm), **norm_style, **guess_style)

# res_norm = y_norm - model(x_norm, *p_norm)
# ress.errorbar(x_norm, res_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)

# fit_norm.errorbar(x_norm, y_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)
# fit_norm.plot(x_norm, model(x_norm, *p_norm), **norm_style)
# fit_norm.plot(x_norm, model(x_norm, *p0_norm), **norm_style, **guess_style)

# ress_norm.errorbar(x_norm, res_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)

# # General plot stuff
# fits.set_xticks([])
# fits.legend()

# ress.set_xlabel("Frequency wrt $^{39}$K cog [MHz]")
# fits.set_ylabel("Counts")
# ress.set_ylabel("Counts - Fit")

# ress_norm.set_xlabel("Frequency wrt $^{39}$K cog [MHz]")
# # fit_norm.set_ylabel("Counts")
# # ress_norm.set_ylabel("Counts - Fit")

# ress_flip.set_xlabel("Frequency wrt $^{39}$K cog [MHz]")
# # fit_flip.set_ylabel("Counts")
# # ress_flip.set_ylabel("Counts - Fit")

# title = "Sublevel Fits"
# flip_stats = "$\\chi^2_{Flip}$: " + f"{np.round(X2_flip / (len(y_flip) - len(p_flip)), 2)}"
# norm_stats = "$\\chi^2_{Norm}$: " + f"{np.round(X2_norm / (len(y_norm) - len(p_norm)), 2)}"
# fig.suptitle(" ; ".join([title, flip_stats, norm_stats]))

# fig.subplots_adjust(hspace=0, wspace=0.1, right=0.95, left=0.05)
# plt.show()
