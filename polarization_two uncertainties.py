import numpy as np
import matplotlib.pyplot as plt
from functions.physics import NuclearPolarizationF2_41K, NuclearPolarizationErrorF2_41K
from routines.makeSpectrum import makeSpectrum
from routines.load import load
from routines.poisson import fit
from functions.models import peaks, F2_pi_sublevels, F2_pi_sublevels_FAST
from scipy.constants import physical_constants
from tabulate import tabulate
from scipy.optimize import curve_fit, minimize
from routines.uncertainties import computeInverseCorrelationMatrix, computeInverseCorrelationMatrix_ALT, estimateErrors, estimateErrorsMonteCarlo
from routines.uncertainties import chi2_Poisson

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
model = lambda x, am2, am1, a0, a1, a2, s: F2_pi_sublevels_FAST(x, am2, am1, a0, a1, a2, x0, h, s, g/2, B)

# Define Maximum Likelihood estimator
mle = lambda p, args: - np.sum(args[1] * np.log(model(args[0], *p)) - model(args[0], *p))
chi2 = lambda p, args: chi2_Poisson(model(args[0], *p), y_flip)

# Fit populations
minimize_kwargs = {"bounds" : [(0, np.inf)]*5 + [(0.01, np.inf)], "tol" : 1e-16}

p0_flip = [1, 1, 1, 1, 1, 1]
p0_flip, _ = curve_fit(model, x_flip, y_flip, p0=p0_flip)
res_flip = minimize(mle, p0_flip, args=[x_flip, y_flip], **minimize_kwargs)
p_flip_mle = res_flip.x
res_flip = minimize(chi2, p0_flip, args=[x_flip, y_flip], **minimize_kwargs)
p_flip_chi2 = res_flip.x

p_flip = p_flip_chi2

print(p_flip_mle)
print(p_flip_chi2)

# Plot chi2 by varying a single parameter holding the others fixed
# for n, p in enumerate(p_flip):
#     x = p + (p+1)*np.linspace(-0.9, 0.9, 100)
#     f = [model(x_flip, *[*p_flip[:n], X, *p_flip[n+1:]]) for X in x]
#     print(f)
#     y = [chi2_Poisson(F, y_flip) for F in f]
#     plt.plot(x, y)
#     plt.show()
    