from subprocess import run
import numpy as np
import matplotlib.pyplot as plt
from functions.physics import NuclearPolarizationErrorF2_41K, NuclearPolarizationF2_41K
from routines.makeSpectrum import makeSpectrum
from routines.load import load
from routines.poisson import fit_physica as poisson_fit
from routines.poisson import fit as alt_poisson_fit
from functions.models import peaks, F2_pi_sublevels
from scipy.optimize import curve_fit
from scipy.constants import physical_constants
from tabulate import tabulate

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
p_flip, _, _, _, _ = alt_poisson_fit(model, x_flip, y_flip, p0_flip, bounds=[(0, np.inf)]*6)

p0_norm = [80, 800, x_norm[np.argmax(y_norm)], 9, 1, 1]
p_norm, _, _, _, _ = alt_poisson_fit(model, x_norm, y_norm, p0_norm, bounds=[(0, np.inf)]*6)

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

# Fit populations
p0_flip = [10, 10, 10, 10, 10, 1]
p_flip, _, _, err_flip, X2_flip = alt_poisson_fit(model, x_flip, y_flip, p0_flip, bounds=[(-np.inf, np.inf)]*5 + [(0.01, np.inf)])

p0_norm = [10, 10, 10, 10, 10, 1]
p_norm, _, _, err_norm, X2_norm = alt_poisson_fit(model, x_norm, y_norm, p0_norm, bounds=[(-np.inf, np.inf)]*5 + [(0.01, np.inf)])

# Enforce positive sign on parameters
p_flip = np.abs(p_flip)
p_norm = np.abs(p_norm)

# Print nuclear polarization and population levels
pnames = ["am2", "am1", "a0", "a1", "a2", "s", "P"]

p_flip_list = list(p_flip)
err_flip_list = list(err_flip)
p_norm_list = list(p_norm)
err_norm_list = list(err_norm)

p_flip_list.append(np.round(NuclearPolarizationF2_41K(p_flip[0], p_flip[1], p_flip[2], p_flip[3], p_flip[4]), 4))
err_flip_list.append(np.round(NuclearPolarizationErrorF2_41K(p_flip[0], p_flip[1], p_flip[2], p_flip[3], p_flip[4], err_flip[0], err_flip[1], err_flip[2], err_flip[3], err_flip[4]), 2))
p_norm_list.append(np.round(NuclearPolarizationF2_41K(p_norm[0], p_norm[1], p_norm[2], p_norm[3], p_norm[4]), 2))
err_norm_list.append(np.round(NuclearPolarizationErrorF2_41K(p_norm[0], p_norm[1], p_norm[2], p_norm[3], p_norm[4], err_norm[0], err_norm[1], err_norm[2], err_norm[3], err_norm[4]), 2))

print(tabulate(zip(pnames, np.round(p_flip_list, 4), np.round(err_flip_list, 4), np.round(p_norm_list, 4), np.round(err_norm_list, 4)), headers=["Name", "Flip", "Error", "Norm", "Error"]))

# Plot fits and spectra on the same plot
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fits = fig.add_subplot(gs[0, 0])
ress = fig.add_subplot(gs[1, 0])

# Plot OP_flip
fits.errorbar(x_flip, y_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)
fits.plot(x_flip, model(x_flip, *p_flip), **flip_style)
fits.plot(x_flip, model(x_flip, *p0_flip), **flip_style, **guess_style)

res_flip = y_flip - model(x_flip, *p_flip)
ress.errorbar(x_flip, res_flip, np.sqrt(y_flip), **flip_style, **flip_ebar)

# Plot OP_norm
fits.errorbar(x_norm, y_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)
fits.plot(x_norm, model(x_norm, *p_norm), **norm_style)
fits.plot(x_norm, model(x_norm, *p0_norm), **norm_style, **guess_style)

res_norm = y_norm - model(x_norm, *p_norm)
ress.errorbar(x_norm, res_norm, np.sqrt(y_norm), **norm_style, **norm_ebar)

# General plot stuff
fits.set_xticks([])
fits.legend()

ress.set_xlabel("Frequency wrt $^{39}$K cog [MHz]")
fits.set_ylabel("Counts")
ress.set_ylabel("Counts - Fit")

title = "Sublevel Fits"
flip_stats = "$\\chi^2_{Flip}$: " + f"{np.round(X2_flip / (len(y_flip) - len(p_flip)), 2)}"
norm_stats = "$\\chi^2_{Norm}$: " + f"{np.round(X2_norm / (len(y_norm) - len(p_norm)), 2)}"
fig.suptitle(" ; ".join([title, flip_stats, norm_stats]))

fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
