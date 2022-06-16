import numpy as np
import matplotlib.pyplot as plt
from functions.physics import NuclearPolarizationF2_41K
from routines.makeSpectrum import makeSpectrum
from routines.load import load
from routines.poisson import fit_physica as poisson_fit
from routines.poisson import fit as alt_poisson_fit
from functions.models import peaks, F2_pi_sublevels
from scipy.optimize import curve_fit

# root_path = "/home/felix/fklose/Data/ROOT_Files/"
root_path = "./"
run_flip = "output03512.root"
run_norm = "output03514.root"

fname = "output03514.root"

# Load data
data = load(root_path + fname)

x, y = makeSpectrum(fname, data)

# First we only fit two peaks to the data to determine the location of the 2->2' transition and hence the magnetic field
p0 = [80, 800, x[np.argmax(y)], 9, 1, 1]

model = lambda x, A, B, x0, h, s, g: peaks(x, A, B, x0, h, s, g)

# Get a better guess using least squares fit
# p, _ = curve_fit(model, x, n, p0)
# Perform final fit using a poisson fit
# p, E1, E2, sigma, X2 = poisson_fit(model, x, n, p, iter=200)
p, E1, E2, sigma, X2 = alt_poisson_fit(model, x, y, p0, bounds=[(0, np.inf)]*6)

print("Peaks Fit:")
print(*np.round(p0, 2))
print(*np.round(p, 2))
print(*np.round(E1, 2))
print(*np.round(E2, 2))
print(*np.round(sigma, 2))
print(np.round(X2 / (len(x) - len(p)), 2))

# Plot spectrum and peaks fit
fig = plt.figure(figsize=(6, 4))
fig.suptitle("Peaks Fit")
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fit = fig.add_subplot(gs[0,0])
res = fig.add_subplot(gs[1,0])
fit.errorbar(x, y, np.sqrt(y), capsize=3, color="black", ls="", label=f"{sum(y)} Events", marker=".")
fit.plot(x, model(x, *p0), color="red", ls="dashed", label="Guess")
fit.plot(x, model(x, *p), color="magenta", label="Fit")
res.errorbar(x, y - model(x, *p), np.sqrt(y), color="black", capsize=3, ls="", marker=".")
fit.set_xticks([])
res.set_xlabel("AOM Steps")
fit.set_ylabel("Counts")
res.set_ylabel("Counts - Fit")
fit.legend()
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()

# Knowing the location of the 2->2' peak we can fix some parameters
x0 = 363.61
x0_err = 0.48
h = 9.92
g = 1.1/2
B = (x0 - p[2]) * (2/3 * 1.399)**(-1)

p0 = [10, 10, 10, 10, 100, 1]

model = lambda x, am2, am1, a0, a1, a2, s: F2_pi_sublevels(x, am2, am1, a0, a1, a2, x0, h, s, g, B)

# Get a better guess using least squares fit
p1, _ = curve_fit(model, x, y, p0, bounds=([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), max_nfev=1000)
p0 = p1
# Perform final fit using a poisson fit
# p, E1, E2, sigma, X2 = poisson_fit(model, x, n, p, iter=200)
p, E1, E2, sigma, X2 = alt_poisson_fit(model, x, y, p0, bounds=[(0, np.inf)]*len(p0))

print()
print("Sublevel Fit")
print("am2 am1 a0 a1 a2 s")
print("p0: ", *np.round(p0, 2))
print("p: ", *np.round(p, 2))
print("E1: ", *np.round(E1, 2))
print("E2: ", *np.round(E2, 2))
print("sigma: ", *np.round(sigma, 2))
print("X2: ", np.round(X2 / (len(x) - len(p)), 2))

print(NuclearPolarizationF2_41K(*p[0:5]))

# Plot spectrum and sublevel fit
fig = plt.figure(figsize=(6, 4))
fig.suptitle("Sublevel Fit")
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fit = fig.add_subplot(gs[0,0])
res = fig.add_subplot(gs[1,0])
fit.errorbar(x, y, np.sqrt(y), capsize=3, color="black", ls="", label=f"{sum(y)} Events", marker=".")
fit.plot(x, model(x, *p0), color="red", ls="dashed", label="Guess")
# fit.plot(x, model(x, *p1), color="red", ls="dashed", label="Better Guess")
fit.plot(x, model(x, *p), color="magenta", label="Fit")
res.errorbar(x, y - model(x, *p), np.sqrt(y), color="black", capsize=3, ls="", marker=".")
fit.set_xticks([])
res.set_xlabel("AOM Steps")
fit.set_ylabel("Counts")
res.set_ylabel("Counts - Fit")
fit.legend()
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()