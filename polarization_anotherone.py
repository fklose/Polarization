from scipy.optimize import minimize
from functions.models import F2_pi_sublevels, peaks
from routines.makeSpectrum import makeSpectrum
from routines.load import load
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

# Filenames
root_path = "./"
run_flip = "output03512.root"
run_norm = "output03514.root"

# Load data
data_flip = load(root_path + run_flip)
data_norm = load(root_path + run_norm)

x_flip, y_flip = makeSpectrum(run_flip, data_flip)
x_norm, y_norm = makeSpectrum(run_norm, data_norm)

# Fit two peaks to the data
# model = lambda x, A, B, s, g: peaks(x, A, B, 363.61, 9.94, s, g)
model = lambda x, am2, am1, am0, a1, a2, s, B: F2_pi_sublevels(x, am2, am1, am0, a1, a2, 363.61, 9.94, s, 1.1/2, B)[0]
mle = lambda p, x, y: - np.sum(y*np.log(model(x, *p)) - model(x, *p))
p_flip = minimize(mle, p0 := [1, 1, 1, 1, 1, 1, 1], args=(x_flip, y_flip), bounds=[(0, np.inf)]*5 + [(1e-1, 2), (-4, 4)]).x
p_norm = minimize(mle, p0 := [1, 1, 1, 1, 1, 1, 1], args=(x_norm, y_norm), bounds=[(0, np.inf)]*5 + [(1e-1, 2), (-4, 4)]).x
print(np.round(p_flip, 2))
print(np.round(p_norm, 2))

# print((p_flip + p_norm)[2:] / 2)

# A, B, x0, h, s, g = p_flip
# A, B, s, g = p_flip

# Literature value for x0 is 363.61
# mu_B = physical_constants["Bohr magneton in Hz/T"][0] * 1e-6 * 1e-4
# B0 = (363.61 - x0) / (2/3) / mu_B
# print(B0)
# Literature value for h is 9.94
# B1 = (h - 9.94) / (2 * (2/3) * mu_B)
# B1 = (h - 9.94) / (mu_B)
# print(B1)
print(w := np.mean(x_flip[1:] - x_flip[:-1]))
plt.bar(x_flip, y_flip, edgecolor="black", width=w, yerr=np.sqrt(y_flip))
# plt.errorbar(x_flip, y_flip, np.sqrt(y_flip), ls="", label="Data", color="black")
plt.plot(x_flip, model(x_flip, *p_flip), label="Fit", color="black")
plt.plot(x_flip, model(x_flip, *p0), label="Initial Guess", color="red", ls="dashed")
for i in range(0, 5):
    plt.plot(x_flip, F2_pi_sublevels(x_flip, *p_flip[0:5], 363.61, 9.94, p_flip[5], 1.1/2, p_flip[6])[1][i], ls="dotted", color="orange")
plt.legend()
plt.show()

# plt.errorbar(x_flip, y_flip, np.sqrt(y_flip), ls="", marker="o", label="Flip", color="black")
# plt.errorbar(x_norm, y_norm, np.sqrt(y_norm), ls="", marker="o", label="Norm", color="purple")
# plt.show()