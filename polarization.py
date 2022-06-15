import numpy as np
import matplotlib.pyplot as plt
from routines.load import load
from routines.poisson import fit as poisson_fit
from functions.models import peaks, F2_pi_sublevels
from scipy.optimize import curve_fit

# root_path = "/home/felix/fklose/Data/ROOT_Files/"
root_path = "./"
run_flip = "output03512.root"
run_norm = "output03514.root"

# Load data
data = load(root_path + run_flip)

TTTL_OP_Beam = data[:, 1]
Run_time = data[:, 2]
TDC_PHOTO_DIODE_LE = data[:, 3]
TDC_ION_MCP_LE = data[:, 4]
TDC_ION_MCP_LE_Count = data[:, 5]
TDC_PHOTO_DIODE_LE_Count = data[:, 6]
TDC_DL_X1_LE = data[:, 7]
TDC_DL_X2_LE = data[:, 8]
TDC_DL_Z1_LE = data[:, 9]
TDC_DL_Z2_LE = data[:, 10]
TDC_DL_X1_LE_Count = data[:, 11]
TDC_DL_X2_LE_Count = data[:, 12]
TDC_DL_Z1_LE_Count = data[:, 13]
TDC_DL_Z2_LE_Count = data[:, 14]

if int(run_flip[-10:-5]) >= 3438:
    QDC_EIO0 = data[:, 15]
    QDC_EIO1 = data[:, 16]
    QDC_EIO2 = data[:, 17]
    QDC_EIO3 = data[:, 18]
    QDC_EIO4 = data[:, 19]
    QDC_EIO5 = data[:, 20]

# Compute relevant observables
TOF_LE = TDC_ION_MCP_LE - TDC_PHOTO_DIODE_LE

POS_X = TDC_DL_X1_LE - TDC_DL_X2_LE
POS_Z = TDC_DL_Z1_LE - TDC_DL_Z2_LE

TOF_Count = TDC_ION_MCP_LE_Count - TDC_PHOTO_DIODE_LE_Count

POS_X_Count = TDC_DL_X1_LE_Count - TDC_DL_X2_LE_Count
POS_Z_Count = TDC_DL_Z1_LE_Count - TDC_DL_Z2_LE_Count

TTTL_OP_Beam /= 1e6

# Convert QDC_EIO arrays into logical values
threshold = 1000
QDC_EIO0 = (QDC_EIO0 >= threshold)
QDC_EIO1 = (QDC_EIO1 >= threshold)
QDC_EIO2 = (QDC_EIO2 >= threshold)
QDC_EIO3 = (QDC_EIO3 >= threshold)
QDC_EIO4 = (QDC_EIO4 >= threshold)
QDC_EIO5 = (QDC_EIO5 >= threshold)

# Make cuts on _Counts
TOF_LE_Count_Cut = (TOF_Count == 0)
POS_X_Count_Cut = (POS_X_Count == 0)
POS_Z_Count_Cut = (POS_Z_Count == 0)

Count_Cuts = TOF_LE_Count_Cut & POS_X_Count_Cut & POS_Z_Count_Cut

# Make cuts on remaining observables

# Time of Flight (TOF) spatial Y-axis
# TODO Come up with robust way of making this automatic
lb, ub = np.mean(TOF_LE) - 1000, np.mean(TOF_LE) + 1000
TOF_LE_Cut = (lb <= TOF_LE) & (TOF_LE <= ub)
lb, ub = np.mean(TOF_LE[TOF_LE_Cut]) - 500, np.mean(TOF_LE[TOF_LE_Cut]) + 500
TOF_LE_Cut = (lb <= TOF_LE) & (TOF_LE <= ub)
lb, ub = np.mean(TOF_LE[TOF_LE_Cut]) - 500, np.mean(TOF_LE[TOF_LE_Cut]) + 500
TOF_LE_Cut = (lb <= TOF_LE) & (TOF_LE <= ub)

fig, ax = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)
ax[0].hist(TOF_LE, color="black", histtype="step", label=f"{len(TOF_LE)} Events", bins=200)
ax[1].hist(TOF_LE[TOF_LE_Cut], color="black", histtype="step", label=f"{len(TOF_LE[TOF_LE_Cut])} Events", bins=200)
ax[0].set_ylabel("Counts")
ax[1].set_ylabel("Counts")
ax[0].legend()
ax[1].legend()
plt.close()

# Optical Pumping time (OP)
TTTL_OP_Beam_Cut = (0 <= TTTL_OP_Beam) & (TTTL_OP_Beam <= 4)

fig, ax = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)
ax[0].hist(TTTL_OP_Beam, color="black", histtype="step", label=f"{len(TTTL_OP_Beam)} Events", bins=200)
ax[1].hist(TTTL_OP_Beam[TTTL_OP_Beam_Cut], color="black", histtype="step", label=f"{len(TTTL_OP_Beam[TTTL_OP_Beam_Cut])} Events", bins=200)
ax[0].set_xlabel("Time [$\mu s$]")
ax[1].set_xlabel("Time [$\mu s$]")
ax[0].set_ylabel("Counts")
ax[1].set_ylabel("Counts")
ax[0].legend()
ax[1].legend()
plt.close()

# AND all the cuts together
Cuts = TTTL_OP_Beam_Cut & TOF_LE_Cut & Count_Cuts

# Run time
if int(run_flip[-10:-5]) <= 3438:
    
    Run_time_Cut = (0 <= Run_time) & (Run_time <= 850)
    
    Cuts &= Run_time_Cut
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)
    ax[0].hist(Run_time, color="black", histtype="step", label=f"{len(Run_time)} Events", bins=200)
    ax[1].hist(Run_time[Run_time_Cut], color="black", histtype="step", label=f"{len(Run_time[Run_time_Cut])} Events", bins=200)
    ax[0].set_xlabel("Time [$\mu s$]")
    ax[1].set_xlabel("Time [$\mu s$]")
    ax[0].set_ylabel("Counts")
    ax[1].set_ylabel("Counts")
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
else:
    # Remove all events where AOM is in zero position to remove deadtime between scan and nuclear acquisition start
    bits = 1*QDC_EIO0 + 2*QDC_EIO1 + 4*QDC_EIO2 + 8*QDC_EIO3 + 16*QDC_EIO4 + 32*QDC_EIO5
    
    bits_cut = (bits > 0)
    
    Cuts &= bits_cut
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)
    ax[0].plot(bits, color="black", label=f"{len(bits)} Events")
    ax[1].plot(bits[bits_cut], color="black", label=f"{len(bits[bits_cut])} Events")
    ax[0].set_xlabel("Event")
    ax[1].set_xlabel("Event")
    ax[0].set_ylabel("AOM Step")
    ax[1].set_ylabel("AOM Step")
    ax[0].legend()
    ax[1].legend()
    plt.close()

# Make cut on position of particles

R, X, Z = 100, np.mean(POS_X[Cuts]), np.mean(POS_Z[Cuts])
POS_Cut = (np.sqrt((POS_X - X)**2 + (POS_Z - Z)**2) <= R)
R, X, Z = 100, np.mean(POS_X[Cuts & POS_Cut]), np.mean(POS_Z[Cuts & POS_Cut])
POS_Cut = (np.sqrt((POS_X - X)**2 + (POS_Z - Z)**2) <= R)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
ax[0].hist2d(POS_X[Cuts], POS_Z[Cuts], bins=75, cmap="hot")
ax[1].hist2d(POS_X[Cuts & POS_Cut], POS_Z[Cuts & POS_Cut], bins=50, cmap="hot")
ax[0].set_xlabel("Event")
ax[1].set_xlabel("Event")
ax[0].set_ylabel("AOM Step")
ax[1].set_ylabel("AOM Step")
plt.close()

Cuts &= POS_Cut

# Computing the spectrum

# Start at 1 since all 0 counts are removed
bins = [i for i in range(1, 54)]

# n, bins, _ = fit.hist(bits[Cuts], color="black", histtype="step", label=f"{len(bits[Cuts])} Events", bins=bins)
n, bins = np.histogram(bits[Cuts], bins=bins)

# Calibrate x-axis
AOM_V, AOM_f = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration.csv", unpack=True, delimiter=",")

V_low = 7.63 # [V]
V_high = 9.58 # [V]
dV = (V_high - V_low) / 52

# Compute programmed AOM voltage steps
V = np.array([i * dV for i in range(53)])[1:] + V_low

lock = 64.48

x = 2*np.interp(V, AOM_V, AOM_f) + lock

# First we only fit two peaks to the data to determine the location of the 2->2' transition and hence the magnetic field
p0 = [80, 800, x[np.argmax(n)], 9, 1, 1]

model = lambda x, A, B, x0, h, s, g: peaks(x, A, B, x0, h, s, g)

# Get a better guess using least squares fit
p, _ = curve_fit(model, x, n, p0)
# Perform final fit using a poisson fit
p, E1, E2, sigma, X2 = poisson_fit(model, x, n, p, iter=200)

print("Peaks Fit:")
print(*np.round(p0, 2))
print(*np.round(p, 2))
# print(*np.round(E1, 2))
# print(*np.round(E2, 2))
# print(*np.round(sigma, 2))
# print(np.round(X2 / (len(x) - len(p)), 2))

# Plot spectrum and peaks fit
fig = plt.figure(figsize=(6, 4))
fig.suptitle("Peaks Fit")
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fit = fig.add_subplot(gs[0,0])
res = fig.add_subplot(gs[1,0])
fit.errorbar(x, n, np.sqrt(n), capsize=3, color="black", ls="", label=f"{sum(n)} Events", marker=".")
fit.plot(x, model(x, *p0), color="red", ls="dashed", label="Guess")
fit.plot(x, model(x, *p), color="magenta", label="Fit")
res.errorbar(x, n - model(x, *p), np.sqrt(n), color="black", capsize=3, ls="", marker=".")
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


p0 = [0, 0, 0, 20, 100, 1, B]

model = lambda x, am2, am1, a0, a1, a2, s, B: F2_pi_sublevels(x, am2, am1, a0, a1, a2, x0, h, s, g, B)

# Get a better guess using least squares fit
p, _ = curve_fit(model, x, n, p0, bounds=([0, 0, 0, 0, 0, 0, -10], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 10]))
# Perform final fit using a poisson fit
p, E1, E2, sigma, X2 = poisson_fit(model, x, n, p, iter=200)

print("Sublevel Fit")
print(*np.round(p0, 2))
print(*np.round(p, 2))
# print(*np.round(E1, 2))
# print(*np.round(E2, 2))
# print(*np.round(sigma, 2))
# print(np.round(X2 / (len(x) - len(p)), 2))

# Plot spectrum and sublevel fit
fig = plt.figure(figsize=(6, 4))
fig.suptitle("Sublevel Fit")
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
fit = fig.add_subplot(gs[0,0])
res = fig.add_subplot(gs[1,0])
fit.errorbar(x, n, np.sqrt(n), capsize=3, color="black", ls="", label=f"{sum(n)} Events", marker=".")
fit.plot(x, model(x, *p0), color="red", ls="dashed", label="Guess")
fit.plot(x, model(x, *p), color="magenta", label="Fit")
res.errorbar(x, n - model(x, *p), np.sqrt(n), color="black", capsize=3, ls="", marker=".")
fit.set_xticks([])
res.set_xlabel("AOM Steps")
fit.set_ylabel("Counts")
res.set_ylabel("Counts - Fit")
fit.legend()
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()