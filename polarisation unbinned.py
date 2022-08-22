import numpy as np
import matplotlib.pyplot as plt
from routines.load import load

data = load("/home/felix/fklose/Data/ROOT_Files/output03621.root")

# Extract arrays
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
TTTL_OP_Beam_Cut = (0.83 <= TTTL_OP_Beam) & (TTTL_OP_Beam <= 1.15)

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

# Calibrate x-axis
AOM_V, AOM_f = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration.csv", unpack=True, delimiter=",")
# AOM_V, AOM_f = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_upsweep.csv", unpack=True)
# AOM_V_down, AOM_f_down = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration - 18 Jul 2022.csv", unpack=True)
# AOM_V, AOM_f = np.loadtxt("./AOM Calibrations/Test Calibrations/linear.csv", unpack=True, delimiter=",")

V_low = 7.63 # [V]
V_high = 9.58 # [V]
dV = (V_high - V_low) / 52

# Compute programmed AOM voltage steps taking care to not include the 0th bin
V = np.array([i * dV for i in range(53)])[1:] + V_low

lock = 64.48

frequencies = 2*np.interp(V, AOM_V, AOM_f) + lock

x = frequencies[bits[Cuts] - 1]
y = np.ones(np.shape(bits[Cuts]))

from functions.models import peaks

f = lambda A, B, x0, h, s, g: peaks(x, A, B, x0, h, s, g)
LL = lambda p: - np.sum(y * np.log(f(*p)) - f(*p))

from scipy.optimize import minimize

result = minimize(LL, (1, 1, 365, 10, 1, 1.1/2), bounds=[(0, np.inf), (0, np.inf), (360, 370), (8, 12), (0.1, 10), (0.1, 10)])

print(result.x)

plt.plot(frequencies, peaks(frequencies, *result.x))
plt.show()