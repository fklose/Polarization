# Import external modules
import numpy as np
import matplotlib.pyplot as plt
# Import from files
from _load import load_data, generate_histograms
from _models import sublevel_model
from _physics import nuclear_polarization_41K_F2

PLOT = False
SAVE_OUTPUT = False
LOCKPOINT = 64.48 # MHz

path_flip = "./output03627.root"
path_norm = "./output03628.root"

data_flip = load_data(path_flip)
data_norm = load_data(path_norm)

CUTS = {
    "BITS"          : (1    , 52    ),
    "X"             : (0    , 20    ),
    "Y"             : (1640 , 1720  ),
    "Z"             : (-25  , 10    ),
    "TTTL_OP_Beam"  : (0    , 4200  )
}

# generate_histograms(data_flip, CUTS, False, False)
# generate_histograms(data_norm, CUTS, False, False)

# Apply cuts and generate final spectrum
SPECTRUM_BITS_FLIP = data_flip["BITS"][
        ((CUTS["X"][0]      <=  data_flip["X"])      & (data_flip["X"]    <=  CUTS["X"][1]))      \
    &   ((CUTS["Y"][0]      <=  data_flip["Y"])      & (data_flip["Y"]    <=  CUTS["Y"][1]))      \
    &   ((CUTS["Z"][0]      <=  data_flip["Z"])      & (data_flip["Z"]    <=  CUTS["Z"][1]))      \
    &   ((CUTS["TTTL_OP_Beam"][0] < data_flip["TTTL_OP_Beam"]) & (data_flip["TTTL_OP_Beam"] < CUTS["TTTL_OP_Beam"][1]))
]

SPECTRUM_BITS_NORM = data_norm["BITS"][
        ((CUTS["X"][0]      <=  data_norm["X"])      & (data_norm["X"]    <=  CUTS["X"][1]))      \
    &   ((CUTS["Y"][0]      <=  data_norm["Y"])      & (data_norm["Y"]    <=  CUTS["Y"][1]))      \
    &   ((CUTS["Z"][0]      <=  data_norm["Z"])      & (data_norm["Z"]    <=  CUTS["Z"][1]))      \
    &   ((CUTS["TTTL_OP_Beam"][0] < data_norm["TTTL_OP_Beam"]) & (data_norm["TTTL_OP_Beam"] < CUTS["TTTL_OP_Beam"][1]))
]

# TODO use iminuit -> same minimizer that is found in root

def convert_to_frequency(spectrum_bits):
    
    bins = list(set(sorted(spectrum_bits)))
    counts = np.zeros(max(spectrum_bits) + 1)
    for value in spectrum_bits:
        counts[value] += 1

    x, y = bins, counts

    V = np.linspace(7.79, 9.76, 53)
    VCO_V_down, VCO_f_down = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_downsweep.csv", unpack=True)
    VCO_V_up  , VCO_f_up   = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_upsweep.csv", unpack=True)
    VCO_V = (np.flip(VCO_V_down) + VCO_V_up) / 2
    VCO_f = (np.flip(VCO_f_down) + VCO_f_up) / 2

    x = 2*np.interp(V[x], VCO_V, VCO_f) + LOCKPOINT

    return x, y

x_flip, y_flip = convert_to_frequency(SPECTRUM_BITS_FLIP)
x_norm, y_norm = convert_to_frequency(SPECTRUM_BITS_NORM)

# Reject first bin
x_flip = x_flip[1:]
y_flip = y_flip[1:]
x_norm = x_norm[1:]
y_norm = y_norm[1:]

# Using VCO calibration
V = np.linspace(7.79, 9.76, 53)
VCO_V_down, VCO_f_down = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_downsweep.csv", unpack=True)
VCO_V_up  , VCO_f_up   = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_upsweep.csv", unpack=True)
VCO_V = (np.flip(VCO_V_down) + VCO_V_up) / 2
VCO_f = (np.flip(VCO_f_down) + VCO_f_up) / 2

unbinned_flip = 2*np.interp(V[SPECTRUM_BITS_FLIP], VCO_V, VCO_f) + LOCKPOINT
unbinned_norm = 2*np.interp(V[SPECTRUM_BITS_NORM], VCO_V, VCO_f) + LOCKPOINT

bits = np.array([i for i in range(0, 53)])
frequencies = 2*np.interp(V[bits], VCO_V, VCO_f) + LOCKPOINT

# Make custom frequency bins for fitting
bins = [(frequencies[i+1] + frequencies[i]) / 2 for i in range(len(frequencies) - 1)]
bins = [2*frequencies[0] - bins[0]] + bins + [2*frequencies[-1] - bins[-1]]

# Reject 0th bin
bins = bins[1:]

fig, (norm, flip) = plt.subplots(2, 1, layout="constrained")

norm.set_title("Norm Polarization")
counts_norm, _, _ = norm.hist(unbinned_norm, bins=bins, histtype="step")
norm.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
norm.set_ylabel("Counts")

flip.set_title("Flip Polarization")
counts_flip, _, _ = flip.hist(unbinned_flip, bins=bins, histtype="step")
flip.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
flip.set_ylabel("Counts")

# plt.show()
plt.close()

from _models import sublevel_model_pdf

def global_poisson_likelihood(p, x, y_norm, y_flip):
    
    x0, h, B, s, g, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip = p
    
    norm = sublevel_model(x, x0, h, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, B, s, g)
    flip = sublevel_model(x, x0, h, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip, B, s, g)
    
    return - np.sum(y_norm * np.log(norm) - norm) - np.sum(y_flip * np.log(flip) - flip)

from iminuit import Minuit

c = lambda x0, h, B, s, g, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip: \
    global_poisson_likelihood([x0, h, B, s, g, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip], 
                              frequencies[1:], y_norm, y_flip)

m = Minuit(c, x0=363, h=9.94, B=-2.1, s=10, g=1.1,
           am2_flip=1, am1_flip=1, a0_flip=1, ap1_flip=1, ap2_flip=1,
           am2_norm=1, am1_norm=1, a0_norm=1, ap1_norm=1, ap2_norm=1)

m.limits["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip",
         "am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"] = (0, 1e9)

m.limits["x0"] = (361, 364)
m.limits["B"] = (-3, -2)
m.limits["s"] = (0, None)

m.fixed["h"] = True
m.fixed["g"] = True

m.errordef = m.LIKELIHOOD

m.migrad()
m.hesse()

m.draw_profile("a0_flip")

print("Fit parameters and uncertainties")
for p, v, e in zip(m.parameters, m.values, m.errors):
    s = f"{p}: {np.round(v, 2)} +/- {np.round(e, 2)}"
    print(s)

p_norm = m.values["x0", "h", "am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm", "B", "s", "g"]
p_flip = m.values["x0", "h", "am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip", "B", "s", "g"]

# Calculate nuclear polarization and error
P_NORM = nuclear_polarization_41K_F2(m.values["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"], 
                                     m.errors["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"])
P_FLIP = nuclear_polarization_41K_F2(m.values["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"], 
                                     m.errors["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"])

print(f"P (Norm): {P_NORM}")
print(f"P (Flip): {P_FLIP}")

fig, (norm, flip) = plt.subplots(1, 2, layout="constrained", figsize=(12, 4))

norm.set_title("Norm Polarization")
norm.hist(unbinned_norm, bins=bins, histtype="step")
norm.plot(frequencies[1:], sublevel_model(frequencies[1:], *p_norm), color="black")
norm.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
norm.set_ylabel("Counts")

flip.set_title("Flip Polarization")
flip.hist(unbinned_flip, bins=bins, histtype="step")
flip.plot(frequencies[1:], sublevel_model(frequencies[1:], *p_flip), color="black")
flip.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
flip.set_ylabel("Counts")

plt.show()
# plt.close()

# TODO compute chisq separately for each spectrum and not for the global fit.
# TODO clean up fit functions and move into separate module