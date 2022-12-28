# Import external modules
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
# Import from files
from _load import load_data, generate_histograms
from _models import sublevel_model
from _physics import nuclear_polarization_41K_F2

# If true prints the fitted transition amplitudes
PRINT_AMPLITUDES = False

# Set measurement parameters
V_MIN = 7.79 # Minimum VCO Control Voltage (V)
V_MAX = 9.76 # Maximum VCO Control Voltage (V)
STEPS = 53 # Number of steps in a scan
LOCKPOINT = 64.48 # Frequency of 405 nm lock (MHz)

# Set data input and output paths
OUTPUT_PATH = "./Example"
path_flip = "./Example/output03627.root"
path_norm = "./Example/output03628.root"

data_flip = load_data(path_flip)
data_norm = load_data(path_norm)

CUTS = {
    "BITS"          : (1    , 52    ),
    "X"             : (0    , 20    ),
    "Y"             : (1640 , 1720  ),
    "Z"             : (-25  , 10    ),
    "TTTL_OP_Beam"  : (0    , 4200  )
}

generate_histograms(data_flip, CUTS, OUTPUT_PATH + "/FLIP")
generate_histograms(data_norm, CUTS, OUTPUT_PATH + "/NORM")

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

# Using VCO calibration
V = np.linspace(V_MIN, V_MAX, STEPS)
VCO_V_down, VCO_f_down = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_downsweep.csv", unpack=True)
VCO_V_up  , VCO_f_up   = np.loadtxt("./AOM Calibrations/M1212-aQ50-2/calibration_upsweep.csv", unpack=True)
VCO_V = (np.flip(VCO_V_down) + VCO_V_up) / 2
VCO_f = (np.flip(VCO_f_down) + VCO_f_up) / 2

unbinned_flip = 2*np.interp(V[SPECTRUM_BITS_FLIP], VCO_V, VCO_f) + LOCKPOINT
unbinned_norm = 2*np.interp(V[SPECTRUM_BITS_NORM], VCO_V, VCO_f) + LOCKPOINT

frequencies = 2*np.interp(V, VCO_V, VCO_f) + LOCKPOINT

# Reject first data point as it includes extra counts from not pausing the DAQ on time
frequencies = frequencies[1:]

# Generate bins by selecting midpoints between adjacent frequencies and bin data
bins = [(frequencies[i+1] + frequencies[i]) / 2 for i in range(len(frequencies) - 1)]
bins = [2*frequencies[0] - bins[0]] + bins + [2*frequencies[-1] - bins[-1]]

# Bin data
y_flip, _ = np.histogram(unbinned_flip, bins)
y_norm, _ = np.histogram(unbinned_norm, bins)

# Define the global likelihood function
# This is a sum of two poisson likelihoods, one for the norm spectrum and one for the flip spectrum
def global_poisson_likelihood(x0, h, B, s, g, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip):
    
    global frequencies, y_norm, y_flip
    
    norm = sublevel_model(frequencies, x0, h, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, B, s, g)
    flip = sublevel_model(frequencies, x0, h, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip, B, s, g)
    
    return - np.sum(y_norm * np.log(norm) - norm) - np.sum(y_flip * np.log(flip) - flip)

# Make the minimizer object
m = Minuit(global_poisson_likelihood, x0=363, h=9.94, B=-2.1, s=10, g=1.1,
           am2_flip=1, am1_flip=1, a0_flip=1, ap1_flip=1, ap2_flip=10,
           am2_norm=10, am1_norm=1, a0_norm=1, ap1_norm=1, ap2_norm=1)

# Set parameter limits
m.limits["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip",
         "am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"] = (0, None)
m.limits["x0"] = (361, 364)
m.limits["B"] = (-3, -2)
m.limits["s"] = (0, None)

# Fix parameters
m.fixed["h"] = True
m.fixed["g"] = True

# This tells the minimizer that it is using a log likelihood (important for computing errors)
m.errordef = m.LIKELIHOOD

m.migrad() # Run the minimizer
m.hesse() # Determine errors

print("Fit parameters and errors")
for parameter, value, error in zip(m.parameters, m.values, m.errors):
    if (parameter == "x0") or (parameter == "h") or (parameter == "B") or (parameter == "s") or (parameter == "g") or PRINT_AMPLITUDES:
        if m.fixed[parameter]:
            print(f"{parameter}\t\t:  {np.round(value, 4)} (fixed)")
        else:
            print(f"{parameter}\t\t:  {np.round(value, 4)} +/- {np.round(error, 4)}")

p_norm = m.values["x0", "h", "am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm", "B", "s", "g"]
p_flip = m.values["x0", "h", "am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip", "B", "s", "g"]

# Calculate nuclear polarization and error
P_NORM = nuclear_polarization_41K_F2(m.values["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"], 
                                     m.errors["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"])
P_FLIP = nuclear_polarization_41K_F2(m.values["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"], 
                                     m.errors["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"])

# Print result
print(f"P (Norm)\t:  {np.round(P_NORM[0], 4)} +/- {np.round(P_NORM[1], 4)}")
print(f"P (Flip)\t:  {np.round(P_FLIP[0], 4)} +/- {np.round(P_FLIP[1], 4)}")

# Print fit statistics
chi2_flip = sublevel_model(frequencies, *p_flip)
chi2_flip[y_flip != 0] += y_flip * np.log(y_flip / sublevel_model(frequencies, *p_flip))
chi2_norm = sublevel_model(frequencies, *p_norm)
chi2_norm[y_norm != 0] += y_norm * np.log(y_norm / sublevel_model(frequencies, *p_norm))

chi2 = 2*np.sum(chi2_flip) + 2*np.sum(chi2_norm)

dof = len(y_norm) + len(y_flip) - m.nfit

print(f"chi2\t\t:  {chi2:f}")
print(f"dof\t\t:  {dof:d}")
print(f"chi2/dof\t:  {chi2/dof:f}")

# Plot final spectrum and fits
fig, ((norm, flip), (norm_res, flip_res)) = plt.subplots(2, 2, layout="constrained", figsize=(8, 4), sharex=True, gridspec_kw={})

# Finer steps for plotting fitted function
f_plotting = np.linspace(frequencies[0], frequencies[-1], 1000)

norm.set_title("Norm Polarization")
norm.errorbar(frequencies, y_norm, np.sqrt(y_norm), marker=".", ls="", color="black", markersize=8)
norm.plot(f_plotting, sublevel_model(f_plotting, *p_norm), color="red", lw=2)

flip.set_title("Flip Polarization")
flip.errorbar(frequencies, y_flip, np.sqrt(y_flip), marker=".", ls="", color="black", markersize=8)
flip.plot(f_plotting, sublevel_model(f_plotting, *p_flip), color="red", lw=2)

norm_residuals = y_norm - sublevel_model(frequencies, *p_norm)
norm_res.errorbar(frequencies, norm_residuals, np.sqrt(y_norm), ls="", marker=".", markersize=8, color="black")
norm_res.hlines(0, min(frequencies), max(frequencies), lw=2, color="red")
norm_res.fill_between(frequencies, -np.std(norm_residuals), np.std(norm_residuals), alpha=0.2, color="grey")

flip_residuals = y_flip - sublevel_model(frequencies, *p_flip)
flip_res.errorbar(frequencies, flip_residuals, np.sqrt(y_flip), ls="", marker=".", markersize=8, color="black")
flip_res.hlines(0, min(frequencies), max(frequencies), lw=2, color="red")
flip_res.fill_between(frequencies, -np.std(flip_residuals), np.std(flip_residuals), alpha=0.2, color="grey")
norm.set_ylabel("Counts")
norm.grid()
norm.set_xlim(f_plotting[0], f_plotting[-1])

# flip.set_ylabel("Counts")
flip.grid()
flip.set_xlim(f_plotting[0], f_plotting[-1])

norm_res.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
norm_res.set_ylabel("Data - Model")
norm_res.grid()
norm_res.set_xlim(f_plotting[0], f_plotting[-1])

flip_res.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
# flip_res.set_ylabel("Data - Model")
flip_res.grid()
flip_res.set_xlim(f_plotting[0], f_plotting[-1])

plt.savefig(OUTPUT_PATH + "/fit.png")
plt.savefig(OUTPUT_PATH + "/fit.pdf")

plt.show()