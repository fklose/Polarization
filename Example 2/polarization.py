# Import external modules
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from tabulate import tabulate
# Import from files
from _load import load_data, compute_observables, generate_histograms
from _models import sublevel_model_F2
from _physics import nuclear_polarization_41K_F2

# Set data input and output paths
OUTPUT_PATH = "./Example 2"
PATH_FLIP = OUTPUT_PATH + "/output03512.root"
PATH_NORM = OUTPUT_PATH + "/output03514.root"

# Set measurement parameters
V_MIN = 7.63        # Minimum VCO Control Voltage (V)
V_MAX = 9.58        # Maximum VCO Control Voltage (V)
STEPS = 53          # Number of steps in a scan
LOCKPOINT = 64.48   # Frequency of 405 nm lock (MHz)

# Set the cuts on the data
CUTS = { #  Name              (Lower, Upper )
            "BITS"          : (1    , 52    ),  # Cut on 1*QDC_EIO0 + 2*QDC_EIO1 + 4*QDC_EIO2 + ...
            "X"             : (0    , 20    ),  # Cut on TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)
            "Y"             : (1640 , 1720  ),  # Cut on TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0] (ns)
            "Z"             : (-25  , 10    ),  # Cut on TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)
            "TTTL_OP_Beam"  : (0    , 4200  )   # Cut on TTTL_OP_Beam (us)
}

### CODE STARTS HERE ###

# Load data from root files (see _load.py)
data_flip = load_data(PATH_FLIP)
data_norm = load_data(PATH_NORM)

# Compute observables
data_flip = compute_observables(data_flip)
data_norm = compute_observables(data_norm)

# Generate .root files containing histograms (see _load.py)
generate_histograms(data_flip, CUTS, OUTPUT_PATH + "/histograms_flip.root")
generate_histograms(data_norm, CUTS, OUTPUT_PATH + "/histograms_norm.root")

# Apply cuts on data and generate the spectrum
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

# Obtain scan frequencies using measured VCO voltage
# Note that the VCO voltage-frequency curves are not the same for sweeping up vs sweeping down
VCO_V_down, VCO_f_down = np.loadtxt("./VCO Calibrations/ZX-95-200-S+/calibration_downsweep.csv", unpack=True)
VCO_V_up  , VCO_f_up   = np.loadtxt("./VCO Calibrations/ZX-95-200-S+/calibration_upsweep.csv", unpack=True)

# DAQ does not now about stepping up or down, so average up- and downsweep together
VCO_V = (np.flip(VCO_V_down) + VCO_V_up) / 2
VCO_f = (np.flip(VCO_f_down) + VCO_f_up) / 2

# Compute scan voltages using measured max and minimum scan voltage
# This assumes an even number of steps
V = np.linspace(V_MIN, V_MAX, STEPS)

# Transform DAQ QDC_EIO bits into frequencies i.e. convert data from steps to frequency
unbinned_flip = 2*np.interp(V[SPECTRUM_BITS_FLIP], VCO_V, VCO_f) + LOCKPOINT
unbinned_norm = 2*np.interp(V[SPECTRUM_BITS_NORM], VCO_V, VCO_f) + LOCKPOINT

# Obtain frequency steps over scan range
frequencies = 2*np.interp(V, VCO_V, VCO_f) + LOCKPOINT

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
    
    norm = sublevel_model_F2(frequencies, x0, h, am2_norm, am1_norm, a0_norm, ap1_norm, ap2_norm, B, s, g)
    flip = sublevel_model_F2(frequencies, x0, h, am2_flip, am1_flip, a0_flip, ap1_flip, ap2_flip, B, s, g)
    
    return - np.sum(y_norm * np.log(norm) - norm) - np.sum(y_flip * np.log(flip) - flip)

# Initialize the minimizer object, assigning the cost_function and an initial guess
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

p_norm = m.values["x0", "h", "am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm", "B", "s", "g"]
p_flip = m.values["x0", "h", "am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip", "B", "s", "g"]

# Calculate nuclear polarization and error
P_NORM = nuclear_polarization_41K_F2(m.values["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"], 
                                     m.errors["am2_norm", "am1_norm", "a0_norm", "ap1_norm", "ap2_norm"])
P_FLIP = nuclear_polarization_41K_F2(m.values["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"], 
                                     m.errors["am2_flip", "am1_flip", "a0_flip", "ap1_flip", "ap2_flip"])

# Attempt to compute some fit statistics
# chi2_flip = sublevel_model(frequencies, *p_flip)
# chi2_flip[y_flip != 0] += y_flip * np.log(y_flip / sublevel_model(frequencies, *p_flip))
# chi2_norm = sublevel_model(frequencies, *p_norm)
# chi2_norm[y_norm != 0] += y_norm * np.log(y_norm / sublevel_model(frequencies, *p_norm))

# chi2 = 2*np.sum(chi2_flip) + 2*np.sum(chi2_norm)
chi2 = 0

dof = len(y_norm) + len(y_flip) - m.nfit

# Generate tables for fit results
parameter_units = {"x0":"MHz", "h":"MHz", "B":"G", "s":"MHz", "g":"MHz"}

parameter_table = []
parameter_table_header = ["Parameter", "Value", "", "Standard Error", "Unit", ""]
for parameter, value, error in zip(m.parameters, m.values, m.errors):
    if (parameter == "x0") or (parameter == "h") or (parameter == "B") or (parameter == "s") or (parameter == "g"):
        if m.fixed[parameter]:
            parameter_table.append([f"{parameter}", f"{np.round(value, 4)}", "", "", parameter_units[parameter], "fixed"])
        else:
            parameter_table.append([f"{parameter}", f"{np.round(value, 4)}", "+/-", f"{np.round(error, 4)}", parameter_units[parameter], ""])

transition_strengths_flip_table = []
transition_strengths_norm_table = []
for parameter, value, error in zip(m.parameters, m.values, m.errors):
    if "flip" in parameter:
        transition_strengths_flip_table.append([f"{parameter[:-5]}", f"{np.round(value, 4)}", "+/-", f"{np.round(error, 4)}", "", "FLIP"])
    if "norm" in parameter:
        transition_strengths_norm_table.append([f"{parameter[:-5]}", f"{np.round(value, 4)}", "+/-", f"{np.round(error, 4)}", "", "NORM"])

# Form the string that contains the different parameter tables
table = tabulate(parameter_table, headers=parameter_table_header) + "\n\n" + \
    tabulate(transition_strengths_flip_table, headers=parameter_table_header) + "\n" + \
    f"---> P (Flip): {np.round(P_FLIP[0], 4)} +/- {np.round(P_FLIP[1], 4)}" + "\n\n" + \
    tabulate(transition_strengths_norm_table, headers=parameter_table_header) + "\n" + \
    f"---> P (Norm): {np.round(P_NORM[0], 4)} +/- {np.round(P_NORM[1], 4)}" + "\n\n" + \
    f"chi2\t\t:  {chi2:f}" + "\n" + \
    f"dof\t\t:  {dof:d}" + "\n" + \
    f"chi2/dof\t:  {chi2/dof:f}"

# Print the table containing fit parameters and
print(table)

# Add the parameter cuts to the table string in order to print them to the file
CUTS_table = []
CUTS_table_header = ["Name", "Lower Cut", "Upper Cut"]
for key, value in zip(CUTS.keys(), CUTS.values()):
    CUTS_table.append([key, value[0], value[-1]])

# Save the table along with a printout of the applied cuts in a parameters.txt file in the OUTPUT_PATH
with open(OUTPUT_PATH + "/parameters.txt", "w") as file:
    file.writelines(table)
    file.write("\n\n")
    file.writelines(tabulate(CUTS_table, headers=CUTS_table_header))
    

# Make a figure and axis objects for plotting final spectrum and fits
fig, ((norm, flip), (norm_res, flip_res)) = plt.subplots(2, 2, layout="constrained", figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios":[5, 3]})

# Finer steps for plotting fitted function
f_plotting = np.linspace(frequencies[0] - np.mean(np.diff(frequencies)), frequencies[-1] + np.mean(np.diff(frequencies)), 10000)

# Plot spectra and fits
norm.errorbar(frequencies, y_norm, np.sqrt(y_norm), marker=".", ls="", color="black", markersize=8)
norm.plot(f_plotting, sublevel_model_F2(f_plotting, *p_norm), color="red", lw=2)

flip.errorbar(frequencies, y_flip, np.sqrt(y_flip), marker=".", ls="", color="black", markersize=8)
flip.plot(f_plotting, sublevel_model_F2(f_plotting, *p_flip), color="red", lw=2)

# Plot associated residuals along with 1-sigma region
norm_residuals = y_norm - sublevel_model_F2(frequencies, *p_norm)
norm_res.errorbar(frequencies, norm_residuals, np.sqrt(y_norm), ls="", marker=".", markersize=8, color="black")
norm_res.hlines(np.mean(norm_residuals), min(f_plotting), max(f_plotting), lw=2, color="dimgrey")
norm_res.fill_between(f_plotting, np.mean(norm_residuals) - np.std(norm_residuals), np.mean(norm_residuals) + np.std(norm_residuals), alpha=0.2, color="grey")

flip_residuals = y_flip - sublevel_model_F2(frequencies, *p_flip)
flip_res.errorbar(frequencies, flip_residuals, np.sqrt(y_flip), ls="", marker=".", markersize=8, color="black")
flip_res.hlines(np.mean(flip_residuals), min(f_plotting), max(f_plotting), lw=2, color="dimgrey")
flip_res.fill_between(f_plotting, np.mean(flip_residuals) - np.std(flip_residuals), np.mean(flip_residuals) + np.std(flip_residuals), alpha=0.2, color="grey")

# Format plots, adding and adjusting titles, ticks and ticklabels
alpha = 0.7

norm.set_title("Norm Spectrum")
norm.set_ylabel("Counts") # ylabels are shared across both columns
norm.set_xlim(f_plotting[0], f_plotting[-1])
norm.grid(alpha=alpha)

flip.set_title("Flip Spectrum")
flip.set_xlim(f_plotting[0], f_plotting[-1])
flip.grid(alpha=alpha)

norm_res.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
norm_res.set_ylabel("Data - Model")
norm_res.set_xlim(f_plotting[0], f_plotting[-1])
norm_res.grid(alpha=alpha)

flip_res.set_xlabel("Frequency wrt $^{39}$K cog (MHz)")
flip_res.set_xlim(f_plotting[0], f_plotting[-1])
flip_res.grid(alpha=alpha)

# Adjust vertical range so both spectra are on the same scale
norm_ylim = norm.get_ylim()
flip_ylim = flip.get_ylim()

ylim = (min(norm_ylim[0], flip_ylim[0]), max(norm_ylim[1], flip_ylim[1]))

norm.set_ylim(ylim)
flip.set_ylim(ylim)

# Save figure
plt.savefig(OUTPUT_PATH + "/fit.png")
plt.savefig(OUTPUT_PATH + "/fit.pdf")

plt.show()