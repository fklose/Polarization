# Import values from NIST CODATA data base
from scipy.constants import physical_constants

# Bohr magneton scaled to MHz / G
mu_B = physical_constants["Bohr magneton in Hz/T"][0] / 1e6 / 1e4