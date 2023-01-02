# Fitting 405nm Nuclear Polarization Probe Spectra
This collection of scripts is used to extract and analyze data gathered by TRINAT's nuclear acquisition to measure the nuclear polarization of trapped potassium atoms.

## Files
* `_constants.py`: Stores any relevant physical constants (e.g. $\mu_B$) in a single place.
* `_load.py`: Contains three functions: `load_data()`, `compute_observables()` and `generate_histograms()`:
    * `load_data()` extracts relevant data from the given `.root` file. It also converts units where needed (`TDC_*` $\rightarrow$ $ns$, `TTTL_*` $\rightarrow$ $\mu s$). The function returns a dictionary of numpy arrays.
    * `compute_observables()` takes the dictionary generated by `load_data()` and computes relevant observables (e.g. `Y = TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0]`, ...)
    * `generate_histograms()` plots the data extracted using `load_data()` after running `compute_observables()`. It plots the data before and after applying cuts.
* `_models.py`: Stores models used to fit the data (currently only $F=2 \rightarrow F'$ is implemented).
* `_physics.py`: Stores functions to compute physical quantities (e.g. nuclear polarization).

* `polarization.py`: Performs the analysis using functions defined in the files above. 
Loads data, makes cuts and fits the model to the generated spectra.
Generates a printout of the fit parameters and saves them to a `.txt` file.

## How to use this script
1. Set `OUTPUT_PATH`. I recommend making a folder containing the two `.root` files for FLIP and NORM and using that to store the output of this script to keep all files used in this analysis in the same place for better organization.
2. Set `path_flip` and `path_norm` to point to the correct `.root` files.
3. Set the measurement parameters `V_MIN`, `V_MAX`, `STEPS` and `LOCKPOINT`. These should be recorded for every measurement.
4. Run the script.
5. Check the histograms generated in `.root` files called `histograms_*.root` in the output folder to make sure that cuts are done properly.
Adjust `CUTS` if necessary and run again.

### Prerequisites
#### ROOT
To use this script it requires the following pre-requisites:
* A ROOT install that provides access to ROOT's python interface
    * I have tested this with ROOT6. ROOT5 should also work since I am using very basic functionality, but I have not tested this.

#### Python Modules
* Matplotlib: Required for plotting data.
* Numpy: Used for making cuts on data and computations.
* iMinuit: Used for fitting.
* uncertainties: Used to propagate statistical uncertainties through nuclear polarization calculations.
* scipy: Obtain physical constants from NIST CODATA database. Implementation of Voigt profile.
* tabulate: For making nice tables.

These can be installed using pip:

`pip install matplotlib numpy iminuit uncertainties scipy tabulate`

### Input and Output
1. **fit.pdf/.png**: Image of the two spectra along with fits and residuals. In the residuals plot $\pm$ 1-$\sigma$ of the residuals is indicated by a shaded bar.
2. **histograms_*.root**: Saves the data relevant to the analysis in `.root` files. Included is data before applying cuts and after.
3. **parameters.txt**: Copy of the script output. Includes fit parameters and statistics, nuclear polarization and the cuts applied to the data.

## Fitting Procedure
The main problem with the polarization data is that the sublevels are identified using their frequency shifts.
This is hard since the sublevel shifts depend on three parameters: $x_0$ which is a "frequency" shift related to the isotope shift, $B$ the magnetic field (Zeeman effect) and the laser power (Stark shift) (in its current form the script does not account for a Stark shift and I am only mentioning it here for completeness).

Accounting for the "frequency" shift and the Zeeman shift at the same time leaves us with two unrelated parameters that both affect the frequency locations of the sublevel transitions.
Essentially we have an expression with two unknowns leaving us with infinitely many solutions.
One approach to circumvent this issue would be to fix one of the parameters, but the frequency shifts are usually not known to high enough precision and the magnetic field strength inside the trap is hard to measure in situ.
The second approach is to fit two spectra at opposite pumping polarizations (norm and flip) simultaneously.
This takes advantage of the symmetry of the Zeeman effect when probing it with $\pi$-polarized light.
The magnetic field $B$ controls the "width" of each spectrum, i.e. the splitting between the $m_F = \pm 2$ and $m_F = \pm 1$.
By using spectra taken at roughly opposite polarization we essentially provide data for both $m_F = +2$ and $m_F = -2$ and hence can determine this total width more precisely.

To fit this global model the sum of two log-likelihoods is minimized.
A single dataset $(x_i, y_i)$ can be fitted to some model $f(x; \vec{\alpha})$ where $\vec{\alpha}$ is a vector of parameters by minimizing:
$$-\mathcal{L} = - \log{L} = - \sum_i \left[y_i \log{(f(x_i; \vec{p}))} - f(x_i, \vec{p})\right]$$

To perform a simultaneous fit we add the likelihood functions computed using the norm and flip data:
```math
- \mathcal{L}^\text{Global} = - \mathcal{L}^\text{Flip} - \mathcal{L}^\text{Norm}
```
```math
-\mathcal{L}^\text{Global} = - \sum_i^{N^\text{Flip}} \left[y^\text{Flip}_i \log{(f(x^\text{Flip}_i; \vec{\alpha}, \vec{b}^\text{Flip}))} - f(x^\text{Flip}_i; \vec{\alpha}, \vec{b}^\text{Flip})\right]
```
```math
- \sum_i^{N^\text{Norm}} \left[y^\text{Norm}_i \log{(f(x^\text{Norm}_i; \vec{\alpha}, \vec{b}^\text{Norm}))} - f(x^\text{Norm}_i; \vec{\alpha}, \vec{b}^\text{Norm})\right]
```
Here we have 3 parameter vectors $\vec{\alpha}$ and $\vec{b}$ where:
```math
\vec{\alpha} = \begin{pmatrix}x_0 \quad h \quad B \quad s \quad g\end{pmatrix},
```
contains the shared parameters and
```math
\vec{b} = \begin{pmatrix}
    a_{-2} \quad a_{-1} \quad a_{0} \quad a_{+1} \quad a_{+2}
\end{pmatrix},
```
contains the sublevel populations for norm polarization $\vec{b}^\text{Norm}$ and flip polarization $\vec{b}^\text{Flip}$.
The model fitted to each spectrum is given by:
```math
f(x; \vec{\alpha}, \vec{b}) = \sum_{m_F=-1}^{1} s_{1, m_F} a_{m_F} V\left(f - x_{0} + h - \frac{2}{3} m_F \bar{\mu}_B B, g, s \right) \nonumber
```
```math
+ \sum_{m_F=-2}^{2} s_{2, m_F} a_{m_F} V\left(x - x_{0} - \frac{1}{3} m_F \bar{\mu}_B B, g, s \right)
```
Here $s_{F, m_F}$ are the transition strengths (a weighting factor), $V(x, g, s)$ is the Voigt profile (a convolution of a Lorentzian and Gaussian distribution) where $g$ and $s$ are the widths of the Lorentzian and Gaussian part respectively and $\bar{\mu}_B = \mu_B / h$ where $\mu_B$ is the Bohr magneton.