# Fitting 405nm Nuclear Polarization Probe Spectra
This collection of scripts is used to extract and analyze data gathered by TRINAT's nuclear acquisition in order  to measure the nuclear polarization of trapped Potassium atoms.

## Files
* `_constants.py`: Place to store relevant physical constants as global variables.
* `_load.py`: Contains two functions: `load()` and `generate_histograms()`. `load()` is responsible for loading the ntuple and extracting the relevant data as well as converting it to proper units. `generate_histograms()` plots the data extracted using load. It plots the data before and after applying cuts.
* `_models.py`: Stores various models for fitting the data.
* `_physics.py`: Stores functions to compute physical quantities (e.g. nuclear polarization)

* `polarization.py`: Performs the analysis, using functions defined in the files above. Loads data, makes cuts and fitting the model to the generated spectra.

## How to use this script
Inside `polarization.py` store location of `.root` files in `path_flip` and `path_norm`. Set `LOCKPOINT` to the correct frequency that the 405 laser was locked to. If necessary the cuts applied on the data can be changed by modifying the appropriate entry in the `CUTS` dictionary. Entries are listed as `"name": (lower bound, upper bound)`.

Running the script will produce a plot of the two spectra (norm and flip) and will print the fit parameters and the calculated nuclear polarization along with the related uncertainties to the command line.

If `MAKE_HISTOGRAMS` is set to `True` the script will generate relevant histograms in the working directory.

### Prerequisites
To use this script it requires the following pre-requisites:
* A ROOT install (I have tested this using ROOT6 however I have tried to use very basic functionality in order to try and be compatible with ROOT5 but have not tested this).
    * To install ROOT I recommend using conda, but other methods should works just as well as long as you have access to the python interface.

#### Python Modules
* matplotlib: Required for plotting data.
* numpy: Used for making cuts on data and computations.
* iminuit: Used for fitting.
* uncertainties: Used to propagate uncertainties through calculations.
* scipy: Implementation of voigt profile as well as access to their NIST maintained physical constants database.

### Input and Output
As mentioned previously this script only needs the two root files corresponding to opposite polarizations of optical pumping light (i.e. norm and flip or $\sigma^+$ and $\sigma^-$).

As output it will print the fit parameters and save an image of the final plot. If `MAKE_HISTOGRAMS` is set to `True` or `1`, it will also generate histograms of the ntuple data with and without cuts applied to it.

## Fit Procedure
In this last section I want to quickly describe the fit procedure that is used.
The main problem with the polarization data is that is that the subleves are identified using their frequency shifts.
This makes it hard, as with well polarized data we will only see a single peak which will not correspond to the frequency center (frequency of $F=2 \rightarrow 2', m_F=0$) of the transition, as all sublevels apart from the $m_F=0$ subleves are subject to a Zeeman shift ($m_F=0$ can also experience a Stark shift but this does not seem to be an issue in this case although it should be investigated again).
This means that in practice it is very hard to consistently and accurately predict the sublevel populations when only fitting a single scan.

The solution is to fit the data at opposite polarizations at the same time, sharing some parameters between both datasets.
The important ones to share are the location of $F=2 \rightarrow 2', m_F=0$ labelled as $x_0$ and the strength of the magnetic field $B$.
It might also be reasonable to fit $\sigma$ and $\gamma$ independently for each scan but since they seem to be related to the transition linewidth and the probe laser linewidth they are not very likely to change from scan to scan.
For each polarization the sublevel populations are fitted independently.
To determine the best fit a log-likelihood is minimized with the general form being:
$$-\log{L} = - \sum_i \left[y_i \log{(f(x_i, \vec{p}))} - f(x_i, \vec{p})\right]$$
Here $\vec{p}$ is a vector containing the parameters to be minimized.
This likelihood function takes the poisson statistics of the errors on the counts into account in order to avoid biasing the fit.

To perform the simultaneous fit what happens is that the script creates likelihood functions for each dataset $L_{\text{norm}}$ and $L_\text{flip}$. To have the minimizer fit both functions at the same time we simply add their negative logs giving:
$$- \log{L_\text{flip}} - \log{L_\text{norm}}$$
This sum is minimized when both the log likelihoods are at minimum. I have also tried using the product but that does not seem to work.

This is implemented in the function `global_poisson_likelihood()` which is then wrapped inside a lambda function to only expose the fit parameters as arguments which is required for `iminuit`'s minimizer `Minuit` to work (the `global_poisson_likelihood()` function requires some extra parameters to function which are provided when wrapping it in the lambda function).