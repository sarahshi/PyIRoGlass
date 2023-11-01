# PyIRoGlass
[![PyPI](https://badgen.net/pypi/v/PyIRoGlass)](https://pypi.org/project/PyIRoGlass/)
[![Build Status](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/pyiroglass/badge/?version=latest)](https://pyiroglass.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main/graph/badge.svg)](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb)
[![Python 3.7](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyIRoGlass is a Bayesian MCMC-founded Python algorithm for determining volatile concentrations and speciation for $\mathrm{H_2O_{t, 3550}}$, $\mathrm{H_2O_{m, 1635}}$, $\mathrm{CO_{3, 1515}^{2-}}$, $\mathrm{CO_{3, 1430}^{2-}}$, $\mathrm{H_2O_{m, 5200}}$, and $\mathrm{OH_{4500}}$ from basaltic to andesitic transmission FTIR spectra. PyIRoGlass is written in the open-source language Python3 with the $\mathrm{MC^3}$ package, allowing for the proper sampling of parameter space and the determination of volatile concentrations with uncertainties. 

## Documentation
Check the [documentation](https://pyiroglass.readthedocs.io/en/latest/) for a run-through of the PyIRoGlass code. And be sure to read the manuscript.

## Run on the Cloud 
If you do not have Python installed locally, run PyIRoGlass on [Google Colab](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb).

## Run and Install Locally
Obtain a version of Python between 3.7 and 3.11 if you do not already have it installed. PyIRoGlass can be installed with one line. Open terminal and type the following:

```
pip install PyIRoGlass
```

Make sure that you keep up with the latest version of PyIRoGlass. To upgrade to the latest version of PyIRoGlass, open terminal and type the following: 

```
pip install PyIRoGlass --upgrade
```

Quantifying volatile concentrations in magmas is critical for understanding magma storage, phase equilibria, and eruption processes. We present PyIRoGlass, an open-source Python package for quantifying $\mathrm{H_2O}$ and $\mathrm{CO_2}$ species concentrations in the transmission FTIR spectra of basaltic to andesitic glasses. We leverage a database of naturally degassed melt inclusions and back-arc basin basalts to delineate the fundamental shape and variability of the baseline underlying the $\mathrm{CO_{3}^{2-}}$ and $\mathrm{H_2O_{m, 1635}}$ peaks, in the mid-infrared region. All Beer-Lambert Law parameters are examined to quantify associated uncertainties. PyIRoGlass employs Bayesian inference and Markov Chain Monte Carlo sampling to fit all probable baselines and peaks, solving for best-fit parameters and capturing covariance to offer robust uncertainty estimates. Results from PyIRoGlass agree with independent analysis of experimental devolatilized glasses (within 6\%) and interlaboratory standards (13\% for $\mathrm{H_2O}$, 9\% for $\mathrm{CO_2}$). The open-source nature of PyIRoGlass ensures its adaptability and evolution as more data become available.