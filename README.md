# PyIRoGlass
[![PyPI](https://badgen.net/pypi/v/PyIRoGlass)](https://pypi.org/project/PyIRoGlass/)
[![Build Status](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/pyiroglass/badge/?version=latest)](https://pyiroglass.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main/graph/badge.svg)](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

PyIRoGlass is a Bayesian MCMC-founded Python algorithm, written in the open-source language Python3, for determining $\mathrm{H_2O}$ and $\mathrm{CO_2}$ species concentrations in the transmission FTIR spectra of basaltic to andesitic glasses. We leverage a database of naturally degassed melt inclusions and back-arc basin basalts to delineate the fundamental shape and variability of the baseline underlying the $\mathrm{CO_{3}^{2-}}$ and $\mathrm{H_2O_{m, 1635}}$ peaks, in the mid-infrared region. PyIRoGlass employs Bayesian inference and Markov Chain Monte Carlo sampling to fit all probable baselines and peaks, solving for best-fit parameters and capturing covariance to offer robust uncertainty estimates.

## Preprint
Find the [preprint on EarthArXiv](https://eartharxiv.org/repository/view/6193/) on for a more detailed description of the development and validation of the method. 

## Documentation
Read the [documentation](https://pyiroglass.readthedocs.io/en/latest/) for a run-through of the PyIRoGlass code. 

## Run on the Cloud 
If you do not have Python installed locally, run PyIRoGlass on [Google Colab](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb).

## Run and Install Locally
Obtain a version of Python between 3.8 and 3.12 if you do not already have it installed. PyIRoGlass can be installed with one line. Open terminal and type the following:

```
pip install PyIRoGlass
```

Make sure that you keep up with the latest version of PyIRoGlass. To upgrade to the latest version of PyIRoGlass, open terminal and type the following: 

```
pip install PyIRoGlass --upgrade
```