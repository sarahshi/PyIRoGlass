# PyIRoGlass
[![PyPI](https://badgen.net/pypi/v/PyIRoGlass)](https://pypi.org/project/PyIRoGlass/)
[![Build Status](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/SarahShi/PyIRoGlass/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/pyiroglass/badge/?version=latest)](https://pyiroglass.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main/graph/badge.svg)](https://codecov.io/gh/SarahShi/PyIRoGlass/branch/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/406815894.svg)](https://zenodo.org/doi/10.5281/zenodo.10883628)

PyIRoGlass is a Bayesian MCMC-founded Python algorithm, written in the open-source language Python3, for determining $\mathrm{H_2O}$ and $\mathrm{CO_2}$ species concentrations in the transmission FTIR spectra of basaltic to andesitic glasses. We leverage a database of naturally degassed melt inclusions and back-arc basin basalts to delineate the fundamental shape and variability of the baseline underlying the $\mathrm{CO_{3}^{2-}}$ and $\mathrm{H_2O_{m, 1635}}$ peaks, in the mid-infrared region. PyIRoGlass employs Bayesian inference and Markov Chain Monte Carlo sampling to fit all probable baselines and peaks, solving for best-fit parameters and capturing covariance to offer robust uncertainty estimates.

## Manuscript
Find the [PyIRoGlass manuscript](https://doi.org/10.30909/vol.07.02.471501) published at Volcanica on for a more detailed description of the development and validation of the method. If you use this package in your work, please cite: 

```console
Shi, S., Towbin, W. H., Plank, T., Barth, A., Rasmussen, D., Moussallam, Y., Lee, H. J. and Menke, W. (2024) “PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to FTIR spectra of basaltic-andesitic glasses”, Volcanica, 7(2), pp. 471–501. doi: 10.30909/vol.07.02.471501.
```

```
@article{Shietal2024,
    doi       = {10.30909/vol.07.02.471501},
    url       = {https://doi.org/10.30909/vol.07.02.471501},
    year      = {2024},
    volume    = {7},
    number    = {2},
    pages     = {471-501},
    author    = {Shi, Sarah C. and Towbin, W. Henry and Plank, Terry and Barth, Anna and Rasmussen, Daniel and Moussallam, Yves and Lee, Hyun Joo and Menke, William},
    title     = {PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to FTIR spectra of basaltic-andesitic glasses},
    journal   = {Volcanica}
}
```

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