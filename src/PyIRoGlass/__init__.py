# %%

__author__ = 'Sarah Shi'

import os
import warnings
import numpy as np
import pandas as pd
import mc3

import matplotlib as mpl
from matplotlib import pyplot as plt
import mc3.plots as mp
import mc3.utils as mu
import mc3.stats as ms

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate

from PyIRoGlass.core import *
from PyIRoGlass.thickness import *
from PyIRoGlass.inversion import *

from ._version import __version__
