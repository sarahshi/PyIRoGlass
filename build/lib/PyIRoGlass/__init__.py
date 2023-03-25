# %% 
__author__ = 'Sarah Shi, Henry Towbin'

import os
import copy
import numpy as np
import pandas as pd
import mc3
import warnings

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.use('pgf')
import mc3.utils as mu
import mc3.stats as ms

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate

__all__ = ['trace', 'histogram', 'pairwise', 'rms', 'modelfit', 'subplotter', 'themes',]
themes = {'blue':{'edgecolor':'navy','facecolor':'royalblue','color':'navy'},
    'red': {'edgecolor':'crimson','facecolor':'orangered','color':'darkred'},
    'black':{'edgecolor':'0.3','facecolor':'0.3','color':'black'},
    'green':{'edgecolor':'forestgreen','facecolor':'limegreen','color':'darkgreen'},
    'orange':{'edgecolor':'darkorange','facecolor':'gold','color':'darkgoldenrod'},}
from ._version import __version__

# %%

from PyIRoGlass.dataload import *
from PyIRoGlass.core import *

from PyIRoGlass.reflectance import *
from PyIRoGlass.inversion import *
