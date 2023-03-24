# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages

import os
import sys
import glob
import numpy as np
import pandas as pd
import mc3

import PyIRoGlass as pig

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# %% 

# Get working paths 
path_input = os.getcwd() + '/Inputs/COLAB_BINDER/'

# %% 

# Change paths to direct to folder with reflectance FTIR spectra
REF_PATH_OL = path_input + 'REF/OL/'
REF_OL_FILES = sorted(glob.glob(REF_PATH_OL + "*"))

REF_OL_DFS_FILES, REF_OL_DFS_DICT = pig.Load_SampleCSV(REF_OL_FILES, wn_high = 2700, wn_low = 2100)

# Use DHZ parameterization of olivine reflectance index. 
n_ol = pig.Reflectance_Index(0.72)

REF_FUEGO = pig.Thickness_Processing(REF_OL_DFS_DICT, n = n_ol, wn_high = 2700, wn_low = 2100, remove_baseline = True, plotting = False, phaseol = True)

REF_FUEGO

# %% 

REF_PATH_GL = path_input + 'REF/GL/'
REF_GL_FILES = sorted(glob.glob(REF_PATH_GL + "*"))

REF_GL_DFS_FILES, REF_GL_DFS_DICT = pig.Load_SampleCSV(REF_GL_FILES, wn_high = 2850, wn_low = 1700)

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
n_gl = 1.546

REF_GLASS = pig.Thickness_Processing(REF_GL_DFS_DICT, n = n_gl, wn_high = 2850, wn_low = 1700, remove_baseline = True, plotting = False, phaseol = False)

REF_GLASS

# %%

# Change paths to direct to folder with transmission FTIR spectra 
TRANS_PATHS = path_input + 'TRANS/'
TRANS_FILES = sorted(glob.glob(TRANS_PATHS + "*"))

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATH = path_input + 'Colab_Binder_ChemThick.csv'

# Change to be what you want the prefix of your output files to be. 
OUTPUT_PATH = 'RESULTS'


MICOMP, THICKNESS = pig.Load_ChemistryThickness(CHEMTHICK_PATH)

DFS_FILES, DFS_DICT = pig.Load_SampleCSV(TRANS_FILES, wn_high = 5500, wn_low = 1000)
DF_OUTPUT, FAILURES = pig.Run_All_Spectra(DFS_DICT, OUTPUT_PATH)
DF_OUTPUT.to_csv('FINALDATA/TRANS_DF.csv')

T_ROOM = 25 # C
P_ROOM = 1 # Bar

N = 500000
DENSITY_EPSILON, MEGA_SPREADSHEET = pig.Concentration_Output(DF_OUTPUT, N, THICKNESS, MICOMP, T_ROOM, P_ROOM)
MEGA_SPREADSHEET.to_csv('TRANS_H2OCO2.csv')
DENSITY_EPSILON.to_csv('FINALDATA/TRANS_DensityEpsilon.csv')
MEGA_SPREADSHEET

# %% 
