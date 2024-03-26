# %% -*- coding: utf-8 -*-

""" Created on July 29, 2022 // @author: Sarah Shi and Henry Towbin"""

# Import packages
import os
import sys
import scipy
import numpy as np
import pandas as pd

sys.path.append('../src/')
import PyIRoGlass as pig

from matplotlib import pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22 # Axes labels

# %% OLIVINE

path_input = os.path.dirname(os.getcwd()) + '/Inputs/'
ref_ol_loader = pig.SampleDataLoader(spectrum_path=path_input+'ReflectanceSpectra/FuegoOl/')
ref_ol_dfs_dict = ref_ol_loader.load_spectrum_directory(wn_high=2800, wn_low=2000)

# Use DHZ parameterization of olivine reflectance index. 
n_ol = pig.reflectance_index(0.72)
ref_fuego = pig.calculate_mean_thickness(ref_ol_dfs_dict, n=n_ol, wn_high=2700, wn_low=2100, plotting=False, phaseol=True)
ref_fuego
# ref_fuego.to_csv('FuegoOlThickness.csv')

# %% 

micro = pd.read_csv('FuegoOlMicrometer.csv', index_col=0)
slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(micro.Thickness_Micrometer.values, ref_fuego.Thickness_M.astype(float))
ccc1 = pig.calculate_CCC(micro.Thickness_Micrometer.values, ref_fuego.Thickness_M.astype(float))
rmse1 = pig.calculate_RMSE(micro.Thickness_Micrometer.values-ref_fuego.Thickness_M.astype(float))

sz = 150
range = [0, 90]
fig, ax = plt.subplots(1, 1, figsize = (7.5, 7.5))
ax.plot(range, range, 'k', lw = 1, zorder = 0)
ax.errorbar(micro.Thickness_Micrometer, ref_fuego.Thickness_M, yerr = ref_fuego.Thickness_STD, xerr = 3, ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.scatter(micro.Thickness_Micrometer, ref_fuego.Thickness_M, s = sz, c = '#0C7BDC', edgecolors='black', linewidth = 0.5, zorder = 15)
ax.set_xlim([20, 90])
ax.set_ylim([20, 90])

ax.annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 2)), xy=(0.02, 0.8775), xycoords="axes fraction", fontsize=16)
ax.annotate("CCC="+str(np.round(ccc1, 2)), xy=(0.02, 0.96), xycoords="axes fraction", fontsize=16)
ax.annotate("RMSE="+str(np.round(rmse1, 2))+"; RRMSE="+str(np.round(pig.calculate_RRMSE(micro.Thickness_Micrometer.values, ref_fuego.Thickness_M.values)*100, 2))+'%', xy=(0.02, 0.92), xycoords="axes fraction", fontsize=16)
ax.annotate("m="+str(np.round(slope1, 2)), xy=(0.02, 0.84), xycoords="axes fraction", fontsize=16)
ax.annotate("b="+str(np.round(intercept1, 2)), xy=(0.02, 0.80), xycoords="axes fraction", fontsize=16)

ax.set_xlabel('Micrometer Thickness (µm)')
ax.set_ylabel('Reflectance FTIR Thickness (µm)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('OlThicknessTest.pdf',bbox_inches='tight', pad_inches = 0.025)

# %% BASALTIC GLASS - High Quality Spectra

ref_gl_loader = pig.SampleDataLoader(spectrum_path=path_input+'ReflectanceSpectra/rf_ND70/')
ref_gl_dfs_dict = ref_gl_loader.load_spectrum_directory(wn_high=2850, wn_low=1700)

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
n_gl = 1.546
ref_nd70_high = pig.calculate_mean_thickness(ref_gl_dfs_dict, n=n_gl, wn_high=2850, wn_low=1700, plotting=False, phaseol=False)
ref_nd70_high
# ref_nd70_high.to_csv('ND70_Thickness_HighQuality.csv')

# %%
