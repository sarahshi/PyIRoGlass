# %% -*- coding: utf-8 -*-

""" Created on January 20, 2023 // @author: Sarah Shi for figures"""

import numpy as np
import pandas as pd 

import os
import glob 
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import rc, cm
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

parent_dir = os.path.split(os.getcwd())[0]

h2o_free = pd.read_csv('H2O_Free.csv', index_col = 'Wavenumber')
co2_free = pd.read_csv('CO2_Free.csv', index_col = 'Wavenumber')
co2_free1 = pd.read_csv('CO2_free_baselines_Water_removed_5-19-21.csv', index_col = 'Wavenumber')
dan = pd.read_csv('Dan_Cleaned.csv', index_col = 'Wavenumber')

BaselinePCA = pd.read_csv(parent_dir + '/PEAKFIT_FINAL/InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv(parent_dir + '/PEAKFIT_FINAL/InputData/Water_Peak_1635_All.csv')

# %%

fig, ax = plt.subplots(2, 2, figsize = (14, 14))
ax = ax.flatten()

wn = h2o_free.index
for i in h2o_free.columns:
    abs = h2o_free[i]
    ax[0].plot(wn, abs, alpha = 0.5)
ax[0].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = 'Baseline')
ax[0].set_xlim([1275, 2400])
ax[0].set_ylim([-0.75, 2.5])
ax[0].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[0].set_ylabel('Absorbance')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].invert_xaxis()
ax[0].legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)
ax[0].text(2375, 2.4, 'Best Devolatilized Spectra', ha = 'left', fontsize = 16)


wn = co2_free1.index
for i in co2_free1.columns:
    abs = co2_free1[i]
    ax[1].plot(wn, abs, alpha = 0.5)
ax[1].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = 'Baseline')
ax[1].set_xlim([1275, 2400])
ax[1].set_ylim([-0.75, 2.5])
ax[1].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[1].set_ylabel('Absorbance')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].invert_xaxis()
ax[1].legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)

wn = dan.index
for i in dan.columns:
    abs = dan[i]
    ax[2].plot(wn, abs, alpha = 0.5)
ax[2].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = 'Baseline')
ax[2].set_xlim([1275, 2400])
ax[2].set_ylim([-0.75, 2.5])
ax[2].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[2].set_ylabel('Absorbance')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[2].invert_xaxis()
ax[2].legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)

plt.tight_layout()

# %%
