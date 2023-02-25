# %% -*- coding: utf-8 -*-

""" Created on January 20, 2023 // @author: Sarah Shi for figures"""

import numpy as np
import pandas as pd 

import os
import glob 
from pathlib import Path
from scipy import signal

from matplotlib import pyplot as plt
from matplotlib import rc, cm
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

wn_low = 1275
wn_high = 2400

parent_dir = os.path.split(os.getcwd())[0]

bl_i = pd.read_csv('BLi.csv', index_col = 'Wavenumber')
bl_i = bl_i[wn_low:wn_high]

h2o_free = pd.read_csv('H2O_Free.csv', index_col = 'Wavenumber')
h2o_free = h2o_free[wn_low:wn_high]

devol = pd.read_csv('Devolatilized.csv', index_col = 'Wavenumber')
devol = devol[wn_low:wn_high]

dan = pd.read_csv('Dan_Cleaned.csv', index_col = 'Wavenumber')
dan = dan[wn_low:wn_high]

peak1635 = pd.read_csv('1635_PeakBLRemoved.csv', index_col = 'Wavenumber')
peak1635 = peak1635[wn_low:wn_high]

BaselinePCA = pd.read_csv(parent_dir + '/PEAKFIT_FINAL/InputData/Baseline_Avg+PCA.csv', index_col = 'Wavenumber')
BaselinePCA = BaselinePCA[wn_low:wn_high]

H2OPCA = pd.read_csv(parent_dir + '/PEAKFIT_FINAL/InputData/Water_Peak_1635_All.csv', index_col = 'Wavenumber')
H2OPCA = H2OPCA[wn_low:wn_high]

def rescale(abs, range): 
    abs_m = abs * (range / np.abs(abs.iloc[0] - abs.iloc[-1]))
    abs_mb = abs_m - abs_m.iloc[-1]
    return abs_mb

def rescale_peak(abs, range): 
    abs_m = abs * (range / np.max(abs))
    abs_mb = abs_m - abs_m.iloc[-1]
    return abs_mb


# %% 


h2o_free_scale = h2o_free.apply(lambda x: rescale(x, 2))

fig, ax = plt.subplots(3, 2, figsize = (14, 21))
ax = ax.flatten()
ax[0].plot(h2o_free.index, rescale(bl_i, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline_i}}$')
ax[0].plot(h2o_free.index, h2o_free_scale, alpha = 0.5)
ax[0].plot(h2o_free.index, h2o_free_scale.iloc[:, -1], alpha = 0.5, label = 'Devolatilized Spectra, n='+str(np.shape(h2o_free)[1]))
ax[0].plot(h2o_free.index, rescale(bl_i, 2), c = '#171008', lw = 3)
ax[0].set_xlim([1200, 2450])
ax[0].set_ylim([-0.25, 2.25])
ax[0].set_title('A.')
# ax[0].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[0].set_ylabel('Absorbance')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].invert_xaxis()
ax[0].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)

dan_scale = dan.apply(lambda x: rescale(x, 2))
ax[1].plot(bl_i.index, rescale(bl_i, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline_i}}$')
ax[1].plot(dan.index, dan_scale, alpha = 0.5)
ax[1].plot(dan.index, dan_scale.iloc[:, -1], alpha = 0.5, label = 'Decarbonated Spectra, n='+str(np.shape(dan_scale)[1]))
ax[1].plot(h2o_free.index, rescale(bl_i, 2), c = '#171008', lw = 3)
ax[1].set_xlim([1200, 2450])
ax[1].set_ylim([-0.25, 2.25])
ax[1].set_title('B.')
# ax[1].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[1].set_ylabel('Absorbance')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].invert_xaxis()
ax[1].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)

peak1635_scale = peak1635.apply(lambda x: rescale_peak(x, 2))
ax[2].plot(H2OPCA.index, rescale_peak(H2OPCA.Average_1630_Peak, 2), c = '#171008', lw = 3, label = '$\overline{\mathregular{H_2O_{m, 1635}}}$')
ax[2].plot(peak1635_scale.index, peak1635_scale, alpha = 0.5)
ax[2].plot(peak1635_scale.index, peak1635_scale.iloc[:, -1], alpha = 0.5, label = '$\mathregular{H_2O_{m, 1635}}$, Spectra n='+str(np.shape(peak1635)[1]))
ax[2].plot(H2OPCA.index, rescale_peak(H2OPCA.Average_1630_Peak, 2), c = '#171008', lw = 3)
ax[2].set_xlim([1200, 2450])
ax[2].set_ylim([-0.25, 2.25])
ax[2].set_title('C.')
# ax[2].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[2].set_ylabel('Absorbance')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[2].invert_xaxis()
ax[2].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)


devol_scale = devol.apply(lambda x: rescale(x, 2))
ax[3].plot(BaselinePCA.index, rescale(BaselinePCA.Average_Baseline, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline}}$')
ax[3].plot(devol_scale.index, devol_scale, alpha = 0.5)
ax[3].plot(devol_scale.index, devol_scale.iloc[:, -1], alpha = 0.5, label = 'Peak Stripped Spectra, n='+str(np.shape(devol)[1]))
ax[3].plot(BaselinePCA.index, rescale(BaselinePCA.Average_Baseline, 2), c = '#171008', lw = 3)
ax[3].set_xlim([1200, 2450])
ax[3].set_ylim([-0.25, 2.25])
ax[3].set_title('D.')
# ax[3].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[3].set_ylabel('Absorbance')
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].invert_xaxis()
ax[3].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)

BaselinePCA = pd.read_csv(parent_dir + '/PEAKFIT_FINAL/InputData/Baseline_Avg+PCA.csv', index_col = 'Wavenumber')

ax[4].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline}}$')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, label = 'PC1')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_2, c = '#E42211', lw = 2, label = 'PC2')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_3, c = '#5DB147', lw = 2, label = 'PC3')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_4, c = '#F9C300', lw = 2, label = 'PC4')
ax[4].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)
ax[4].set_xlim([1200, 2450])
ax[4].set_title('E.')
# ax[4].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[4].set_ylabel('Absorbance')
ax[4].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[4].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[4].invert_xaxis()

ax[5].plot(H2OPCA.index, H2OPCA.Average_1630_Peak, c = '#171008', lw = 3, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[5].plot(H2OPCA.index, H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC', lw = 2, label = 'PC1')
ax[5].plot(H2OPCA.index, H2OPCA['1630_Peak_PCA_2'], c = '#E42211', lw = 2, label = 'PC2')
ax[5].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 14}, frameon=False)
ax[5].set_xlim([1200, 2450])
# ax[5].set_ylim([-0.4, 1.2])
ax[5].set_title('F.')
ax[5].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
# ax[5].set_ylabel('Absorbance')
ax[5].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[5].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[5].invert_xaxis()

plt.tight_layout()
plt.savefig('AllBaselines.pdf')

# %%
