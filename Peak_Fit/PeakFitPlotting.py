# %% -*- coding: utf-8 -*-
""" Created on August 1, 2021 // @author: Sarah Shi """

# Import packages

import os
import sys
import time
import glob
import warnings 
import mc3
import numpy as np
import pandas as pd

import scipy.signal as signal
import scipy.interpolate as interpolate

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20 # Axes labels

# %% PCA Component Plotting

BaselinePCA = pd.read_csv('./InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv('./InputData/Water_Peak_1635_All.csv')
H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

spec = pd.read_csv('./InputData/AC4_OL49_021920_30x30_H2O_a.csv')
spec1 = pd.read_csv('./InputData/AC4_OL53_101220_256s_30x30_a.csv')

x = pd.read_csv('./InputData/x.csv') 

data_H2O5200_1 = pd.read_csv('./InputData/data_H2O5200_1.csv')
data_H2O5200_2 = pd.read_csv('./InputData/data_H2O5200_2.csv')
data_H2O5200_3 = pd.read_csv('./InputData/data_H2O5200_3.csv')

data_H2O4500_1 = pd.read_csv('./InputData/data_H2O4500_1.csv')
data_H2O4500_2 = pd.read_csv('./InputData/data_H2O4500_2.csv')
data_H2O4500_3 = pd.read_csv('./InputData/data_H2O4500_3.csv')

krige_output_5200_1 = pd.read_csv('./InputData/krige_output_5200_1.csv')
krige_output_5200_2 = pd.read_csv('./InputData/krige_output_5200_2.csv')
krige_output_5200_3 = pd.read_csv('./InputData/krige_output_5200_3.csv')

krige_output_4500_1 = pd.read_csv('./InputData/krige_output_4500_1.csv')
krige_output_4500_2 = pd.read_csv('./InputData/krige_output_4500_2.csv')
krige_output_4500_3 = pd.read_csv('./InputData/krige_output_4500_3.csv')

plot_output_3550_1 = pd.read_csv('./InputData/plot_output_3550_1.csv') 
plot_output_3550_2 = pd.read_csv('./InputData/plot_output_3550_2.csv') 
plot_output_3550_3 = pd.read_csv('./InputData/plot_output_3550_3.csv') 

data_H2O3550_1 = pd.read_csv('./InputData/data_H2O3550_1.csv') 
data_H2O3550_2 = pd.read_csv('./InputData/data_H2O3550_2.csv') 
data_H2O3550_3 = pd.read_csv('./InputData/data_H2O3550_3.csv') 

Baseline_Solve_BP = pd.read_csv('./InputData/Baseline_Solve_BP.csv') 
Baseline_Array_Plot = pd.read_csv('./InputData/Baseline_Array_Plot.csv').to_numpy()
linearray = pd.read_csv('./InputData/linearray.csv') 

H1635_BP = pd.read_csv('./InputData/H1635_BP.csv') 
CO2P1515_BP = pd.read_csv('./InputData/CO2P1515_BP.csv') 
CO2P1430_BP = pd.read_csv('./InputData/CO2P1430_BP.csv') 
carbonate = pd.read_csv('./InputData/carbonate.csv') 


# %% 


BaselinePCA = pd.read_csv('./InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv('./InputData/Water_Peak_1635_All.csv')
H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

n1 = 15
sz = 150
fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax[0].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[0].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_1[0::n1], c = '#0C7BDC', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC1}}$', zorder = 20)
ax[0].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC1}}$',)

ax[1].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[1].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_2[0::n1], c = '#E42211', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC2}}$', zorder = 15)
ax[1].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_2, c = '#E42211', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC2}}$',)

ax[2].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[2].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_3[0::n1], c = '#5DB147', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC3}}$', zorder = 10)
ax[2].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_3, c = '#5DB147', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC3}}$',)

ax[3].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[3].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_4[0::n1], c = '#F9C300', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC4}}$', zorder = 5)
ax[3].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_4, c = '#F9C300', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC4}}$',)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])

ax[0].set_ylim([-0.6, 0.6])
ax[1].set_ylim([-0.6, 0.6])
ax[2].set_ylim([-0.6, 0.6])
ax[3].set_ylim([-0.6, 0.6])

ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)

ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)

ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[3].annotate("D.", xy=(0.0375, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)

plt.tight_layout()
plt.savefig('BL+PCVectors_Subplot.pdf', bbox_inches='tight', pad_inches = 0.025)


# %% 


H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax[0].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[0].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_1'], c = '#0C7BDC', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + $\mathregular{\overline{H_2O_{m, 1635}}_{PC1}}$', zorder = 20)

ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC',  lw = 2.0, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ - $\mathregular{\overline{H_2O_{m, 1635}}_{PC1}}$', zorder = 20)

ax[2].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[2].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_2'], c = '#E42211', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + $\mathregular{\overline{H_2O_{m, 1635}}_{PC2}}$', zorder = 20)

ax[3].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[3].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_2'], c = '#E42211',  lw = 2.0, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ - $\mathregular{\overline{H_2O_{m, 1635}}_{PC2}}$', zorder = 20)

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[3].annotate("D.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])
ax[0].set_ylim([-0.15, 1.15])
ax[1].set_ylim([-0.15, 1.15])
ax[2].set_ylim([-0.15, 1.15])
ax[3].set_ylim([-0.15, 1.15])

ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)
plt.tight_layout()
plt.savefig('H2Om1635+PCVectors_Subplot+.pdf', bbox_inches='tight', pad_inches = 0.025)

# %% 

fig = plt.figure(figsize = (15, 15))
gs = fig.add_gridspec(ncols = 4, nrows = 5)
ax0 = plt.subplot(gs[0, :])
ax1 = plt.subplot(gs[1, 0:2])
ax2 = plt.subplot(gs[2, 0:2])
ax3 = plt.subplot(gs[1:3, 2:4])
ax4 = plt.subplot(gs[3:5, 0:2])
ax5 = plt.subplot(gs[3:5, 2:4])

ax0.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
t1 = ax0.text(5200, 0.175, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#0C7BDC', pad=0.03))
t2 = ax0.text(4500, 0.175, '$\mathregular{OH_{4500}^-}$', ha = 'center', fontsize = 16)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#5DB147', pad=0.03))
t3 = ax0.text(3500, 0.175, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 16)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#E42211', pad=0.03))
t4 = ax0.text(1635, 1.55, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 16)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax0.text(1475, 0.175, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 15, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
t6 = ax0.text(2350, 0.175, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 16)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#7E2F8E', pad=0.03))
ax0.annotate("A.", xy=(0.0075, 0.87), xycoords="axes fraction", fontsize=20, weight='bold')
ax0.set_xlim([1250, 5500])
ax0.set_ylim([0, 3])
ax0.invert_xaxis()

ax1.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax1.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.Absorbance_Hat, c = '#0C7BDC', lw = 1, label = 'Median Filtered Peak')
ax1.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax1.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax1.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1, label = 'ALS Baselines')
ax1.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.annotate("B.", xy=(0.015, 0.87), xycoords="axes fraction", fontsize=20, weight='bold')
t1 = ax1.text(5150, 0.6625, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#0C7BDC', pad=0.03))
t2 = ax1.text(4485, 0.5225, '$\mathregular{OH_{4500}^-}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#5DB147', pad=0.03))
ax1.legend(loc = 'lower left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax1.set_xlim([4100, 5500])
ax1.set_ylim([0.4, 0.7])
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax1.axes.xaxis.set_ticklabels([])
ax1.invert_xaxis()


ax2.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1.Absorbance), c = 'k', lw = 1, label = 'Baseline Subtracted Peak')
ax2.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2['Subtracted_Peak'] - np.min(krige_output_5200_2.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3['Subtracted_Peak'] - np.min(krige_output_5200_3.Absorbance), c = 'k', lw = 1)
ax2.plot(krige_output_5200_1.Wavenumber, krige_output_5200_1.Absorbance - np.min(krige_output_5200_1.Absorbance), c = '#0C7BDC', lw = 2, label = 'Gaussian Kriged Peak')
ax2.plot(krige_output_5200_2.Wavenumber, krige_output_5200_2.Absorbance - np.min(krige_output_5200_2.Absorbance), c = '#074984', lw = 2)
ax2.plot(krige_output_5200_3.Wavenumber, krige_output_5200_3.Absorbance - np.min(krige_output_5200_3.Absorbance), c = '#6DAFEA', lw = 2)
ax2.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2['Subtracted_Peak'] - np.min(krige_output_4500_2.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3['Subtracted_Peak'] - np.min(krige_output_4500_3.Absorbance), c = 'k', lw = 1)
ax2.plot(krige_output_4500_1.Wavenumber, krige_output_4500_1.Absorbance - np.min(krige_output_4500_1.Absorbance), c = '#417B31', lw = 2)
ax2.plot(krige_output_4500_2.Wavenumber, krige_output_4500_2.Absorbance - np.min(krige_output_4500_2.Absorbance), c = '#5DB147', lw = 2) 
ax2.plot(krige_output_4500_3.Wavenumber, krige_output_4500_3.Absorbance - np.min(krige_output_4500_3.Absorbance), c = '#8DC87E', lw = 2)
ax2.annotate("C.", xy=(0.015, 0.87), xycoords="axes fraction", fontsize=20, weight='bold')
t1 = ax2.text(5150, 0.0015, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#0C7BDC', pad=0.03))
t2 = ax2.text(4485, 0.0015, '$\mathregular{OH_{4500}^-}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#5DB147', pad=0.03))
ax2.text(5150, 0.0275, 'PH=0.025±0.001\n1.82±0.35 wt.%', ha = 'center', fontsize = 16, color='#0C7BDC')
ax2.text(4485, 0.0165, 'PH=0.015±0.001\n1.25±0.26 wt.%', ha = 'center', fontsize = 16, color='#417B31')
ax2.legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 14}, frameon=False)
ax2.set_xlim([4100, 5500])
ax2.set_ylim([0, 0.04])
ax2.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax2.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax2.invert_xaxis()


ax3.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax3.plot(data_H2O3550_1.Wavenumber, data_H2O3550_1.BL_MIR_3550, linestyle = '--', dashes = (3, 4), c = '#5E5E5E', lw = 1.5, label = 'ALS Baseline')
ax3.plot(plot_output_3550_1.Wavenumber, plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5, label = 'Median Filtered Peak')
ax3.plot(plot_output_3550_2.Wavenumber, plot_output_3550_2['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)
ax3.plot(plot_output_3550_3.Wavenumber, plot_output_3550_3['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)
t3 = ax3.text(3325, 0.6, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 18)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#E42211', pad=0.03))
t4 = ax3.text(1635, 1.5, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax3.text(1475, 0.5, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 18, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
t6 = ax3.text(2350, 0.6, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 18)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#7E2F8E', pad=0.03))
ax3.text(3325, 0.15, 'PH=2.17±0.02\n2.25±0.16 wt.%', ha = 'center', fontsize = 16, color='#E42211')
ax3.annotate("D.", xy=(0.015, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax3.legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 14}, frameon=False)
ax3.set_xlim([1250, 4000])
ax3.set_ylim([0, 3])
ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax3.invert_xaxis()

ax4.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax4.plot(x.Wavenumber, Baseline_Array_Plot[1, :], c = '#5E5E5E', lw = 1.5, label = 'MC$\mathregular{{^3}}$ Sampled Baselines')
for i in range(0, 100, 2):
    ax4.plot(x.Wavenumber, Baseline_Array_Plot[i, :], c = '#5E5E5E', lw = 1.5)
ax4.plot(x.Wavenumber, H1635_BP['Absorbance'], c = '#E69F00', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$')
ax4.plot(x.Wavenumber, CO2P1515_BP['Absorbance'], c = '#E42211', lw = 2, label = '$\mathregular{CO_{3, 1515}^{2-}}$')
ax4.plot(x.Wavenumber, CO2P1430_BP['Absorbance'], c = '#009E73', lw = 2, label = '$\mathregular{CO_{3, 1430}^{2-}}$')
ax4.plot(x.Wavenumber, carbonate['Absorbance'], c = '#9A5ABD', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Spectrum')
ax4.plot(x.Wavenumber, Baseline_Solve_BP['Absorbance'], linestyle = '--', dashes = (2, 2), c = 'k', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Baseline')
t4 = ax4.text(1635, 1.475, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax4.text(1470, 0.925, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 18, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
ax4.text(1635, 0.51, 'PH=0.66±0.01\n1.31±0.19 wt.%', ha = 'center', fontsize = 16, color='#E69F00')
ax4.text(1475, 0.625, 'PH=0.11±0.01\n737±36 ppm', ha = 'center', fontsize = 16, color='k')


ax4.annotate("E.", xy=(0.015, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')

handles, labels = ax4.get_legend_handles_labels()
order = [0,2,3,4,1,6,5]
ax4.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = (0.01, 0.525), labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.85, prop={'size': 14}, frameon=False)
ax4.set_xlim([1250, 2400])
ax4.set_ylim([0.4, 1.6])
ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax4.invert_xaxis()


spec_fit = spec[(spec.Wavenumber>1250) & (spec.Wavenumber<2400)]
ax5.axhline(y=0, color='k', linestyle='--', dashes=(8, 8), linewidth=0.75)
ax5.scatter(spec_fit.Wavenumber[2::5], spec_fit.Absorbance[2::5] - carbonate['Absorbance'].values[2::5], c = 'k', label = 'MC$\mathregular{{^3}}$ Fit Residual')
ax5.errorbar(spec_fit.Wavenumber[2::5], spec_fit.Absorbance[2::5] - carbonate['Absorbance'].values[2::5], 
             yerr = np.abs(spec_fit.Absorbance[2::5] - carbonate['Absorbance'].values[2::5]) * 0.1, lw = 0.5, c = 'k', fmt='none')
ax5.legend(loc = 'lower left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 14}, frameon=False)
ax5.annotate("F.", xy=(0.015, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax5.set_xlim([1250, 2400])
ax5.set_ylim([-0.03, 0.03])
ax5.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax5.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax5.invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.03)
fig.supylabel('Absorbance', x = 0.03)

plt.tight_layout()
plt.savefig('AllPeaks.pdf', bbox_inches='tight', pad_inches = 0.025)


# %% 

# %%


fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax0 = ax[0]
ax1 = ax[1]
ax3 = ax[2]
ax4 = ax[3]

ax0.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax0.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax0.axhline(3, c = 'k')
ax0.axvspan(5150, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax0.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
ax0.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax0.axvspan(2300, 2400, 0, 3, color = '#7E2F8E', lw = 0, alpha=0.2)
ax0.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax0.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
ax0.scatter(4000, 3.0, s = 200, marker = '|', c = 'k')
ax0.text(4750, 2.825, 'Near IR', ha = 'center', fontsize = 16)
ax0.text(2637.5, 2.825, 'Mid IR', ha = 'center', fontsize = 16)
t1 = ax0.text(5050, -0.1, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#0C7BDC', pad=0.03))
t2 = ax0.text(4250, -0.1, '$\mathregular{OH_{4500}^-}$', ha = 'center', fontsize = 16)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#5DB147', pad=0.03))
t3 = ax0.text(3380, -0.1, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 16)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#E42211', pad=0.03))
t4 = ax0.text(1725, 1.55, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 16)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax0.text(1550, -0.1, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 15, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
t6 = ax0.text(2350, -0.1, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 16)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#7E2F8E', pad=0.03))
ax0.annotate("A.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax0.set_xlim([1200, 5500])
ax0.set_ylim([-0.25, 3])
ax0.invert_xaxis()

ax1.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax1.plot(spec1.Wavenumber, spec1.Absorbance + 0.325, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax1.axvspan(5125, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax1.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
t1 = ax1.text(5200, 0.42, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#0C7BDC', pad=0.03))
t2 = ax1.text(4500, 0.42, '$\mathregular{OH_{4500}^-}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#5DB147', pad=0.03))
ax1.annotate("B.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax1.set_xlim([4100, 5500])
ax1.set_ylim([0.4, 0.7])
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax1.invert_xaxis()

ax3.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax3.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax3.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax3.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax3.axvspan(2300, 2400, 0, 3, color = '#7E2F8E', lw = 0, alpha=0.2)
ax3.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
t3 = ax3.text(3380, -0.1, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 18)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#E42211', pad=0.03))
t4 = ax3.text(1725, 1.55, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax3.text(1550, -0.1, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 18, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
t6 = ax3.text(2350, -0.1, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 18)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#7E2F8E', pad=0.03))
ax3.annotate("C.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax3.set_xlim([1200, 4000])
ax3.set_ylim([-0.25, 3])
ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax3.invert_xaxis()

ax4.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax4.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax4.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax4.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax4.axvspan(2300, 2400, 0, 3, color = '#7E2F8E', lw = 0, alpha=0.2)
ax4.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
t4 = ax4.text(1635, 1.475, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#F9C300', pad=0.03))
t5 = ax4.text(1470, 0.1, '$\mathregular{CO_{3}^{2-}}$\n$\mathregular{_{1515, 1430}}$', ha = 'center', fontsize = 18, linespacing=0.5)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='k', pad=0.03))
t6 = ax4.text(2280, 0.1, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 18)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='#7E2F8E', pad=0.03))
ax4.annotate("D.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax4.set_xlim([1200, 2400])
ax4.set_ylim([0, 1.6])
ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax4.invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)


plt.tight_layout()
# plt.savefig('AllPeaks_Prelim.pdf', bbox_inches='tight', pad_inches = 0.025)

# %%
