# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi for plotting"""

# Import packages

import os
import sys
import glob
import numpy as np
import pandas as pd
import mc3

sys.path.append('../src/')
import PyIRoGlass as pig

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 16})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20 # Axes labels


# %% 

path_par = os.path.dirname(os.getcwd()) + '/'

standards_path = path_par + 'Inputs/TransmissionSpectra/Standards/'
standards_FILES = sorted(glob.glob(standards_path + "*.csv"))

hj_path = 'DevolatilizedBL/'
FILES = sorted(glob.glob(hj_path + "*"))
DFS_FILES, DFS_DICT = pig.Load_SampleCSV(FILES, wn_high = 2400, wn_low = 1250)

Wavenumber = pig.Load_Wavenumber('BaselineAvgPC.npz')
PCmatrix = pig.Load_PC('BaselineAvgPC.npz')
Peak_1635_PCmatrix = pig.Load_PC('H2Om1635PC.npz')
npz_path = path_par + 'NPZTXTFILES/STD/'
npz_FILES = sorted(glob.glob(npz_path + "*.npz"))


# Iterate over the files in the HJ_fitted_baseline folder
for hj_file_path, data in DFS_DICT.items():
    # Extract the common part of the filename (e.g., samplename_fitted_baseline)
    hj_filename = os.path.basename(hj_file_path)
    common_part = hj_filename.split('_fitted_baseline')[0]

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.set_title(common_part)
    ax.set_xlabel('Wavenumber $(\mathregular{cm^{-1}})$')
    ax.set_ylabel('Absorbance')

    column_names = ["Wavenumber", "Absorbance"]
    csv_data = pd.read_csv(standards_path+common_part+'.csv', names=column_names)
    csv_data = csv_data.set_index('Wavenumber')
    ax.plot(csv_data[1250:2400].index, csv_data[1250:2400].Absorbance, label='FTIR Spectrum')

    hj_data = pd.read_csv(hj_path+common_part+'_fitted_baseline.csv', names=column_names)
    hj_data = hj_data.set_index('Wavenumber')
    ax.plot(hj_data[1250:2400].index, hj_data[1250:2400].Absorbance, label='Devolatilized Baseline')

    mc3_output = np.load(npz_path+common_part+'.npz')
    Nvectors=5
    PC_BP = mc3_output['bestp'][0:Nvectors]
    PC_STD = mc3_output['stdp'][0:Nvectors]
    Baseline_Solve_BP = PC_BP * PCmatrix.T
    Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

    m_BP, b_BP = mc3_output['bestp'][-2:None]
    m_STD, b_STD = mc3_output['stdp'][-2:None]
    Line_BP = pig.Linear(Wavenumber, m_BP, b_BP) 
    Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

    ax.plot(Wavenumber, Baseline_Solve_BP, label='PyIRoGlass Baseline')
    ax.legend(loc = 'upper left', labelspacing = 0.5, handletextpad = 0.25, handlelength = 1.00, prop={'size': 14}, frameon=False)
    ax.set_xlim([2400, 1250])
    # ax.set_ylim([-0.25, 2.5])
    ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    plt.savefig('png/'+common_part+'_BLcomp.png', bbox_inches='tight', pad_inches = 0.025)


# %%

keys_of_interest = ['CI_Ref_6_1_100x100_256s_040523_sp1', 'CI_Ref_10_chip1_100x100_256s_sp3',
                    'CI_Ref_bas_9_chip2_100x100_256s_sp4', 'ND70_05_03_06032022_80x100_sp3']
shortened = ['CI_Ref_6', 'CI_Ref_10', 'CI_Ref_bas_9', 'ND70-5']
annotate = ['A.', 'B.', 'C.', 'D.']


start = 0 
fig, axs = plt.subplots(2, 2, figsize=(14, 14))
axs = axs.ravel()
for index, key in enumerate(keys_of_interest):
    hj_file_path = key
    hj_filename = os.path.basename(hj_file_path)
    common_part = hj_filename.split('_fitted_baseline')[0]


    ax = axs[index]

    column_names = ["Wavenumber", "Absorbance"]
    csv_data = pd.read_csv(standards_path+common_part+'.csv', names=column_names)
    csv_data = csv_data.set_index('Wavenumber')

    # Determine the local min and max absorbance for this plot
    local_min_absorbance = min(csv_data[1300:2200].Absorbance)
    local_max_absorbance = max(csv_data[1300:2200].Absorbance)
    
    local_min_absorbance = round(local_min_absorbance * 10) / 10
    local_max_absorbance = round(local_max_absorbance * 10) / 10

    ax.plot(csv_data[1250:2400].index, csv_data[1250:2400].Absorbance, color='#171008', lw=2, label='FTIR Spectrum')

    hj_data = pd.read_csv(hj_path+common_part+'_fitted_baseline.csv', names=column_names)
    hj_data = hj_data.set_index('Wavenumber')
    ax.plot(hj_data[1250:2400].index, hj_data[1250:2400].Absorbance, color = '#9A5ABD', lw=2, label='Devolatilized Baseline')

    mc3_output = np.load(npz_path+common_part+'.npz')
    Nvectors=5
    CO2P_BP = mc3_output['bestp'][-11:-5]
    H2OmP1635_BP = mc3_output['bestp'][-5:-2]
    CO2P_STD = mc3_output['stdp'][-11:-5]
    H2OmP1635_STD = mc3_output['stdp'][-5:-2]
    H2OmP1635_STD[0] = H2OmP1635_STD[0]

    PC_BP = mc3_output['bestp'][0:Nvectors]
    PC_STD = mc3_output['stdp'][0:Nvectors]
    Baseline_Solve_BP = PC_BP @ PCmatrix.T
    Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

    m_BP, b_BP = mc3_output['bestp'][-2:None]
    m_STD, b_STD = mc3_output['stdp'][-2:None]
    Line_BP = pig.Linear(Wavenumber, m_BP, b_BP) 
    Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

    H1635_BP = H2OmP1635_BP @ Peak_1635_PCmatrix.T
    H1635_BP = np.asarray(H1635_BP).ravel()
    CO2P1430_BP = pig.Gauss(Wavenumber, CO2P_BP[0], CO2P_BP[1], A=CO2P_BP[2])
    CO2P1515_BP = pig.Gauss(Wavenumber, CO2P_BP[3], CO2P_BP[4], A=CO2P_BP[5])
    H1635_SOLVE = H1635_BP + Baseline_Solve_BP
    CO2P1515_SOLVE = CO2P1515_BP + Baseline_Solve_BP
    CO2P1430_SOLVE = CO2P1430_BP + Baseline_Solve_BP
    carbonate = pig.Carbonate(mc3_output['meanp'], Wavenumber, PCmatrix, Peak_1635_PCmatrix, Nvectors)

    ax.plot(Wavenumber, Baseline_Solve_BP, linestyle = '--', dashes = (2.5, 2.5), lw = 2, color='#171008', label='PyIRoGlass Baseline', zorder=30)
    ax.plot(Wavenumber, H1635_SOLVE, c = '#E69F00', lw = 1, label = '$\mathregular{H_2O_{m, 1635}}$')
    ax.plot(Wavenumber, CO2P1515_SOLVE, c = '#E42211', lw = 1, label = '$\mathregular{CO_{3, 1515}^{2-}}$')
    ax.plot(Wavenumber, CO2P1430_SOLVE, c = '#009E73', lw = 1, label = '$\mathregular{CO_{3, 1430}^{2-}}$')

    ax.legend(loc = (0.015, 0.65), labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.95, prop={'size': 14}, frameon=False)
    ax.set_xlim([2400, 1250])
    ax.set_ylim([local_min_absorbance-0.075, local_max_absorbance+0.075])

    ax.annotate(annotate[start]+' '+shortened[start], xy=(0.0225, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')

    ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    
    start += 1

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.03)
fig.supylabel('Absorbance', x = 0.03)

# Save the figure after all subplots are filled
plt.tight_layout()
plt.savefig('Combined_BLcomp.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()


# %% 
