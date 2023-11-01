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
import matplotlib.gridspec as gridspec
import seaborn as sns

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

# %% 


path_par = os.path.dirname(os.getcwd()) + '/'


Wavenumber = pig.Load_Wavenumber('BaselineAvgPC.npz')
PCmatrix = pig.Load_PC('BaselineAvgPC.npz')
Peak_1635_PCmatrix = pig.Load_PC('H2Om1635PC.npz')
npz_path = path_par + 'NPZTXTFILES/STD/'
npz_FILES = sorted(glob.glob(npz_path + "*.npz"))

standards_path = path_par + 'Inputs/TransmissionSpectra/Standards/'
standards_FILES = sorted(glob.glob(standards_path + "*.csv"))

hj_path = 'DevolatilizedBL/'
FILES = sorted(glob.glob(hj_path + "*"))
DFS_FILES, DFS_DICT = pig.Load_SampleCSV(FILES, wn_high = 2400, wn_low = 1250)
original_length = len(DFS_DICT)

PH_lim = pd.read_csv('PHComparison_lim.csv', index_col=0)
modified_index = [idx + "_fitted_baseline" for idx in PH_lim.index]

DICT_filt = {key: value for key, value in DFS_DICT.items() if key in modified_index}

# Iterate over the files in the HJ_fitted_baseline folder
for hj_file_path, data in DICT_filt.items():
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


# badspec = np.array(['CI_IPGP_B6_1_50x50_256s_sp1', 'CI_IPGP_B6_2_50x50_256s_sp1', 'CI_IPGP_B6_1_50x50_256s_sp2', 'CI_IPGP_NBO_2_2_1_100x100_256s_sp1', 
#                     'CI_Ref_13_1_100x100_256s_sp1', 'CI_Ref_13_1_100x100_256s_sp2', 'CI_Ref_13_1_100x100_256s_sp3', 'CI_Ref_13_1_100x100_256s_sp4', 
#                     'CI_Ref_22_1_100x100_256s_sp1', 'CI_Ref_22_1_100x100_256s_sp2', 'CI_Ref_22_1_100x100_256s_sp3', 
#                     'CI_Ref_23_1_100x100_256s_040523_sp1', 'CI_Ref_23_1_100x100_256s_040523_sp3', 'CI_Ref_23_1_100x100_256s_sp4', 'CI_Ref_23_1_100x100_256s_sp5',
#                     'CI_Ref_25_1_100x100_256s_sp3',
#                     'CI_Ref_bas_1_1_100x100_256s_sp1', 'CI_Ref_bas_1_1_100x100_256s_sp2', 
#                     'CI_Ref_bas_1_2_100x100_256s_sp1', 'CI_Ref_bas_1_2_100x100_256s_sp2', 
#                     'CI_Ref_bas_2_1_100x100_256s_sp1', 
#                     'CI_Ref_bas_2_2_100x100_256s_4sp1', 'CI_Ref_bas_2_2_100x100_256s_sp2', 'CI_Ref_bas_2_2_100x100_256s_sp3', 
#                     'CI_Ref_bas_2_3_100x100_256s_sp1', 
#                     'CI_Ref_bas_3_1_100x100_256s_051423_sp1', 'CI_Ref_bas_3_2_100x100_256s_051423_sp1', 'CI_Ref_bas_3_3_100x100_256s_sp1', 
#                     'CI_Ref_bas_4_1_100x100_256s_sp1', 'CI_Ref_bas_4_1_100x100_256s_sp2',
#                     'LMT_BA3_2_50x50_256s_sp1', 'LMT_BA3_2_50x50_256s_sp2', 'CI_LMT_BA5_2_50x50x_256s_sp1', 
#                     'ND70_02_01_06032022_150x150_sp1',
#                     'ND70_5_2_29June2022_150x150_sp2',  'ND70_05_02_06032022_150x150_sp1', 'ND70_05_02_06032022_150x150_sp2', 'ND70_05_02_06032022_150x150_sp3',
#                     'ND70_05_03_06032022_80x100_sp3', 'ND70_0503_29June2022_95x80_256s_sp2',
#                     'ND70_06_02_75um', 'ND70_6-2_08042022_150x150_sp1', 'ND70_6-2_08042022_150x150_sp2', 'ND70_6-2_08042022_150x150_sp3'])
# badspec_with_suffix = [item + '_fitted_baseline' for item in badspec]


# %% 

import shutil

# Adjust dictionary keys
DFS_DICT_ND70 = {key: value for key, value in DICT_filt.items() if 'CI_' not in key}
DFS_DICT_ND70 = {key: value for key, value in DFS_DICT_ND70.items() if 'LMT_' not in key}
DFS_DICT_ND70 = {key: value for key, value in DFS_DICT_ND70.items() if 'INSOL_' not in key}

adjusted_DFS_DICT = {key.replace("_fitted_baseline", ""): value for key, value in DFS_DICT_ND70.items()}

# source_directory = '../Inputs/TransmissionSpectra/Standards/'  
# source_directory = '../FIGURES/STD/' 
source_directory = 'png/' 
destination_directory = os.path.expanduser('ND70Spec/')  # Adjust the path if needed

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    # Remove the file extension from the filename and check if it's in the adjusted dictionary
    if filename.rsplit('.', 1)[0] in adjusted_DFS_DICT:
        source_file_path = os.path.join(source_directory, filename)
        # Add .CSV to the end of the destination filename
        destination_file_path = os.path.join(destination_directory, filename)
        
        # Copy the file
        shutil.copy(source_file_path, destination_file_path)


# for filename in os.listdir(source_directory):
#     # Remove the _BLcomp.png suffix and the file extension from the filename
#     base_filename = filename.rsplit('_BLcomp.png', 1)[0]
    
#     # Check if the base filename is in the adjusted dictionary and filename ends with _BLcomp.png
#     if base_filename in adjusted_DFS_DICT and filename.endswith('_BLcomp.png'):
#         source_file_path = os.path.join(source_directory, filename)
#         destination_file_path = os.path.join(destination_directory, filename)
        
#         # Copy the file
#         shutil.copy(source_file_path, destination_file_path)


# %%
# %%
# %%

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
    data = csv_data[1000:5500]
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
                    'CI_Ref_bas_9_chip2_100x100_256s_sp4', 'CI_Ref_27_chip2_100x100_256s_sp1'] # 
shortened = ['CI-Ref-6', 'CI-Ref-10', 'CI-Ref-Bas-9', 'CI-Ref-27']
annotate = ['A.', 'B.', 'C.', 'D.']


start = 0 
fig, axs = plt.subplots(2, 2, figsize=(13, 13))
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

    ax.legend(loc = (0.010, 0.58), labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.8, prop={'size': 16}, frameon=False)
    ax.set_xlim([2400, 1250])
    ax.set_ylim([local_min_absorbance-0.075, local_max_absorbance+0.075])

    ax.annotate(annotate[start]+' '+shortened[start], xy=(0.0225, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')

    ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    
    start += 1


axs[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
axs[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)


fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.04)

# Save the figure after all subplots are filled
plt.tight_layout()
plt.savefig('Combined_BLcomp1.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()


# %% 


# %%

standards_path = path_par + 'Inputs/TransmissionSpectra/Standards/'

keys_of_interest = ['ND70_02-01_30June2022_150x150_sp2', 'ND70_03-01_30June2022_150x150_sp1',
                    'ND70_4-2_150x150_08042022_sp1', 'ND70_5_2_29June2022_150x150_sp1', 
                    'ND70_6_2_chip3_100x100_256s_sp1'] # ND70_05_03_06032022_150x50_sp1
shortened = ['ND70-2-01', 'ND70-3-01', 'ND70-4-02', 'ND70-5-02', 'ND70-6-02']
annotate = ['A.', 'B.', 'C.', 'D.', 'E.']

start = 0 
fig, axs = plt.subplots(2, 3, figsize=(21, 14))
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
fig.delaxes(axs[5])
plt.tight_layout()
plt.savefig('ND70_Carbonate.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()


# %% 

keys_of_interest = ['ND70_02-01_30June2022_150x150_sp2', 'ND70_03-01_30June2022_150x150_sp1',
                    'ND70_4-2_150x150_08042022_sp1', 'ND70_5_2_29June2022_150x150_sp2', 
                    'ND70_6_2_chip3_100x100_256s_sp1'] # ND70_05_03_06032022_150x50_sp1
shortened = ['ND70-2-01', 'ND70-3-01', 'ND70-4-02', 'ND70-5-02',  'ND70-6-02']
annotate = ['A.', 'B.', 'C.', 'D.', 'E.']

start = 0 
fig = plt.figure(figsize=(18, 30))
for index, key in enumerate(keys_of_interest):

    hj_file_path = key
    hj_filename = os.path.basename(hj_file_path)
    common_part = hj_filename.split('_fitted_baseline')[0]

    column_names = ["Wavenumber", "Absorbance"]
    csv_data = pd.read_csv(standards_path+common_part+'.csv', names=column_names)
    csv_data = csv_data.set_index('Wavenumber')

    gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gridspec.GridSpec(5, 3)[index, :])

    ax1 = fig.add_subplot(gs[0, 0])
    plotmin = np.round(np.min(csv_data[4200:5400]['Absorbance']) * 20) / 20
    plotmax = np.round(np.max(csv_data[4200:5400]['Absorbance']) * 20) / 20
    ax1.plot(csv_data[4200:5400].index, csv_data[4200:5400]['Absorbance'], color='#171008', lw=2, label='FTIR Spectrum')  # Modify as per your data requirements
    data_H2O4500_1, krige_output_4500_1, PH_4500_krige_1, STN_4500_1 = pig.NearIR_Process(csv_data[4200:5400], 4225, 4650, 'OH')
    data_H2O5200_1, krige_output_5200_1, PH_5200_krige_1, STN_5200_1 = pig.NearIR_Process(csv_data[4200:5400], 4850, 5375, 'H2O')
    ax1.plot(data_H2O5200_1.index, data_H2O5200_1['BL_NIR_H2O'], 'lightsteelblue', label='ALS Baseline')
    ax1.plot(data_H2O4500_1.index, data_H2O4500_1['BL_NIR_H2O'], 'lightsteelblue')
    ax1.set_xlim([5400, 4200])
    ax1.set_ylim([plotmin-0.02, plotmax+0.04])
    ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    ax1.legend(loc='upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.95, prop={'size': 14}, frameon=False)

    max_1 = np.max(data_H2O4500_1['Subtracted_Peak'])
    max_2 = np.max(data_H2O5200_1['Subtracted_Peak'])
    plotmax = np.round(max(max_1, max_2) * 20) / 20 
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data_H2O5200_1.index, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1['Absorbance']), color='#171008', label='Baseline Subtracted Peak')  # Modify as per your data requirements
    ax2.plot(data_H2O4500_1.index, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1['Absorbance']), color='#171008')  # Modify as per your data requirements
    ax2.plot(krige_output_5200_1.index, krige_output_5200_1['Absorbance'] - np.min(krige_output_5200_1['Absorbance']), color='#0C7BDC', label='Gaussian Kriged Peak')
    ax2.plot(krige_output_4500_1.index, krige_output_4500_1['Absorbance'] - np.min(krige_output_4500_1['Absorbance']), color='#417B31')
    ax2.set_xlim([5400, 4200])
    ax2.set_ylim([0, plotmax+0.02])
    ax2.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax2.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    ax2.legend(loc='upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.95, prop={'size': 14}, frameon=False)

    plotmax = np.round(np.max(csv_data[1250:1250+2750]['Absorbance'].to_numpy()), decimals = 0)
    plotmin = np.round(np.min(csv_data[1250:1250+2750]['Absorbance'].to_numpy()), decimals = 0)
    data_H2O3550_1, plot_output_3550_1, PH_3550_1, plotindex1 = pig.MidIR_Process(csv_data[1250:1250+2750], 2300, 4000)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(csv_data[2000:4000].index, csv_data[2000:4000].Absorbance, color='#171008', lw=2, label='FTIR Spectrum')
    ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    ax3.plot(data_H2O3550_1['Absorbance'].index, data_H2O3550_1['BL_MIR_3550'], 'silver', label = 'ALS Baseline')
    ax3.plot(plot_output_3550_1.index, (plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550']), 'r', linewidth = 2, label='Median Filtered Peak')
    ax3.legend(labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.95, prop={'size': 14}, frameon=False)
    ax3.set_xlim([4000, 2000])
    ax3.set_ylim([plotmin-0.25, plotmax+0.5])
    ax3.set_title(annotate[start]+' '+shortened[start])

    local_min_absorbance = np.round(np.min(csv_data[1300:2200]['Absorbance']) * 10) / 10 
    local_max_absorbance = np.round(np.max(csv_data[1300:2200]['Absorbance']) * 10) / 10 
    ax4 = fig.add_subplot(gs[:, 2])
    hj_data = pd.read_csv(hj_path+common_part+'_fitted_baseline.csv', names=column_names)
    hj_data = hj_data.set_index('Wavenumber')
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
    ax4.plot(csv_data[1250:2400].index, csv_data[1250:2400].Absorbance, color='#171008', lw=2, label='FTIR Spectrum')
    ax4.plot(Wavenumber, Baseline_Solve_BP, linestyle = '--', dashes = (2.5, 2.5), lw = 2, color='#171008', label='PyIRoGlass Baseline', zorder=30)
    ax4.plot(Wavenumber, H1635_SOLVE, c = '#E69F00', lw = 1, label = '$\mathregular{H_2O_{m, 1635}}$')
    ax4.plot(Wavenumber, CO2P1515_SOLVE, c = '#E42211', lw = 1, label = '$\mathregular{CO_{3, 1515}^{2-}}$')
    ax4.plot(Wavenumber, CO2P1430_SOLVE, c = '#009E73', lw = 1, label = '$\mathregular{CO_{3, 1430}^{2-}}$')
    ax4.plot(hj_data[1250:2400].index, hj_data[1250:2400].Absorbance, color = '#9A5ABD', lw=2, label='Devolatilized Baseline')
    ax4.legend(loc = (0.015, 0.55), labelspacing = 0.2, handletextpad = 0.5, handlelength = 0.95, prop={'size': 14}, frameon=False)
    ax4.set_xlim([2400, 1250])
    ax4.set_ylim([local_min_absorbance-0.075, local_max_absorbance+0.075])
    ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)

    start += 1

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.09)
fig.supylabel('Absorbance', x = 0.07)
plt.savefig('ND70_Merge.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()


# %%


# %% 
