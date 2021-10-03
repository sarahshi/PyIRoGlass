# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi and Henry Towbin """

# %% Import packages

import numpy as np
import pandas as pd
import mc3
import os
import glob
import warnings

import scipy.signal as signal
import scipy.interpolate as interpolate
import Automated_Baselines as Auto_B # Handles automated baseline functions
import MC3_BL_Plotting as mc3plots

from pykrige import OrdinaryKriging
from pathlib import Path

from matplotlib import pyplot as plt

# %%


def Load_SampleCSV(paths, H2O_wn_high, H2O_wn_low): 

    """The Load_SampleCSV function takes the inputs of the path to a directory with all sample CSVs, 
    wavenumber high, wavenumber low values. The function outputs a dictionary of each sample's associated 
    wavenumbers and absorbances."""

    CO2_dfs = []
    H2O_dfs = []
    files = []

    for path in paths:
        head_tail = os.path.split(path)
        file = head_tail[1][0:-4]

        df = pd.read_csv(path, names= ['Wavenumber', 'Absorbance'])
        df.set_index('Wavenumber', inplace = True)
        H2O_spec = df.loc[H2O_wn_low:H2O_wn_high]
        H2O_dfs.append(H2O_spec)
        files.append(file)

    H2O_zipobj = zip(files, H2O_dfs)
    H2O_dfs_dict = dict(H2O_zipobj)

    return files, H2O_dfs_dict


def Load_PCA(PCA_Path):

    """The Load_PCA function takes the input of a path to a CSV of predetermined PCA components. 
    The function returns a dataframe with the PCA components."""

    wn_high = 2200
    wn_low = 1275

    PCA_DF = pd.read_csv(PCA_Path, index_col = "Wavenumber")
    PCA_DF = PCA_DF[wn_low:wn_high]
    PCA_matrix = np.matrix(PCA_DF.to_numpy())

    return PCA_matrix


def Load_Wavenumber(Wavenumber_Path):

    """The Load_PCA function takes the input of a path to a CSV of the wavenumbers associated with PCA components. 
    The function returns a dataframe with the wavenumbers."""

    wn_high = 2200
    wn_low = 1275

    Wavenumber_DF = pd.read_csv(Wavenumber_Path, index_col = "Wavenumber")
    Wavenumber_DF = Wavenumber_DF[wn_low:wn_high]
    Wavenumber = np.array(Wavenumber_DF.index)

    return Wavenumber


def Load_ChemistryThickness(ChemistryThickness_Path):

    """The Load_ChemistryThickness function takes the input of a path to a CSV with MI chemistry or thickness. 
    The function returns dataframes with these data."""

    ChemistryThickness = pd.read_csv(ChemistryThickness_Path)
    ChemistryThickness.set_index('Sample', inplace = True)

    return ChemistryThickness

def Gauss(x, mu, sd, A=1):

    """The Gauss function takes the inputs of the wavenumbers of interest, center of peak, 
    standard deviation (or width of peak), and amplitude. The function outputs a Gaussian fit 
    for the CO3^{2-} doublet peaks at 1515 and 1430 cm^-1 peaks."""

    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))

    return G


def Linear(x, m, b):

    """The Linear function takes the inputs of wavenumbers of interest, tilt, and offset for 
    adjusting the model data. The function returns a linear offset taking the form of y = mx+b."""

    b = np.ones_like(x) * b
    m = np.arange(0, max(x.shape)) * m 
    tilt_offset = m + b 

    return tilt_offset


def Carbonate(P, x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors = 5): 

    """The Carbonate function takes in the inputs of fitting parameters P, wavenumbers x, PCA matrix, 
    number of PCA vectors of interest. The function calculates the molecular H2O_{1635} peak, 
    CO3^{2-} Gaussian peaks, linear offset, and baseline. The function then returns the model data."""

    PCA_Weights = np.array([P[0:Nvectors]])
    Peak_Weights = np.array([P[-5:-2]])

    peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude = P[Nvectors:-5]
    m, b = P[-2:None]

    Peak_1635 = Peak_Weights * Peak_1635_PCAmatrix.T
    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = Linear(x, m, b) 

    baseline = PCA_Weights * PCAmatrix.T
    model_data = baseline + linear_offset + Peak_1635 + G1515 + G1430
    model_data = np.array(model_data)[0,:]

    return model_data


def NearIR_Process(data, H2O_wn_low, H2O_wn_high, peak): 

    """The NearIR_Process function inputs the dictionary data, the Near-IR H2O (5200 peak) or OH (4500 peak) wavenumbers of interest, 
    and the H2O or OH peak of interest and returns a dataframe of the absorbance data in the region of interest, median filtered data, 
    baseline subtracted absorbance, the kriged data output, the peak height, and the signal to noise ratio. 
    The function median filters the peak absorbance (given the noise inherent to these Near-IR peaks for MI), 
    fits an asymmetric least squares determined baseline to the peak, and subtracts the baseline to determine the 
    peak absorbance. The baseline-subtracted peak is then kriged, to further reduce noise and to obtain peak height. 
    The signal to noise ratio is then determined and if the ratio is high, a warning is outputted and the 
    user can consider the usefulness of these peaks. This function is used three times with slightly different 
    H2O wavenumber ranges, so that uncertainty can be assessed."""

    data_H2O = data[H2O_wn_low:H2O_wn_high]
    data_output = pd.DataFrame(columns = ['Absorbance', 'Absorbance_Hat', 'BL_NIR_H2O', 'Subtracted_Peak'], index = data_H2O.index)
    data_output['Absorbance'] = data_H2O
    data_output['Absorbance_Hat'] = signal.medfilt(data_H2O['Absorbance'], 5)
    data_output['BL_NIR_H2O'] = Auto_B.als_baseline(data_output['Absorbance_Hat'], asymmetry_param=0.001, smoothness_param=1e9, max_iters=10, conv_thresh=1e-5)
    data_output['Subtracted_Peak'] = data_output['Absorbance_Hat'] - data_output['BL_NIR_H2O']
    
    krige_wn_range = np.linspace(H2O_wn_low-5, H2O_wn_high+5, H2O_wn_high-H2O_wn_low+11)
    krige_peak = OrdinaryKriging(data_H2O.index, np.zeros(data_output['Subtracted_Peak'].shape), data_output['Subtracted_Peak'], variogram_model = 'gaussian')
    krige_abs, krige_std = krige_peak.execute("grid", krige_wn_range, np.array([0.0]))
    krige_output = pd.DataFrame(columns = ['Absorbance', 'STD'], index = krige_wn_range)
    krige_output['Absorbance'] = np.asarray(np.squeeze(krige_abs))
    krige_output['STD'] = np.asarray(np.squeeze(krige_std))

    if peak == 'OH': # 4500 peak
        PR_4500_low = 4400
        PR_4500_high = 4600
        PH_max = np.max(data_output['Subtracted_Peak'][PR_4500_low:PR_4500_high])
        PH = np.max(data_output['Subtracted_Peak'][PR_4500_low:PR_4500_high])
        PH_krige = np.max(krige_output['Absorbance'][PR_4500_low:PR_4500_high]) - np.min(krige_output['Absorbance'])
        PH_krige_index = int(data_output['Subtracted_Peak'][data_output['Subtracted_Peak'] == PH_max].index.to_numpy())
        PH_std = np.std(data_output['Subtracted_Peak'][PH_krige_index-50:PH_krige_index+50])
        STN = PH_krige / PH_std
    elif peak == 'H2O': # 5200 peak
        PR_5200_low = 5100
        PR_5200_high = 5300
        PH_max = np.max(data_output['Subtracted_Peak'][PR_5200_low:PR_5200_high])
        PH = np.max(data_output['Subtracted_Peak'][PR_5200_low:PR_5200_high])
        PH_krige = np.max(krige_output['Absorbance'][PR_5200_low:PR_5200_high]) - np.min(krige_output['Absorbance'])
        PH_krige_index = int(data_output['Subtracted_Peak'][data_output['Subtracted_Peak'] == PH_max].index.to_numpy())
        PH_std = np.std(data_output['Subtracted_Peak'][PH_krige_index-50:PH_krige_index+50])
        STN = PH_krige / PH_std

    return data_output, krige_output, PH_krige, STN


def MidIR_Process(data, H2O_wn_low, H2O_wn_high):

    """The MidIR_Process function with the inputs of dictionary data and the Mid-IR total H2O wavenumbers of interest and returns 
    a dataframe of the absorbance data in the region of interest, the kriged data output, and the peak height. 
    This function is used three times with slightly different H2O wavenumber ranges, so that uncertainty in peak height can be assessed."""

    data_H2O3550 = data[H2O_wn_low:H2O_wn_high]
    data_output = pd.DataFrame(columns = ['Absorbance', 'BL_MIR_3550', 'Subtracted_Peak', 'Subtracted_Peak_Hat'], index = data_H2O3550.index)
    data_output['Absorbance'] = data_H2O3550['Absorbance']
    data_output['BL_MIR_3550'] = Auto_B.als_baseline(data_H2O3550['Absorbance'], asymmetry_param=0.0010, smoothness_param=1e11, max_iters=20, conv_thresh=1e-7)
    data_output['Subtracted_Peak'] = data_H2O3550['Absorbance'] - data_output['BL_MIR_3550']
    data_output['Subtracted_Peak_Hat'] = signal.medfilt(data_output['Subtracted_Peak'], 21)

    peak_wn_low = 3300
    peak_wn_high = 3600
    plot_output = data_output[peak_wn_low:peak_wn_high]
    plotindex = np.argmax(plot_output['Absorbance'].index.to_numpy() > 3400)
    PH_3550 = np.max(plot_output['Subtracted_Peak_Hat'])

    return data_output, plot_output, PH_3550, plotindex



def MCMC(data, uncert, indparams, log, savefile):
    
    """The MCMC function takes the required inputs and runs the Monte Carlo-Markov Chain. The function ouputs the 
    mc3_output which contains all of the best fit parameters and standard deviations."""

    func = Carbonate
    params = np.array([1.0, 0.5, 0.0001, 0.0001, 0.00001, 1430.0, 25, 0.01, 1515, 25, 0.01, 0.5, 0.10, 0.01, 5e-4, 0.65])
    pstep = 0.01 * params
    pmin = np.array([0.00, -5.0, -1.0, -1.0, -0.5, 1420.0, 22.5, 0.001, 1505.0, 22.5, 0.001, 0.0, -1.0, -1.0, -5e-2, -10.0])
    pmax = np.array([10.0,  5.0,  1.0,  1.0,  0.5, 1440.0, 35.0, 0.600, 1525.0, 35.0, 0.600, 5.0,  1.0,  1.0,  5e-2,  10.0])
    priorlow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 0.0, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorup  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 0.0, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pnames   = ['Avg_BL',"PCA1","PCA2","PCA3","PCA4",'peak_G1430','std_G1430','G1430_amplitude','peak_G1515','std_G1515','G1515_amplitude','Average_1635Peak','1635PeakPCA1','1635PeakPCA2','m','b']
    texnames = ['$\overline{BL}$',"$PCA_1$","$PCA_2$","$PCA_3$",'$PCA_4$','$P_{1430}$','$S_{1430}$','$A_{1430}$','$P_{1515}$','$S_{1515}$','$A_{1515}$','$\overline{H_{1635}}$','${H_{1635,PCA_1}}$','${H_{1635,PCA_2}}$','m','b']

    mc3_output = mc3.sample(data, uncert, func=func, params=params, indparams=indparams, 
        pstep=pstep, pmin=pmin, pmax=pmax, priorlow=priorlow, priorup=priorup, 
        pnames=pnames, texnames=texnames, sampler='snooker', rms=False,
        nsamples=2.5e5, nchains=9, ncpu=3, burnin=5000, thinning=1,
        leastsq='trf', chisqscale=False, grtest=True, grbreak=1.01, grnmin=0.5,
        hsize=10, kickoff='normal', wlike=False, plots=False, log=log, savefile=savefile)

    return mc3_output

# %% 


def Run_All_Spectra(dfs_dict, paths):

    """The Run_All_Spectra function inputs the dictionary of dataframes that were created by the Load_SampleCSV function and allows 
    for all of the samples to be batched and run through the function. The function exports the best fit and standard deviations 
    of peak locations, peak widths, and peak heights, as well as the PCA vectors used to fit the spectra. These values are 
    exported in a csv file and figures are created for each individual sample."""

    PCAmatrix = Load_PCA(paths[0])
    Peak_1635_PCAmatrix = Load_PCA(paths[1])
    Wavenumber = Load_Wavenumber(paths[0])
    exportpath = paths[-1]
    Nvectors = 5
    indparams = [Wavenumber, PCAmatrix, Peak_1635_PCAmatrix, Nvectors]

    output_dir = ["./FIGURES/", "./PLOTFILES/", "./NPZFILES/", "./LOGFILES/"] 
    for ii in range(len(output_dir)):
        if not os.path.exists(output_dir[ii]+exportpath):
            os.makedirs(output_dir[ii]+exportpath)

    plotpath = 'PLOTFILES/' + exportpath + '/'
    logpath = 'LOGFILES/' + exportpath + '/'
    savefilepath = 'NPZFILES/' + exportpath + '/'
    figurepath = 'FIGURES/' + exportpath + '/'

    additional_dir = ["TRACE", "HISTOGRAM", "PAIRWISE", "MODELFIT"]
    for ii in range(len(additional_dir)): 
        if not os.path.exists(plotpath+additional_dir[ii]): 
            os.makedirs(plotpath+additional_dir[ii])

    # P_ = peak_, _M = _mean, _BP = best parameter, #STD = _stdev
    DF_Output = pd.DataFrame(columns = ['P_H1635_BP','P_H1635_STD','H2Om_1635_MAX','P_H1635_PCA1_BP','P_H1635_PCA1_STD','P_H1635_PCA2_BP','P_H1635_PCA2_STD',
    'P_G1515_BP','P_G1515_STD','STD_G1515_BP','STD_G1515_STD','AMP_G1515_BP','AMP_G1515_STD','P_G1430_BP','P_G1430_STD','STD_G1430_BP','STD_G1430_STD','AMP_G1430_BP','AMP_G1430_STD', 
    'AVG_BL_BP','AVG_BL_STD','PCA1_BP','PCA1_STD','PCA2_BP','PCA2_STD','PCA3_BP','PCA3_STD','PCA4_BP','PCA4_STD','m_BP','m_STD','b_BP','b_STD'])
    NEAR_IR_PH = pd.DataFrame(columns=['P_H5200_M', 'P_H4500_M', 'P_H5200_STD', 'P_H4500_STD', 'S2N_P5200', 'S2N_P4500', 'ERR_5200', 'ERR_4500'])
    H2O_3550_PH = pd.DataFrame(columns=['PH_3550_M', 'PH_3550_STD'])
    H2O_3550_MAX = pd.DataFrame(columns=['H2OT_3550_MAX', 'H2OT_3550_SAT?'])
    CO2_PH_M = pd.DataFrame(columns = ['PH1635_M', 'PH1635_STD', 'PH1515_M', 'PH1515_STD', 'PH1430_M', 'PH1430_STD'])
    CO2_PH_BP = pd.DataFrame(columns = ['PH1635_BP', 'PH1635_BP_STD', 'PH1515_BP', 'PH1515_BP_STD', 'PH1430_BP', 'PH1430_BP_STD'])

    # Run the MCMC:
    failures = []
    error = []
    error_4500 = []
    error_5200 = []

    try: 
        for files, data in dfs_dict.items(): 
            
            H2O4500_wn_low_1 = 4250
            H2O4500_wn_high_1 = 4675
            data_H2O4500_1, krige_output_4500_1, PH_4500_krige_1, STN_4500_1 = NearIR_Process(data, H2O4500_wn_low_1, H2O4500_wn_high_1, 'OH')
            H2O4500_wn_low_2 = 4225
            H2O4500_wn_high_2 = 4650
            data_H2O4500_2, krige_output_4500_2, PH_4500_krige_2, STN_4500_2 = NearIR_Process(data, H2O4500_wn_low_2, H2O4500_wn_high_2, 'OH')
            H2O4500_wn_low_3 = 4275
            H2O4500_wn_high_3 = 4700
            data_H2O4500_3, krige_output_4500_3, PH_4500_krige_3, STN_4500_3 = NearIR_Process(data, H2O4500_wn_low_3, H2O4500_wn_high_3, 'OH')

            # Three repeat baselines for the H2Om_{5200}
            H2O5200_wn_low_1 = 4875
            H2O5200_wn_high_1 = 5400
            data_H2O5200_1, krige_output_5200_1, PH_5200_krige_1, STN_5200_1 = NearIR_Process(data, H2O5200_wn_low_1, H2O5200_wn_high_1, 'H2O')
            H2O5200_wn_low_2 = 4850
            H2O5200_wn_high_2 = 5375
            data_H2O5200_2, krige_output_5200_2, PH_5200_krige_2, STN_5200_2 = NearIR_Process(data, H2O5200_wn_low_2, H2O5200_wn_high_2, 'H2O')
            H2O5200_wn_low_3 = 4900
            H2O5200_wn_high_3 = 5425
            data_H2O5200_3, krige_output_5200_3, PH_5200_krige_3, STN_5200_3 = NearIR_Process(data, H2O5200_wn_low_3, H2O5200_wn_high_3, 'H2O')

            PH_4500_krige = np.array([PH_4500_krige_1, PH_4500_krige_2, PH_4500_krige_3])
            PH_4500_krige_M = np.mean(PH_4500_krige)
            PH_4500_krige_STD = np.std(PH_4500_krige)
            PH_5200_krige = np.array([PH_5200_krige_1, PH_5200_krige_2, PH_5200_krige_1])
            PH_5200_krige_M = np.mean(PH_5200_krige)
            PH_5200_krige_STD = np.std(PH_5200_krige)

            STN_4500 = np.array([STN_4500_1, STN_4500_2, STN_4500_3])
            STN_4500_M = np.average(STN_4500)
            STN_5200 = np.array([STN_5200_1, STN_5200_2, STN_5200_3])
            STN_5200_M = np.average(STN_5200)

            if STN_4500_M >= 4.0: 
                error_4500 = '-'
            elif STN_4500_M < 4.0: 
                error_4500 = '*'

            if STN_5200_M >= 4.0: 
                error_5200 = '-'
            elif STN_5200_M < 4.0: 
                error_5200 = '*'

            plotmin = np.round(np.min(data[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']), decimals = 1)
            plotmax = np.round(np.max(data[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']), decimals = 1)
            fig, ax = plt.subplots(figsize = (26, 8))
            ax1 = plt.subplot2grid((2, 3), (0, 0))
            ax1.plot(data.index, data['Absorbance'], 'k', linewidth = 1.5)
            ax1.plot(data_H2O5200_1.index, data_H2O5200_1['Absorbance_Hat'], data_H2O5200_2.index, data_H2O5200_2['Absorbance_Hat'], data_H2O5200_3.index, data_H2O5200_3['Absorbance_Hat'])
            ax1.plot(data_H2O4500_1.index, data_H2O4500_1['Absorbance_Hat'], data_H2O4500_2.index, data_H2O4500_2['Absorbance_Hat'], data_H2O4500_3.index, data_H2O4500_3['Absorbance_Hat'])
            ax1.plot(data_H2O5200_1.index, data_H2O5200_1['BL_NIR_H2O'], data_H2O5200_2.index, data_H2O5200_2['BL_NIR_H2O'], data_H2O5200_3.index, data_H2O5200_3['BL_NIR_H2O'])
            ax1.plot(data_H2O4500_1.index, data_H2O4500_1['BL_NIR_H2O'], data_H2O4500_2.index, data_H2O4500_2['BL_NIR_H2O'], data_H2O4500_3.index, data_H2O4500_3['BL_NIR_H2O'])
            ax1.annotate(f"5200 cm $^{{- 1}}$  Peak Height Gaussian: {PH_5200_krige_M:.4f} ± {PH_5200_krige_STD:.4f} $\ cm ^{{- 1}} $, S2N = {STN_5200_M:.2f}", (0.025, 0.90), xycoords = 'axes fraction')
            ax1.annotate(f"4500 cm $^{{- 1}}$  Peak Height Gaussian: {PH_4500_krige_M:.4f} ± {PH_4500_krige_STD:.4f} $\ cm ^{{- 1}} $, S2N = {STN_4500_M:.2f}", (0.025, 0.80), xycoords = 'axes fraction')
            ax1.set_ylabel('Absorbance')
            ax1.legend(['Near IR Data','_','_','Median Filtered 5200','_','_','Median Filtered 4500','_','_','Baseline 5200','_','_','Baseline 4500'], prop={'size': 12})
            warnings.filterwarnings("ignore", category = UserWarning)
            ax1.set_xlim([4200, 5400])
            ax1.set_ylim([plotmin-0.075, plotmax+0.075])
            ax1.invert_xaxis()

            plotmax = np.round(np.max(data_H2O4500_1['Subtracted_Peak']), decimals = 1)
            ax2 = plt.subplot2grid((2, 3), (1, 0))
            ax2.plot(data_H2O5200_1.index, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1['Absorbance']), 'k', data_H2O5200_2.index, data_H2O5200_2['Subtracted_Peak'] - np.min(krige_output_5200_2['Absorbance']), 'k', 
                data_H2O5200_3.index, data_H2O5200_3['Subtracted_Peak'] - np.min(krige_output_5200_3['Absorbance']), 'k', label = 'Subtracted Baseline 5200')
            ax2.plot(data_H2O4500_1.index, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1['Absorbance']), 'k', data_H2O4500_2.index, data_H2O4500_2['Subtracted_Peak'] - np.min(krige_output_4500_2['Absorbance']), 'k', 
                data_H2O4500_3.index, data_H2O4500_3['Subtracted_Peak'] - np.min(krige_output_4500_3['Absorbance']), 'k', label = 'Subtracted Baseline 4500')
            ax2.plot(krige_output_5200_1.index, krige_output_5200_1['Absorbance'] - np.min(krige_output_5200_1['Absorbance']), krige_output_5200_2.index, krige_output_5200_2['Absorbance'] - np.min(krige_output_5200_2['Absorbance']),
                krige_output_5200_3.index, krige_output_5200_3['Absorbance'] - np.min(krige_output_5200_3['Absorbance']), label = 'Gaussian 5200')
            ax2.plot(krige_output_4500_1.index, krige_output_4500_1['Absorbance'] - np.min(krige_output_4500_1['Absorbance']), krige_output_4500_2.index, krige_output_4500_2['Absorbance'] - np.min(krige_output_4500_2['Absorbance']), 
                krige_output_4500_3.index, krige_output_4500_3['Absorbance'] - np.min(krige_output_4500_3['Absorbance']), label = 'Gaussian 4500')
            ax2.set_xlabel('Wavenumber $cm^{-1}$')    
            ax2.set_ylabel('Absorbance')
            ax2.legend(['Subtracted 5200','_','_','Subtracted 4500','_','_','_','_','Gaussian 5200','_','_','Gaussian 4500'])
            warnings.filterwarnings("ignore", category = UserWarning)
            ax2.set_xlim([4200, 5400])
            ax2.set_ylim([0, plotmax+0.05])
            ax2.invert_xaxis()

            NEAR_IR_PH.loc[files] = pd.Series({'P_H5200_M': PH_5200_krige_M, 'P_H4500_M': PH_4500_krige_M, 
            'P_H5200_STD': PH_5200_krige_STD, 'P_H4500_STD': PH_4500_krige_STD, 
            'S2N_P5200': STN_5200_M, 'S2N_P4500': STN_4500_M, 'ERR_5200': error_5200, 'ERR_4500': error_4500})

            H2O3550_wn_low_1  = 1900
            H2O3550_wn_high_1 = 4400
            data_H2O3550_1, plot_output_3550_1, PH_3550_1, plotindex1 = MidIR_Process(data, H2O3550_wn_low_1, H2O3550_wn_high_1)
            H2O3550_wn_low_2  = 2100
            H2O3550_wn_high_2 = 4200
            data_H2O3550_2, plot_output_3550_2, PH_3550_2, plotindex2 = MidIR_Process(data, H2O3550_wn_low_2, H2O3550_wn_high_2)
            H2O3550_wn_low_3  = 2300
            H2O3550_wn_high_3 = 4000
            data_H2O3550_3, plot_output_3550_3, PH_3550_3, plotindex3 = MidIR_Process(data, H2O3550_wn_low_3, H2O3550_wn_high_3)
            PH_3550_M = np.mean([PH_3550_1, PH_3550_2, PH_3550_3])
            PH_3550_STD = np.std([PH_3550_1, PH_3550_2, PH_3550_3])

            MAX_3550_ABS = data_H2O3550_1['Absorbance'].to_numpy()[np.argmax(data_H2O3550_1['Absorbance'].index.to_numpy() > 3550)]
            MAX_1635_ABS = data_H2O3550_1['Absorbance'].to_numpy()[np.argmax((data_H2O3550_1['Absorbance'].index.to_numpy() > 1600) & (data_H2O3550_1['Absorbance'].index.to_numpy() < 1800))]

            if MAX_3550_ABS < 2: 
                error = '-'
            elif MAX_3550_ABS >= 2: 
                error = '*'
            
            plotmax = np.round(np.max(data_H2O3550_1['Absorbance'].to_numpy()), decimals = 0)
            plotmin = np.round(np.min(data_H2O3550_1['Absorbance'].to_numpy()), decimals = 0)
            ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan = 2)
            ax3.plot(data.index, data['Absorbance'], 'k')
            ax3.plot(data_H2O3550_1['Absorbance'].index, data_H2O3550_1['BL_MIR_3550'], data_H2O3550_2['Absorbance'].index, 
                data_H2O3550_2['BL_MIR_3550'], data_H2O3550_3['Absorbance'].index, data_H2O3550_3['BL_MIR_3550'])
            ax3.plot(plot_output_3550_1.index, (plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.plot(plot_output_3550_2.index, (plot_output_3550_2['Subtracted_Peak_Hat']+plot_output_3550_2['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.plot(plot_output_3550_3.index, (plot_output_3550_3['Subtracted_Peak_Hat']+plot_output_3550_3['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.set_title(files)
            ax3.annotate(f"3550 cm $^{{- 1}}$  Peak Height: {PH_3550_M:.4f} ± {PH_3550_STD:.4f} $\ cm ^{{- 1}}$", (0.025, 0.95), xycoords = 'axes fraction')
            ax3.set_xlabel('Wavenumber $cm^{-1}$')
            ax3.set_xlim([1275, 4000])
            ax3.set_ylabel('Absorbance')
            ax3.set_ylim([plotmin-0.25, plotmax+0.5])
            ax3.invert_xaxis()

            H2O_3550_PH.loc[files] = pd.Series({'PH_3550_M': PH_3550_M, 'PH_3550_STD': PH_3550_STD})
            H2O_3550_MAX.loc[files] = pd.Series({'H2OT_3550_MAX': MAX_3550_ABS, 'H2OT_3550_SAT?': error})

            df_length = np.shape(Wavenumber)[0]
            
            CO2_wn_high = 2200
            CO2_wn_low  = 1275 

            spec = data[CO2_wn_low:CO2_wn_high]

            if spec.shape[0] != df_length:              
                interp_wn = np.linspace(spec.index[0], spec.index[-1], df_length)
                interp_abs = interpolate.interp1d(spec.index, spec['Absorbance'])(interp_wn)
                spec = spec.reindex(index = interp_wn)
                spec['Absorbance'] = interp_abs
                spec = spec['Absorbance'].to_numpy()
            elif spec.shape[0] == df_length: 
                spec = spec['Absorbance'].to_numpy()

            uncert = np.ones_like(spec) * 0.01
            mc3_output = MCMC(data = spec, uncert = uncert, indparams = indparams, log = logpath+files+'.log', savefile=savefilepath+files+'.npz')

            PCA_BP = mc3_output['bestp'][0:Nvectors]
            CO2P_BP = mc3_output['bestp'][-11:-5]
            H2OmP1635_BP = mc3_output['bestp'][-5:-2]
            m_BP, b_BP = mc3_output['bestp'][-2:None]
            Baseline_Solve_BP = PCA_BP * PCAmatrix.T
            Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

            PCA_STD = mc3_output['stdp'][0:Nvectors]
            CO2P_STD = mc3_output['stdp'][-11:-5]
            H2OmP1635_STD = mc3_output['stdp'][-5:-2]
            H2OmP1635_STD[0] = H2OmP1635_STD[0]
            m_STD, b_STD = mc3_output['stdp'][-2:None]

            Line_BP = Linear(Wavenumber, m_BP, b_BP) 
            Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

            H1635_BP = H2OmP1635_BP * Peak_1635_PCAmatrix.T
            H1635_BP = np.asarray(H1635_BP).ravel()

            CO2P1430_BP = Gauss(Wavenumber, CO2P_BP[0], CO2P_BP[1], A=CO2P_BP[2])
            CO2P1515_BP = Gauss(Wavenumber, CO2P_BP[3], CO2P_BP[4], A=CO2P_BP[5])

            posteriorerror = np.load(savefilepath+files+'.npz')
            samplingerror = posteriorerror['posterior'][:, 0:5]
            samplingerror = samplingerror[0:np.shape(posteriorerror['posterior'][:, :])[0]:2000, :]
            lineerror = posteriorerror['posterior'][:, -2:None]
            lineerror = lineerror[0:np.shape(posteriorerror['posterior'][:, :])[0]:2000, :]
            Baseline_Array = np.array(samplingerror * PCAmatrix[:, :].T)
            Baseline_Array_Plot = Baseline_Array

            ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan = 2)
            for i in range(np.shape(Baseline_Array)[0]):
                linearray = Linear(Wavenumber, lineerror[i, 0], lineerror[i, 1])
                Baseline_Array_Plot[i, :] = Baseline_Array[i, :] + linearray
                plt.plot(Wavenumber, Baseline_Array_Plot[i, :], 'dimgray', linewidth = 0.25)
            ax4.plot(Wavenumber, spec, 'tab:blue', linewidth = 2.5, label = 'FTIR Spectrum')
            ax4.plot(Wavenumber, H1635_BP + Baseline_Solve_BP, 'tab:orange', linewidth = 1.5, label = '1635')
            ax4.plot(Wavenumber, CO2P1515_BP + Baseline_Solve_BP, 'tab:green', linewidth = 2.5, label = '1515')
            ax4.plot(Wavenumber, CO2P1430_BP + Baseline_Solve_BP, 'tab:red', linewidth = 2.5, label = '1430')
            ax4.plot(Wavenumber, Carbonate(mc3_output['meanp'], Wavenumber, PCAmatrix, Peak_1635_PCAmatrix, Nvectors), 'tab:purple', linewidth = 1.5, label = 'MC3 Fit')
            ax4.plot(Wavenumber, Baseline_Solve_BP, 'k', linewidth = 1.5, label = 'Baseline')
            ax4.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {H2OmP1635_BP[0]:.3f} ± {H2OmP1635_STD[0]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.95), xycoords = 'axes fraction')
            ax4.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {CO2P_BP[5]:.3f} ± {CO2P_STD[5]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.90), xycoords = 'axes fraction')
            ax4.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {CO2P_BP[2]:.3f} ± {CO2P_STD[2]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.85), xycoords = 'axes fraction')
            ax4.set_title(files)
            ax4.set_xlim([1275, 2000])
            ax4.set_xlabel('Wavenumber $cm^{-1}$')
            ax4.set_ylabel('Absorbance')
            ax4.legend(loc = 'upper right', prop={'size': 12})
            ax4.invert_xaxis()
            plt.tight_layout()
            plt.savefig(figurepath + files + '.pdf', backend='pgf')
            plt.close('all')

            texnames = ['$\overline{BL}$',"$PCA_1$","$PCA_2$","$PCA_3$",'$PCA_4$','$P_{1430}$','$S_{1430}$','$A_{1430}$','$P_{1515}$','$S_{1515}$','$A_{1515}$','$\overline{H_{1635}}$','${H_{1635,PCA_1}}$','${H_{1635,PCA_2}}$','m','b']

            fig1 = mc3plots.trace(mc3_output['posterior'], title = files, zchain=mc3_output['zchain'], burnin=mc3_output['burnin'], pnames=texnames, savefile = plotpath+'/TRACE/'+files+'_trace.pdf')
            plt.close('all')
            fig2 = mc3plots.histogram(mc3_output['posterior'], title = files, pnames=texnames, bestp=mc3_output['bestp'], savefile = plotpath+'/HISTOGRAM/'+files+'_histogram.pdf', quantile=0.683)
            plt.close('all')
            fig3 = mc3plots.pairwise(mc3_output['posterior'], title = files, pnames=texnames, bestp=mc3_output['bestp'], savefile = plotpath+'/PAIRWISE/'+files+'_pairwise.pdf')
            plt.close('all')
            fig4 = mc3plots.modelfit(spec, uncert, indparams[0], mc3_output['best_model'], title = files, savefile=plotpath+'/MODELFIT/'+files+'_modelfit.pdf')
            plt.close('all')
            
            # Create dataframe of best fit parameters and their standard deviations
            DF_Output.loc[files] = pd.Series({'P_H1635_BP':H2OmP1635_BP[0],'P_H1635_STD':H2OmP1635_STD[0],'H2Om_1635_MAX':MAX_1635_ABS,
            'P_H1635_PCA1_BP':H2OmP1635_BP[1],'P_H1635_PCA1_STD':H2OmP1635_STD[1],'P_H1635_PCA2_BP':H2OmP1635_BP[2],'P_H1635_PCA2_STD':H2OmP1635_STD[2], 
            'P_G1515_BP':CO2P_BP[3],'P_G1515_STD':CO2P_STD[3],'STD_G1515_BP':CO2P_BP[4],'STD_G1515_STD':CO2P_STD[4],'AMP_G1515_BP':CO2P_BP[5],'AMP_G1515_STD':CO2P_STD[5],
            'P_G1430_BP':CO2P_BP[0],'P_G1430_STD':CO2P_STD[0],'STD_G1430_BP':CO2P_BP[1],'STD_G1430_STD':CO2P_STD[1],'AMP_G1430_BP':CO2P_BP[2],'AMP_G1430_STD':CO2P_STD[2],
            'AVG_BL_BP':PCA_BP[0],'AVG_BL_STD':PCA_STD[0],'PCA1_BP':PCA_BP[1],'PCA1_STD':PCA_STD[1],'PCA2_BP':PCA_BP[2],'PCA2_STD':PCA_STD[2], 
            'PCA3_BP':PCA_BP[3],'PCA3_STD':PCA_STD[3],'PCA4_BP':PCA_BP[4],'PCA4_STD':PCA_STD[4],'m_BP':m_BP,'m_STD':m_STD,'b_BP':b_BP,'b_STD':b_STD})

            CO2_PH_BP.loc[files] = pd.Series({'PH1635_BP': H2OmP1635_BP[0], 'PH1635_BP_STD': H2OmP1635_STD[0],
            'PH1515_BP': CO2P_BP[5], 'PH1515_BP_STD': CO2P_STD[5], 
            'PH1430_BP': CO2P_BP[2], 'PH1430_BP_STD': CO2P_STD[2]})
            
    except:
        failures.append(files)
        print(files + ' failed.')

    DF_Output = pd.concat([H2O_3550_PH, H2O_3550_MAX, DF_Output], axis = 1)
    Volatiles_PH = pd.concat([H2O_3550_PH, CO2_PH_BP], axis = 1)

    return DF_Output, Volatiles_PH, NEAR_IR_PH, failures

# %% 

def Beer_Lambert(molar_mass, absorbance, sigma_absorbance, density, sigma_density, thickness, sigma_thickness, epsilon, sigma_epsilon):

    """The Beer_Lambert function applies the Beer-Lambert Law with the inputs of molar mass, absorbance, density, thickness, and epsilon (absorbance coefficient), 
    as well as the uncertainty associated with each term aside from molar mass and returns the concentration with a multiplier of concentration uncertainty,
    calculated by standard error propagation techniques."""

    # https://sites.fas.harvard.edu/~scphys/nsta/error_propagation.pdf

    # concentration_std = pd.DataFrame(columns = ['STD'])

    concentration = pd.Series()
    concentration_std = pd.Series()
    concentration = (1e6 * molar_mass * absorbance) / (density * thickness * epsilon)

    # sigma_concentration_mult = pd.Series()
    # sigma_concentration_mult = np.sqrt((sigma_absorbance/absorbance)**2 + (sigma_density/density)**2 + (sigma_thickness/thickness)**2 + (sigma_epsilon/epsilon)**2)

    return concentration


def Beer_Lambert_Error(N, molar_mass, absorbance, sigma_absorbance, density, sigma_density, thickness, sigma_thickness, epsilon, sigma_epsilon): 

    """The Beer_Lambert_Error function applies a simple Monte Carlo with the same inputs as the Beer_Lambert function, with the addition 
    of N (number of samples), and outputs the uncertainty in concentration. Absorbance, density, and thickness is treated with a Gaussian distribution, 
    and absorbance coefficient is treated with a uniform distribution."""

    # https://astrofrog.github.io/py4sci/_static/Practice%20Problem%20-%20Monte-Carlo%20Error%20Propagation%20-%20Sample%20Solution.html
    gaussian_concentration = (1e6 * molar_mass * np.random.normal(absorbance, sigma_absorbance, N) / (np.random.normal(density, sigma_density, N) 
        * np.random.normal(thickness, sigma_thickness, N) * np.random.uniform(epsilon-sigma_epsilon, epsilon+sigma_epsilon, N)))
    concentration_std = np.std(gaussian_concentration)

    return concentration_std


def Density_Calculation(MI_Composition):

    """The Density_Calculation function inputs the MI composition file and outputs the glass density at room temperature and pressure. 
    The mole fraction is calculated. The total molar volume xivibari is determined from sum of the mole fractions of each oxide * partial molar volume 
    at room temperature and pressure of analysis. The gram formula weight gfw is then determined by summing the mole fractions*molar masses. 
    The density is finally determined by dividing gram formula weight by total molar volume."""

    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
        'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

    T_room = 25 # during analysis (deg. C)
    P_room = 1  # during analysis (bars)

    # Partial Molar Volumes from Lesher and Spera, 2015
    par_molar_vol = {'SiO2': (26.86-1.89*P_room/1000), 'TiO2': (23.16+7.24*(T_room+273-1673)/1000-2.31*P_room/1000), 'Al2O3': (37.42-2.26*P_room/1000), 
        'Fe2O3': (42.13+9.09*(T_room+273-1673)/1000-2.53*P_room/1000), 'FeO': (13.65+2.92*(T_room+273-1673)/1000-0.45*P_room/1000),
        'MgO': (11.69+3.27*(T_room+273-1673)/1000+0.27*P_room/1000), 'CaO': (16.53+3.74*(T_room+273-1673)/1000+0.34*P_room/1000), 
        'Na2O': (28.88+7.68*(T_room+273-1673)/1000-2.4*P_room/1000), 'K2O': (45.07+12.08*(T_room+273-1673)/1000-6.75*P_room/1000), 
        'H2O': (26.27+9.46*(T_room+273-1673)/1000-3.15*P_room/1000)}

    mol = pd.DataFrame()
    for oxide in MI_Composition:
        mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]

    mol_tot = pd.DataFrame()
    mol_tot = mol.sum(axis = 1)

    oxide_density = pd.DataFrame()
    xivbari = pd.DataFrame()
    gfw = pd.DataFrame() # gram formula weight
    for oxide in MI_Composition:
        if oxide in par_molar_vol:
            xivbari[oxide] = mol[oxide]/mol_tot*par_molar_vol[oxide]
            gfw[oxide] = mol[oxide]/mol_tot*molar_mass[oxide]

    xivbari_tot = xivbari.sum(axis = 1)
    gfw_tot = gfw.sum(axis = 1)
    
    density = 1000 * gfw_tot / xivbari_tot

    return mol, density


def Concentration_Output(Volatiles_PH, N, thickness, MI_Composition):

    """The Concentration_Output function inputs a dictionary with the peak heights for the total H2O peak (3550 cm^-1), molecular H2O peak (1635 cm^-1), 
    and carbonate peaks (1515 and 1430 cm^-1), number of samples for the Monte Carlo, thickness information, and MI composition, and 
    outputs the concentrations and uncertainties for each peak. Both the best fit parameter and mean from the MC3 code are used to calculate 
    concentration."""

    mega_spreadsheet = pd.DataFrame(columns = ['H2OT_3550_M', 'H2OT_3550_STD', 'H2Om_1635_BP', 'H2Om_1635_STD', 
        'CO2_1515_BP', 'CO2_1515_STD', 'CO2_1430_BP', 'CO2_1430_STD'])
    epsilon = pd.DataFrame(columns=['Tau', 'Na/Na+Ca', 'epsilon_H2OT_3550', 'sigma_epsilon_H2OT_3550', 'epsilon_H2Om_1635', 'sigma_epsilon_H2Om_1635', 
        'epsilon_CO2', 'sigma_epsilon_CO2'])
    density_df = pd.DataFrame(columns=['Density'])
    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
        'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

    MI_Composition['H2O'] = 0
    mol, density = Density_Calculation(MI_Composition)

    cation_tot = mol.sum(axis = 1) + mol['Al2O3'] + mol['Na2O'] + mol['K2O'] + mol['P2O5']
    Na_NaCa = (2*mol['Na2O']) / ((2*mol['Na2O']) + mol['CaO'])
    SiAl_tot = (mol['SiO2'] + (2*mol['Al2O3'])) / cation_tot

    mest_3550 = np.array([15.725557, 71.368691])
    mest_1635 = np.array([-50.397564, 124.250534])
    mest_CO2 = np.array([440.6964, -355.2053])

    covm_est_3550 = np.diag([38.4640, 77.8597])
    covm_est_1635 = np.diag([20.8503, 39.3875])
    covm_est_CO2 = np.diag([103.7645, 379.9891])
    sigma_thickness = 3

    G_SiAl = np.ones((2, 1))
    G_NaCa = np.ones((2, 1))
    covz_error_SiAl = np.zeros((2, 2))
    covz_error_NaCa = np.zeros((2, 2))

    for i in MI_Composition.index:
        epsilon_H2OT_3550 = mest_3550[0]+(mest_3550[1]*SiAl_tot[i])
        epsilon_H2Om_1635 = mest_1635[0]+(mest_1635[1]*SiAl_tot[i])
        epsilon_CO2 = mest_CO2[0]+(mest_CO2[1]*Na_NaCa[i])

        G_SiAl[1, 0] = SiAl_tot[i]
        G_NaCa[1, 0] = Na_NaCa[i]
        covz_error_SiAl[1, 1] = SiAl_tot[i] * 0.01 # 1 sigma
        covz_error_NaCa[1, 1] = Na_NaCa[i] * 0.01

        CT_int_3550 = (G_SiAl*covm_est_3550*np.transpose(G_SiAl)) + (mest_3550*covz_error_SiAl*np.transpose(mest_3550))
        CT68_3550 = np.sqrt(np.mean(np.diag(CT_int_3550)))

        CT_int_1635 = (G_SiAl*covm_est_1635*np.transpose(G_SiAl)) + (mest_1635*covz_error_SiAl*np.transpose(mest_1635))
        CT68_1635 = np.sqrt(np.mean(np.diag(CT_int_1635)))

        CT_int_CO2 = (G_NaCa*covm_est_CO2*np.transpose(G_NaCa)) + (mest_CO2*covz_error_NaCa*np.transpose(mest_CO2))
        CT68_CO2 = np.sqrt(np.mean(np.diag(CT_int_CO2)))

        epsilon.loc[i] = pd.Series({'Tau': SiAl_tot[i], 'Na/Na+Ca': Na_NaCa[i], 'epsilon_H2OT_3550': epsilon_H2OT_3550, 'sigma_epsilon_H2OT_3550': CT68_3550, 
        'epsilon_H2Om_1635': epsilon_H2Om_1635, 'sigma_epsilon_H2Om_1635': CT68_1635, 'epsilon_CO2': epsilon_CO2, 'sigma_epsilon_CO2': CT68_CO2})

    # Doing density-H2O iterations:
    for i in range(10):
        H2OT_3550_I = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['PH_3550_M'], Volatiles_PH['PH_3550_STD'],
            density, density * 0.025, thickness['Thickness'], sigma_thickness, epsilon['epsilon_H2OT_3550'], epsilon['sigma_epsilon_H2OT_3550'])
        MI_Composition['H2O'] = H2OT_3550_I
        mol, density = Density_Calculation(MI_Composition)

    for j in Volatiles_PH.index: 
        H2OT_3550_M = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['PH_3550_M'][j], Volatiles_PH['PH_3550_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_H2OT_3550'][j], epsilon['sigma_epsilon_H2OT_3550'][j])
        H2Om_1635_BP = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['PH1635_BP'][j], Volatiles_PH['PH1635_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_H2Om_1635'][j], epsilon['sigma_epsilon_H2Om_1635'][j])
        CO2_1515_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_PH['PH1515_BP'][j], Volatiles_PH['PH1515_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_CO2'][j], epsilon['sigma_epsilon_CO2'][j])
        CO2_1430_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_PH['PH1430_BP'][j], Volatiles_PH['PH1430_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_CO2'][j], epsilon['sigma_epsilon_CO2'][j])
        CO2_1515_BP *= 10000
        CO2_1430_BP *= 10000

        H2OT_3550_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['PH_3550_M'][j], Volatiles_PH['PH_3550_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_H2OT_3550'][j], epsilon['sigma_epsilon_H2OT_3550'][j])
        H2Om_1635_BP_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['PH1635_BP'][j], Volatiles_PH['PH1635_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_H2Om_1635'][j], epsilon['sigma_epsilon_H2Om_1635'][j])
        CO2_1515_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_PH['PH1515_BP'][j], Volatiles_PH['PH1515_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_CO2'][j], epsilon['sigma_epsilon_CO2'][j])
        CO2_1430_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_PH['PH1430_BP'][j], Volatiles_PH['PH1430_BP_STD'][j],
            density[j], density[j] * 0.025, thickness['Thickness'][j], sigma_thickness, epsilon['epsilon_CO2'][j], epsilon['sigma_epsilon_CO2'][j])
        CO2_1515_BP_STD *= 10000
        CO2_1430_BP_STD *= 10000

        mega_spreadsheet.loc[j] = pd.Series({'H2OT_3550_M': H2OT_3550_M, 'H2OT_3550_STD': H2OT_3550_M_STD, 'H2Om_1635_BP': H2Om_1635_BP, 'H2Om_1635_STD': H2Om_1635_BP_STD, 
            'CO2_1515_BP': CO2_1515_BP, 'CO2_1515_STD': CO2_1515_BP_STD, 'CO2_1430_BP': CO2_1430_BP, 'CO2_1430_STD': CO2_1430_BP_STD})
        density_df.loc[j] = pd.Series({'Density': density[j]})
    
    epsilon_f = pd.concat([density_df, epsilon], axis = 1)

    return epsilon_f, mega_spreadsheet

# %%


def H2O_Concentration_Output(Volatiles_PH, Volatiles_Concentrations, DF_Output, N, thickness, MI_Composition, inputdensity):

    """The H2O_Concentration_Output function inputs a dictionary with the peak heights for the molecular H2O (5200 cm^-1) and OH peak (4500 cm^-1), 
    number of samples for the Monte Carlo, thickness information, and MI composition, and outputs the concentrations and uncertainties for each peak."""

    H2O_SAT_DENSITY = pd.DataFrame(columns = ['H2OT_3550_SAT', 'Density', 'H2Om_1635_BP'])
    H2O_SAT_DENSITY['H2OT_3550_SAT'] = DF_Output['H2OT_3550_SAT?']
    H2O_SAT_DENSITY['Density'] = inputdensity
    H2O_SAT_DENSITY['H2Om_1635_BP'] = Volatiles_Concentrations['H2Om_1635_BP']

    mega_spreadsheet = pd.DataFrame(columns = ['H2OT_3550_SAT', 'H2Om_5200_M', 'H2Om_5200_M_STD', 'OH_4500_M', 'OH_4500_M_STD'])
    s2nerror = pd.DataFrame(columns = ['P_H5200_S2N', 'P_H4500_S2N', 'ERR_5200', 'ERR_4500'])
    epsilon = pd.DataFrame(columns=['Tau', 'epsilon_H2Om_5200', 'sigma_epsilon_H2Om_5200', 'epsilon_OH_4500', 'sigma_epsilon_OH_4500'])
    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 'MgO': 40.3044, 
        'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}
    density_df = pd.DataFrame(columns=['Density_F'])

    MI_Composition['H2O'] = 0
    mol, density = Density_Calculation(MI_Composition)

    cation_tot = mol.sum(axis = 1) + mol['Al2O3'] + mol['Na2O'] + mol['K2O'] + mol['P2O5']
    SiAl_tot = (mol['SiO2'] + (2*mol['Al2O3'])) / cation_tot

    mest_4500 = np.array([-1.632730, 3.532522])
    mest_5200 = np.array([-2.291420, 4.675528])
    covm_est_4500 = np.diag([0.0329, 0.0708])
    covm_est_5200 = np.diag([0.0129, 0.0276])
    G = np.ones((2, 1))
    covz_error = np.zeros((2, 2))
    sigma_thickness = 3

    for i in MI_Composition.index:
        epsilon_H2Om_5200 = mest_5200[0]+(mest_5200[1]*SiAl_tot[i])
        epsilon_OH_4500 = mest_4500[0]+(mest_4500[1]*SiAl_tot[i])
        G[1, 0] = SiAl_tot[i]
        covz_error[1, 1] = SiAl_tot[i] * 0.01 # 1 sigma
        CT_int_5200 = (G*covm_est_5200*np.transpose(G)) + (mest_5200*covz_error*np.transpose(mest_5200))
        CT_int_4500 = (G*covm_est_4500*np.transpose(G)) + (mest_4500*covz_error*np.transpose(mest_4500))
        CT68_5200 = np.sqrt(np.mean(np.diag(CT_int_5200)))
        CT68_4500 = np.sqrt(np.mean(np.diag(CT_int_4500)))
        epsilon.loc[i] = pd.Series({'Tau': SiAl_tot[i], 'epsilon_H2Om_5200': epsilon_H2Om_5200, 'sigma_epsilon_H2Om_5200': CT68_5200, 
            'epsilon_OH_4500': epsilon_OH_4500, 'sigma_epsilon_OH_4500': CT68_4500})

    # Doing density-H2O iterations:
    for l in Volatiles_PH.index: 
        if H2O_SAT_DENSITY['H2OT_3550_SAT'][l] == '-': 
            H2Om_5200_M = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['P_H5200_M'][l], Volatiles_PH['P_H5200_STD'][l],
                H2O_SAT_DENSITY['Density'][l], H2O_SAT_DENSITY['Density'][l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
            OH_4500_M = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['P_H4500_M'][l], Volatiles_PH['P_H4500_STD'][l],
                H2O_SAT_DENSITY['Density'][l], H2O_SAT_DENSITY['Density'][l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
            H2Om_5200_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['P_H5200_M'][l], Volatiles_PH['P_H5200_STD'][l],
                H2O_SAT_DENSITY['Density'][l], H2O_SAT_DENSITY['Density'][l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
            OH_4500_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['P_H4500_M'][l], Volatiles_PH['P_H4500_STD'][l],
                H2O_SAT_DENSITY['Density'][l], H2O_SAT_DENSITY['Density'][l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
            density_int = H2O_SAT_DENSITY['Density'][l]
        elif H2O_SAT_DENSITY['H2OT_3550_SAT'][l] == '*':
            for k in range(20):
                H2Om_5200_M = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['P_H5200_M'][l], Volatiles_PH['P_H5200_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
                OH_4500_M = Beer_Lambert(molar_mass['H2O'], Volatiles_PH['P_H4500_M'][l], Volatiles_PH['P_H4500_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
                H2Om_5200_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['P_H5200_M'][l], Volatiles_PH['P_H5200_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
                OH_4500_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_PH['P_H4500_M'][l], Volatiles_PH['P_H4500_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
                MI_Composition['H2O'][l] = H2O_SAT_DENSITY['H2Om_1635_BP'][l] + H2Om_5200_M
                mol, density = Density_Calculation(MI_Composition)
            density_int = density[l]
        mega_spreadsheet.loc[l] = pd.Series({'H2OT_3550_SAT': H2O_SAT_DENSITY['H2OT_3550_SAT'][l], 'H2Om_5200_M': H2Om_5200_M, 'H2Om_5200_M_STD': H2Om_5200_M_STD, 'OH_4500_M': OH_4500_M, 'OH_4500_M_STD': OH_4500_M_STD})
        s2nerror.loc[l] = pd.Series({'P_H5200_S2N': Volatiles_PH['S2N_P5200'][l], 'P_H4500_S2N': Volatiles_PH['S2N_P4500'][l], 'ERR_5200': Volatiles_PH['ERR_5200'][l], 'ERR_4500': Volatiles_PH['ERR_4500'][l]})
        density_df.loc[l] = pd.Series({'Density_F': density_int})
    mega_spreadsheet_f = pd.concat([mega_spreadsheet, s2nerror], axis = 1)
    epsilon_f = pd.concat([density_df, epsilon], axis = 1)

    return epsilon_f, mega_spreadsheet_f
