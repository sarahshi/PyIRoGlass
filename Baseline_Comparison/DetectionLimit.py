# %% -*- coding: utf-8 -*-
""" Created on April 15, 2024 // @author: Sarah Shi """

# Import packages
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.append('../src/')
import PyIRoGlass as pig

from scipy import stats
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import scipy.signal as signal

from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22

# %%

df_std = pd.read_csv('../FINALDATA/STD_DF.csv', index_col=0) 
volatiles_std = pd.read_csv('../FINALDATA/STD_H2OCO2.csv', index_col=0) 

p1515_std = df_std['P_1515_BP']
p1430_std = df_std['P_1430_BP']

sloader = pig.SampleDataLoader('../Inputs/TransmissionSpectra/Standards', '../Inputs/StandardChemThick.csv')
sdfs_dict, schem, sthick = sloader.load_all_data()

bppath = '../BLPEAKFILES/'
pklpath = '../PKLFILES/'


# %%

DL_std_i = pd.DataFrame(columns=['BL5200_BL_M', 'BL5200_BL_STD', 
                               'BL4500_BL_M', 'BL4500_BL_STD', 
                               'BL3550_BL_M', 'BL3550_BL_STD', 
                               'BL1635_M', 'BL1635_STD', 
                               'BL1515_M', 'BL1515_STD', 
                               'BL1430_M', 'BL1430_STD'])

num_samples = 50
p_mean = 0.001
p_std = 0.00025
p_values = np.random.normal(loc=p_mean, scale=p_std, size=num_samples)

lambda_mean_MIR = np.log(10**11)  # Converting the mean to log-scale
lambda_std_MIR = 0.01  # Small standard deviation in log-scale
lambda_values_log_MIR = np.random.normal(loc=lambda_mean_MIR, scale=lambda_std_MIR, size=num_samples)
lambda_MIR = np.exp(lambda_values_log_MIR)  # Convert back to the original scale
lambda_min_MIR = 10**10.99
lambda_max_MIR = 10**11.01
lambda_MIR = np.clip(lambda_MIR, lambda_min_MIR, lambda_max_MIR)

lambda_mean_3550 = np.log(10**9)  # Converting the mean to log-scale
lambda_std_3550 = 0.01  # Small standard deviation in log-scale
lambda_values_log_3550 = np.random.normal(loc=lambda_mean_3550, scale=lambda_std_3550, size=num_samples)
lambda_3550 = np.exp(lambda_values_log_3550)  # Convert back to the original scale
lambda_min_3550 = 10**8.99
lambda_max_3550 = 10**9.01
lambda_3550 = np.clip(lambda_3550, lambda_min_3550, lambda_max_3550)

def filter_func(column, first_lower, first_upper, last_lower, last_upper):
    if len(column) < 50:  # Ensure there are enough data points
        return False
    return (first_lower <= column.iloc[0] <= first_upper) and \
           (last_lower <= column.iloc[-1] <= last_upper)

def select_random_columns(group):
    return group.sample(n=20, axis=1, random_state=42)  # Adjust random_state for different random samples

counter = 0 
for files, data in sdfs_dict.items():

    counter += 1
    bestfits = pd.read_csv(bppath+'STD/'+files+'_bestfits.csv', index_col=0)
    baselines = pd.read_csv(bppath+'STD/'+files+'_baselines.csv', index_col=0)
    baselines = baselines.iloc[-350:]

    with open(pklpath+'STD/'+files+".pkl", "rb") as handle:
        als_bls = pickle.load(handle)
    
    H2Om_5200_results = als_bls["H2Om_5200_results"]
    OH_4500_results = als_bls["OH_4500_results"]
    H2Ot_3550_results = als_bls["H2Ot_3550_results"]

    PH_5200_krige = [result["PH_krige"] for result in H2Om_5200_results]
    PH_5200_krige_M, PH_5200_krige_STD = (np.mean(PH_5200_krige),
                                          np.std(PH_5200_krige))
    PH_4500_krige = [result["PH_krige"] for result in OH_4500_results]
    PH_4500_krige_M, PH_4500_krige_STD = (np.mean(PH_4500_krige),
                                          np.std(PH_4500_krige))
    PH_3550_krige = [result["PH"] for result in H2Ot_3550_results]
    PH_3550_krige_M, PH_3550_krige_STD = (np.mean(PH_3550_krige),
                                          np.std(PH_3550_krige))

    H2Om_5200_baselines = [result['peak_fit']['Baseline_NIR'] for result in H2Om_5200_results]
    differences_5200 = [np.abs(result.index-5200) for result in H2Om_5200_baselines]
    closest_5200 = [result.argmin() for result in differences_5200]
    absorbance_5200 = [H2Om_5200_baselines[i].iloc[closest_5200[i]] for i in range(0, len(closest_5200))]
    mean_5200_bl = np.mean(absorbance_5200)
    std_5200_bl = np.std(absorbance_5200)

    OH_4500_baselines = [result['peak_fit']['Baseline_NIR'] for result in OH_4500_results]
    differences_4500 = [np.abs(result.index-4500) for result in OH_4500_baselines]
    closest_4500 = [result.argmin() for result in differences_4500]
    absorbance_4500 = [OH_4500_baselines[i].iloc[closest_4500[i]] for i in range(0, len(closest_4500))]
    mean_4500_bl = np.mean(absorbance_4500)
    std_4500_bl = np.std(absorbance_4500)

    H2Ot_3550_baselines = [result['peak_fit']['Baseline_MIR'] for result in H2Ot_3550_results]
    differences_3550 = [np.abs(result.index-3550) for result in H2Ot_3550_baselines]
    closest_3550 = [result.argmin() for result in differences_3550]
    absorbance_3550 = [H2Ot_3550_baselines[i].iloc[closest_3550[i]] for i in range(0, len(closest_3550))]
    mean_3550_bl = np.mean(absorbance_3550)
    std_3550_bl = np.std(absorbance_3550)

    H2Om_5200_orig = H2Om_5200_results[0]["peak_fit"]["Peak_Subtract"]
    H2Om_5200_krige = H2Om_5200_results[0]["peak_krige"]["Absorbance"]
    H2Om_5200_krige_interp = interpolate.interp1d(H2Om_5200_krige.index, H2Om_5200_krige.values)(H2Om_5200_orig.index)
    H2Om_5200_resid = H2Om_5200_orig - H2Om_5200_krige_interp
    H2Om_5200_noise_M, H2Om_5200_noise_STD = (np.mean(H2Om_5200_resid),
                                              np.std(H2Om_5200_resid))
    OH_4500_orig = OH_4500_results[0]["peak_fit"]["Peak_Subtract"]
    OH_4500_krige = OH_4500_results[0]["peak_krige"]["Absorbance"]
    OH_4500_krige_interp = interpolate.interp1d(OH_4500_krige.index, OH_4500_krige.values)(OH_4500_orig.index)
    OH_4500_resid = OH_4500_orig - OH_4500_krige_interp
    OH_4500_noise_M, OH_4500_noise_STD = (np.mean(OH_4500_resid),
                                          np.std(OH_4500_resid))
    H2Ot_3550_peak = H2Ot_3550_results[0]["plot_output"]["Peak_Subtract_Filt"] + H2Ot_3550_results[0]["plot_output"]["Baseline_MIR"]
    baseline = 0
    H2Ot_3550_filt = signal.medfilt(H2Ot_3550_peak, 3)
    baseline = signal.savgol_filter(H2Ot_3550_filt, 21, 3)
    H2Ot_3550_filt = H2Ot_3550_filt - baseline
    H2Ot_3550_noise_M, H2Ot_3550_noise_STD = (np.mean(H2Ot_3550_filt),
                                              np.std(H2Ot_3550_filt))

    ploc_1515 = p1515_std.loc[files]
    ploc_1430 = p1430_std.loc[files]

    difference_1635 = np.abs(baselines.index - 1635)
    closest_1635 = difference_1635.argmin()
    difference_1515 = np.abs(baselines.index - ploc_1515)
    closest_1515 = difference_1515.argmin()
    difference_1430 = np.abs(baselines.index - ploc_1430)
    closest_1430 = difference_1430.argmin()

    absorbance_1635 = baselines.iloc[closest_1635]
    absorbance_1515 = baselines.iloc[closest_1515]
    absorbance_1430 = baselines.iloc[closest_1430]

    mean_1635_bl = np.mean(absorbance_1635)
    std_1635_bl = np.std(absorbance_1635)
    mean_1515_bl = np.mean(absorbance_1515)
    std_1515_bl = np.std(absorbance_1515)
    mean_1430_bl = np.mean(absorbance_1430)
    std_1430_bl = np.std(absorbance_1430)

    DL_std_i.loc[files] = pd.Series(
        {
            "BL5200_BL_M": mean_5200_bl,
            "BL5200_BL_STD": std_5200_bl,
            "BL4500_BL_M": mean_4500_bl,
            "BL4500_BL_STD": std_4500_bl,
            "BL3550_BL_M": mean_3550_bl,
            "BL3550_BL_STD": std_3550_bl,
            "BL1635_M": mean_1635_bl,
            "BL1635_STD": std_1635_bl,
            "BL1515_M": mean_1515_bl,
            "BL1515_STD": std_1515_bl,
            "BL1430_M": mean_1430_bl,
            "BL1430_STD": std_1430_bl,
        }
    )

    # if counter % 100 == 1:
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(data.index, data['Absorbance'], label='FTIR Spectrum')
    #     plt.plot(bestfits.index, bestfits['Spectrum_Fit'], label='Spectrum Fit')
    #     plt.plot(baselines.index, baselines, 'k')
    #     plt.plot(baselines.index, baselines['Baseline_0'], label='Baseline')
    #     # Setting labels and titles
    #     plt.xlabel('Wavenumber')
    #     plt.ylabel('Absorbance')
    #     plt.xlim([2400, 1250])
    #     plt.title(files)
    #     plt.legend()
    #     plt.show()




    # absorbance_3550 = H2Ot_3550_results[0]['peak_fit'].loc[2100:4200]['Absorbance']
    # index_3550 = H2Ot_3550_results[0]['peak_fit'].loc[2100:4200].index
    # absorbance_5200 = H2Om_5200_results[0]['peak_fit'].loc[4875:5400]['Absorbance']
    # index_5200 = H2Om_5200_results[0]['peak_fit'].loc[4875:5400].index
    # absorbance_4500 = OH_4500_results[0]['peak_fit'].loc[4250:4675]['Absorbance']
    # index_4500 = OH_4500_results[0]['peak_fit'].loc[4250:4675].index

    # baseline_data_3550 = []
    # baseline_data_5200 = []
    # baseline_data_4500 = []

    # for p, s in zip(p_3550, lambda_3550):
    #     baseline_3550 = pig.als_baseline(absorbance_3550, asymmetry_param=p, smoothness_param=s)
    #     baseline_data_3550.append(baseline_3550)

    # for p, s in zip(p_3550, lambda_MIR):
    #     baseline_5200 = pig.als_baseline(absorbance_5200, asymmetry_param=p, smoothness_param=s)
    #     baseline_data_5200.append(baseline_5200)
    #     baseline_4500 = pig.als_baseline(absorbance_4500, asymmetry_param=p, smoothness_param=s)
    #     baseline_data_4500.append(baseline_4500)

    # baseline_array_3550 = np.array(baseline_data_3550).T
    # baseline_df_3550 = pd.DataFrame(baseline_array_3550, index=index_3550)
    # first_mean_3550 = absorbance_3550.values[0:40].mean()
    # last_mean_3550 = absorbance_3550.values[-40:].mean()
    # first_std_3550 = absorbance_3550.values[0:40].std()
    # last_std_3550 = absorbance_3550.values[-40:].std()
    # valid_columns_3550 = baseline_df_3550.apply(
    #     lambda column: filter_func(column,
    #     first_mean_3550-3*first_std_3550,
    #     first_mean_3550+3*first_std_3550,
    #     last_mean_3550-3*last_std_3550,
    #     last_mean_3550+3*last_std_3550), axis=0)
    # filtered_baselines_3550 = baseline_df_3550.loc[:, valid_columns_3550]

    # baseline_array_5200 = np.array(baseline_data_5200).T
    # baseline_df_5200 = pd.DataFrame(baseline_array_5200, index=index_5200)
    # first_mean_5200 = absorbance_5200.values[0:40].mean()
    # last_mean_5200 = absorbance_5200.values[-40:].mean()
    # first_std_5200 = absorbance_5200.values[0:40].std()
    # last_std_5200 = absorbance_5200.values[-40:].std()
    # valid_columns_5200 = baseline_df_5200.apply(
    #     lambda column: filter_func(column,
    #     first_mean_5200-3*first_std_5200,
    #     first_mean_5200+3*first_std_5200,
    #     last_mean_5200-3*last_std_5200,
    #     last_mean_5200+3*last_std_5200), axis=0)
    # filtered_baselines_5200 = baseline_df_5200.loc[:, valid_columns_5200]

    # baseline_array_4500 = np.array(baseline_data_4500).T
    # baseline_df_4500 = pd.DataFrame(baseline_array_4500, index=index_4500)
    # first_mean_4500 = absorbance_4500.values[0:40].mean()
    # last_mean_4500 = absorbance_4500.values[-40:].mean()
    # first_std_4500 = absorbance_4500.values[0:40].std()
    # last_std_4500 = absorbance_4500.values[-40:].std()
    # valid_columns_4500 = baseline_df_4500.apply(
    #     lambda column: filter_func(column,
    #     first_mean_4500-3*first_std_4500,
    #     first_mean_4500+3*first_std_4500,
    #     last_mean_4500-3*last_std_4500,
    #     last_mean_4500+3*last_std_4500), axis=0)
    # filtered_baselines_4500 = baseline_df_4500.loc[:, valid_columns_4500]

    # differences_3550 = np.abs(index_3550 - 3550)
    # closest_idx_3550 = differences_3550.argmin()
    # # absorbance_3550 = absorbance_3550.iloc[closest_idx_3550]
    # absorbance_3550 = absorbance_3550[closest_idx_3550]
    # mean_3550_bl = absorbance_3550.mean()
    # std_3550_bl = absorbance_3550.std()

    # differences_5200 = np.abs(index_5200 - 5200)
    # closest_idx_5200 = differences_5200.argmin()
    # # absorbance_5200 = absorbance_5200.iloc[closest_idx_5200]
    # absorbance_5200 = absorbance_5200[closest_idx_5200]
    # mean_5200_bl = absorbance_5200.mean()
    # std_5200_bl = absorbance_5200.std()

    # differences_4500 = np.abs(index_4500 - 4500)
    # closest_idx_4500 = differences_4500.argmin()
    # # absorbance_4500 = absorbance_4500.iloc[closest_idx_4500]
    # absorbance_4500 = absorbance_4500[closest_idx_4500]
    # mean_4500_bl = absorbance_4500.mean()
    # std_4500_bl = absorbance_4500.std()


DL_std = pd.concat([DL_std_i, sthick['Thickness'], df_std[['H2Ot_3550_SAT', 'ERR_5200', 'ERR_4500']]], axis=1)
DL_std.to_csv('DL_Standards.csv')

# %% 

display(DL_std)

volatiles_std = volatiles_std[~np.isnan(DL_std.Thickness)]
DL_std = DL_std[~np.isnan(DL_std.Thickness)]

fig, ax = plt.subplots(1, 3, figsize=(18, 7))
ax = ax.flatten()
ax[0].scatter(DL_std.Thickness, DL_std.BL1635_STD*3)
ax[0].set_xlabel('Thickness')
ax[0].set_ylabel('BL 1515 STD')
ax[1].scatter(DL_std.Thickness, DL_std.BL1515_STD*3)
ax[1].set_xlabel('Thickness')
ax[1].set_ylabel('BL 1515 STD')
ax[2].scatter(DL_std.Thickness, DL_std.BL1430_STD*3)
ax[2].set_xlabel('Thickness')
ax[2].set_ylabel('BL 1430 STD')
ax[1].set_title('All Standards')
plt.tight_layout()
plt.show()

# %% 

DL_std['DL_5200'] = ((1e6 * 18.01528 * 3*DL_std['BL5200_BL_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_H2Om_5200']))
DL_std['DL_4500'] = ((1e6 * 18.01528 * 3*DL_std['BL4500_BL_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_OH_4500']))
DL_std['DL_3550'] = ((1e6 * 18.01528 * 3*DL_std['BL3550_BL_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_H2Ot_3550']))

DL_std['DL_1635'] = ((1e6 * 18.01528 * 3*DL_std['BL1635_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_H2Om_1635']))
DL_std['DL_1515'] = 1e4 * ((1e6 * 44.01 * 3*DL_std['BL1515_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_CO2']))
DL_std['DL_1430'] = 1e4 * ((1e6 * 44.01 * 3*DL_std['BL1430_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_CO2']))

# DL_std.to_csv('DL_Standards.csv')
# DL_std['DL_5200_noise'] = ((1e6 * 18.01528 * DL_std['BL5200_noise_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_H2Om_5200']))
# DL_std['DL_4500_noise'] = ((1e6 * 18.01528 * DL_std['BL4500_noise_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_OH_4500']))
# DL_std['DL_3550_noise'] = ((1e6 * 18.01528 * DL_std['BL3550_noise_STD']) / (volatiles_std['Density_Sat'] * DL_std['Thickness'] * volatiles_std['epsilon_H2Ot_3550']))

DL_std_sat = DL_std[(DL_std.ERR_5200=='-') & (DL_std.ERR_4500=='-') & (~DL_std.index.str.contains('ETF')) & (~DL_std.index.str.contains('CN92')) & (DL_std.DL_4500 > 0.002) & (DL_std.DL_5200 > 0.002) & (DL_std.DL_5200 < 0.3) & (DL_std.DL_4500 < 0.3)]
volatiles_sat = volatiles_std[(DL_std.ERR_5200=='-') & (DL_std.ERR_4500=='-') & (~DL_std.index.str.contains('ETF')) & (~DL_std.index.str.contains('CN92')) & (DL_std.DL_4500 > 0.002) & (DL_std.DL_5200 > 0.002) & (DL_std.DL_5200 < 0.3) & (DL_std.DL_4500 < 0.3)]

DL_std_unsat = DL_std[(DL_std.ERR_5200=='-') & (DL_std.ERR_4500=='-') & (DL_std.H2Ot_3550_SAT=='-') & (~DL_std.index.str.contains('ETF')) & (~DL_std.index.str.contains('CN92')) & (DL_std.DL_3550 > 0.0005)]
volatiles_unsat = volatiles_std[(DL_std.ERR_5200=='-') & (DL_std.ERR_4500=='-') & (DL_std.H2Ot_3550_SAT=='-') & (~DL_std.index.str.contains('ETF')) & (~DL_std.index.str.contains('CN92')) & (DL_std.DL_3550 > 0.0005) ]


# def model(x, a, b, c):
#     return a * np.exp(-b * x) + c 
# def fit_exponential_decay(thickness, concentration):
#     params, cov = curve_fit(model, thickness, concentration)
#     return params, cov

def model(x, a, b):
    return a / x + b

def fit_exponential_decay(thickness, concentration, p0):
    
    try: 
        params, cov = curve_fit(model, thickness, concentration, p0, maxfev=10000)
        return params, cov

    except RuntimeError as e:
        print("Error: ", e)
        return p0, None  # Returning initial parameters and None for covariance to indicate failure

def sort_data(thickness, concentration):
    sorted_indices = np.argsort(thickness)
    return thickness[sorted_indices], concentration[sorted_indices]

def calculate_intervals(thickness, concentration, params, cov): 

    thickness_sorted, concentration_sorted = sort_data(thickness, concentration)

    # jacobian matrix taken with respect to a, b, c
    J = np.vstack([
        1/thickness_sorted,  # derivative with respect to a
        np.ones_like(thickness_sorted)  # derivative with respect to b
    ]).T

    alpha = 0.32
    n = len(concentration_sorted)
    p = len(params)
    dof = max(0, n-p)
    if dof == 0:
        raise ValueError("Degrees of freedom are zero, adjust your data or model.")

    tval = stats.t.ppf((1.0-alpha)/2, dof)
    ci = tval * np.sqrt((J @ cov @ J.T).diagonal())
    pi = tval * np.sqrt((J @ cov @ J.T).diagonal() + ((concentration_sorted-model(thickness_sorted, *params))**2).sum()/dof)

    return ci, pi

params_1635, cov_1635 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1635'], p0=[0.4, 0.0001])
params_1515, cov_1515 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1515'], p0=[2500, -4])
params_1430, cov_1430 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1430'], p0=[2500, -4])
params_5200, cov_5200 = fit_exponential_decay(DL_std_sat.Thickness, DL_std_sat['DL_5200'], p0=[1, -1])
params_4500, cov_4500 = fit_exponential_decay(DL_std_sat.Thickness, DL_std_sat['DL_4500'], p0=[1, -1])
params_3550, cov_3550 = fit_exponential_decay(DL_std_unsat.Thickness, DL_std_unsat['DL_3550'], p0=[0.1, -1])

# params_1635, cov_1635 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1635'], p0=[0.01, 0.01, 0.01])
# params_1515, cov_1515 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1515'], p0=[200, 0.1, 0.1])
# params_1430, cov_1430 = fit_exponential_decay(DL_std.Thickness, DL_std['DL_1430'], p0=[200, 0.1, 0.1])
# params_5200, cov_5200 = fit_exponential_decay(DL_std_sat.Thickness, DL_std_sat['DL_5200'], p0=[0.01, 0.5, 0.01])
# params_4500, cov_4500 = fit_exponential_decay(DL_std_sat.Thickness, DL_std_sat['DL_4500'], p0=[0.01, 0.5, 0.01])
# params_3550, cov_3550 = fit_exponential_decay(DL_std_unsat.Thickness, DL_std_unsat['DL_3550'], p0=[0.01, 0.01, 0.01])


x_range = np.linspace(0, 250, 344)
curve_1635 = model(x_range, *params_1635)
curve_1515 = model(x_range, *params_1515)
curve_1430 = model(x_range, *params_1430)
pred_1635 = model(DL_std.Thickness, *params_1635)
pred_1515 = model(DL_std.Thickness, *params_1515)
pred_1430 = model(DL_std.Thickness, *params_1430)

x_range_NIR = np.linspace(0, 250, len(DL_std_sat.Thickness))
x_range_MIR = np.linspace(0, 250, len(DL_std_unsat.Thickness))
curve_5200 = model(x_range_NIR, *params_5200)
curve_4500 = model(x_range_NIR, *params_4500)
curve_3550 = model(x_range_MIR, *params_3550)
pred_5200 = model(DL_std_sat.Thickness, *params_5200)
pred_4500 = model(DL_std_sat.Thickness, *params_4500)
pred_3550 = model(DL_std_unsat.Thickness, *params_3550)

func_1635 = ('$\mathregular{H_2O_{m, 1635}=1.0877/\ell-0.0007046}$')
func_1515 = ('$\mathregular{CO_{2, 1515}=2565/\ell-3.9838}$')
func_1430 = ('$\mathregular{CO_{2, 1430}=2565/\ell-3.9838}$')
r2_1635 = pig.calculate_R2(DL_std.DL_1635, pred_1635)
r2_1515 = pig.calculate_R2(DL_std.DL_1515, pred_1515)
r2_1430 = pig.calculate_R2(DL_std.DL_1430, pred_1430)
rmse_1635 = pig.calculate_RMSE(DL_std.DL_1635-pred_1635)
rmse_1515 = pig.calculate_RMSE(DL_std.DL_1515-pred_1515)
rmse_1430 = pig.calculate_RMSE(DL_std.DL_1430-pred_1430)
rrmse_1635 = pig.calculate_RRMSE(DL_std.DL_1635, pred_1635)
rrmse_1515 = pig.calculate_RRMSE(DL_std.DL_1515, pred_1515)
rrmse_1430 = pig.calculate_RRMSE(DL_std.DL_1430, pred_1430)

func_5200 = ('$\mathregular{H_2O_{m, 5200}=1.248/\ell+0.03785}$')
func_4500 = ('$\mathregular{OH_{4500}^-=1.663/\ell+0.01604}$')
func_3550 = ('$\mathregular{H_2O_{t, 3550}=0.2597/\ell+0.002797}$')
r2_5200 = pig.calculate_R2(DL_std_sat.DL_5200, pred_5200)
r2_4500 = pig.calculate_R2(DL_std_sat.DL_4500, pred_4500)
r2_3550 = pig.calculate_R2(DL_std_unsat.DL_3550, pred_3550)
rmse_5200 = pig.calculate_RMSE(DL_std_sat.DL_5200-pred_5200)
rmse_4500 = pig.calculate_RMSE(DL_std_sat.DL_4500-pred_4500)
rmse_3550 = pig.calculate_RMSE(DL_std_unsat.DL_3550-pred_3550)
rrmse_5200 = pig.calculate_RRMSE(DL_std_sat.DL_5200, pred_5200)
rrmse_4500 = pig.calculate_RRMSE(DL_std_sat.DL_4500, pred_4500)
rrmse_3550 = pig.calculate_RRMSE(DL_std_unsat.DL_3550, pred_3550)

ci_1635, pi_1635 = calculate_intervals(DL_std.Thickness, DL_std['DL_1635'], params_1635, cov_1635)
ci_1515, pi_1515 = calculate_intervals(DL_std.Thickness, DL_std['DL_1515'], params_1515, cov_1515)
ci_1430, pi_1430 = calculate_intervals(DL_std.Thickness, DL_std['DL_1430'], params_1430, cov_1430)
ci_5200, pi_5200 = calculate_intervals(DL_std_sat.Thickness, DL_std_sat['DL_5200'], params_5200, cov_5200)
ci_4500, pi_4500 = calculate_intervals(DL_std_sat.Thickness, DL_std_sat['DL_4500'], params_4500, cov_4500)
ci_3550, pi_3550 = calculate_intervals(DL_std_unsat.Thickness, DL_std_unsat['DL_3550'], params_3550, cov_3550)

from matplotlib.ticker import MultipleLocator

sz = 80
fig, ax = plt.subplots(2, 3, figsize=(20, 12))
ax = ax.flatten()
ax[0].plot(x_range, curve_1635, 'k', lw=1, label=func_1635)
ax[0].scatter(DL_std.Thickness, DL_std.DL_1635, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[0].fill_between(x_range, curve_1635-ci_1635, curve_1635+ci_1635, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[0].plot(x_range, curve_1635-pi_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[0].plot(x_range, curve_1635+pi_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[0].set_xlabel('Thickness (µm)')
ax[0].set_ylabel(r'$\mathregular{H_2O_{m, 1635}~Detection~Limit~(wt.\%)}$')
ax[0].set_xlim([0, 250])
ax[0].set_ylim([0, 0.1])
ax[0].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[0].legend(loc=(0.30, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[0].annotate("RMSE="+str(np.round(rmse_1635, 3))+"; RRMSE=16%", xy=(0.32, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[0].annotate('$\mathregular{R^2=}$'+f'{round(r2_1635, 3)}', xy=(0.32, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[0].annotate('A.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))
ax[0].xaxis.set_minor_locator(MultipleLocator(10))
ax[0].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

ax[1].plot(x_range, curve_1515, 'k', lw=1, label=func_1515)
ax[1].scatter(DL_std.Thickness, DL_std.DL_1515, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[1].fill_between(x_range, curve_1515-ci_1515, curve_1515+ci_1515, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[1].plot(x_range, curve_1515-pi_1515, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[1].plot(x_range, curve_1515+pi_1515, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[1].set_xlabel('Thickness (µm)')
ax[1].set_ylabel(r'$\mathregular{CO_{2, 1515}~Detection~Limit~(ppm)}$')
ax[1].set_xlim([0, 250])
ax[1].set_ylim([0, 200])
ax[1].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[1].legend(loc=(0.42, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[1].annotate("RMSE="+str(np.round(rmse_1515, 3))+"; RRMSE=12%", xy=(0.435, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[1].annotate('$\mathregular{R^2=}$'+f'{round(r2_1515, 3)}', xy=(0.435, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[1].annotate('B.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))
ax[1].xaxis.set_minor_locator(MultipleLocator(10))
ax[1].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

ax[2].plot(x_range, curve_1430, 'k', lw=2, label=func_1430)
ax[2].scatter(DL_std.Thickness, DL_std.DL_1430, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[2].fill_between(x_range, curve_1430-ci_1430, curve_1430+ci_1430, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[2].plot(x_range, curve_1430-pi_1430, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[2].plot(x_range, curve_1430+pi_1430, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[2].set_xlabel('Thickness (µm)')
ax[2].set_ylabel(r'$\mathregular{CO_{2, 1430}~Detection~Limit~(ppm)}$')
ax[2].set_xlim([0, 250])
ax[2].set_ylim([0, 200])
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[2].legend(loc=(0.42, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[2].annotate("RMSE="+str(np.round(rmse_1430, 3))+"; RRMSE=12%", xy=(0.435, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[2].annotate('$\mathregular{R^2=}$'+f'{round(r2_1430, 3)}', xy=(0.435, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[2].annotate('C.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'), zorder=20)
ax[2].xaxis.set_minor_locator(MultipleLocator(10))
ax[2].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

ax[3].plot(x_range_NIR, curve_5200, 'k', lw=1, label=func_5200)
ax[3].scatter(DL_std_sat.Thickness, DL_std_sat.DL_5200, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[3].fill_between(x_range_NIR, curve_5200-ci_5200, curve_5200+ci_5200, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[3].plot(x_range_NIR, curve_5200-pi_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[3].plot(x_range_NIR, curve_5200+pi_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[3].set_xlabel('Thickness (µm)')
ax[3].set_ylabel(r'$\mathregular{H_2O_{m, 5200}~Detection~Limit~(wt.\%)}$')
ax[3].set_xlim([0, 250])
ax[3].set_ylim([0, 0.3])
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[3].legend(loc=(0.36, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[3].annotate("RMSE="+str(np.round(rmse_5200, 3))+"; RRMSE=76%", xy=(0.38, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[3].annotate('$\mathregular{R^2=}$'+f'{round(r2_5200, 3)}', xy=(0.38, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[3].annotate('D.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))
ax[3].xaxis.set_minor_locator(MultipleLocator(10))
ax[3].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

ax[4].plot(x_range_NIR, curve_4500, 'k', lw=2, label=func_4500)
ax[4].scatter(DL_std_sat.Thickness, DL_std_sat.DL_4500, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[4].fill_between(x_range_NIR, curve_4500-ci_4500, curve_4500+ci_4500, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[4].plot(x_range_NIR, curve_4500-pi_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[4].plot(x_range_NIR, curve_4500+pi_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[4].set_xlabel('Thickness (µm)')
ax[4].set_ylabel(r'$\mathregular{OH_{4500}^-~Detection~Limit~(wt.\%)}$')
ax[4].set_xlim([0, 250])
ax[4].set_ylim([0, 0.3])
ax[4].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[4].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[4].legend(loc=(0.41, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[4].annotate("RMSE="+str(np.round(rmse_4500, 3))+"; RRMSE=94%", xy=(0.43, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[4].annotate('$\mathregular{R^2=}$'+f'{round(r2_4500, 3)}', xy=(0.43, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[4].annotate('E.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))
ax[4].xaxis.set_minor_locator(MultipleLocator(10))
ax[4].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

ax[5].plot(x_range_MIR, curve_3550, 'k', lw=1, label=func_3550)
ax[5].scatter(DL_std_unsat.Thickness, DL_std_unsat.DL_3550, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20, label='Standards')
ax[5].fill_between(x_range_MIR, curve_3550-ci_3550, curve_3550+ci_3550, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[5].plot(x_range_MIR, curve_3550-pi_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[5].plot(x_range_MIR, curve_3550+pi_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[5].set_xlabel('Thickness (µm)')
ax[5].set_ylabel(r'$\mathregular{H_2O_{t, 3550}~Detection~Limit~(wt.\%)}$')
ax[5].set_xlim([0, 250])
ax[5].set_ylim([0, 0.05])
ax[5].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[5].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[5].legend(loc=(0.33, 0.755), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[5].annotate("RMSE="+str(np.round(rmse_3550, 3))+"; RRMSE=83%", xy=(0.35, 0.73), xycoords="axes fraction", ha='left', fontsize=14)
ax[5].annotate('$\mathregular{R^2=}$'+f'{round(r2_3550, 3)}', xy=(0.35, 0.68), xycoords="axes fraction", ha='left', fontsize=14)
ax[5].annotate('F.', xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))
ax[5].xaxis.set_minor_locator(MultipleLocator(10))
ax[5].tick_params(axis='x', which='minor', direction='in', length=3.5, pad=6.5)

plt.tight_layout()
plt.savefig('DetectionLimit_new1.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()


# %% 
