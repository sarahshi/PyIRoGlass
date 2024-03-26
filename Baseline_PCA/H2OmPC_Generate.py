# %%

import sys
import numpy as np
import pandas as pd

sys.path.append("../src/")
import PyIRoGlass as pig 

from sklearn.decomposition import PCA
from scipy import signal

from matplotlib import pyplot as plt
from matplotlib import rc

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

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.major.pad'] = 6.5
plt.rcParams['ytick.major.pad'] = 6.5

# %%


def savgol_filter(x, smooth_width, poly_order):
    """
    Applies a Savitzky-Golay filter to smooth and differentiate data.

    Parameters:
        x (numeric array): The data to be filtered.
        smooth_width (int): The length of the filter window (must be a
            positive odd integer).
        poly_order (int): The order of the polynomial used to fit the samples.

    Returns:
        filtered (numeric array): The smoothed data.
    """

    return signal.savgol_filter(x, smooth_width, poly_order)


def basic_scale_data(spectrum):
    """
    Scales the input data to a range of -0.5 to 0.5 based on the start and
    end points of the data matrix, then recenters it by subtracting the
    mean spectrum.

    Parameters:
        spectrum (numeric 2D array): The matrix containing spectral data for
            scaling. Each column represents one spectrum.

    Returns:
        spectrum_scale (numeric 2D array): The data after scaling to the range
            of -0.5 to 0.5 and recentring by the mean spectrum.
        mean_baseline (numeric array): The mean baseline spectrum across all
            observations after scaling.
    """

    data_range = spectrum[0, :] - spectrum[-1, :]

    scaled_data = spectrum / np.abs(data_range)

    spectrum_scale = scaled_data - scaled_data[0, :] + 0.5
    mean_baseline = spectrum.mean(axis=1)

    return spectrum_scale, mean_baseline


def subtract_mean(data):
    """
    Normalizes the input data by its range (max - min) and then subtracts
    the mean spectrum from each observation. This function aims to center
    the data around 0 by removing the average trend.

    Parameters:
        data (numeric 2D array): The matrix containing spectral data. Each
            column represents one observation/spectrum.

    Returns:
        normalized_data (numeric 2D array): The data after normalization and
            mean subtraction. Each observation is centered around 0.
        mean_spectrum (numeric array): The mean spectrum computed across all
            observations before mean subtraction. This represents the average
            trend that was present in the original data.
    """

    data_range = data.max(axis=0) - data.min(axis=0)
    data = data / data_range
    mean_peak = data.mean(axis=1)

    data = data - np.array([mean_peak]).T

    return data, mean_peak


def perform_PCA(data, n_vectors=12):
    """
    Performs Principal Component Analysis (PCA) on the provided data matrix.
    This function identifies the principal components that capture the most
    variance in the data and projects the original data onto these PCs.

    Parameters:
        data (numeric 2D array): The matrix containing spectral data. Each
            column represents one observation.
        n_vectors (int, optional): The number of principal components.

    Returns:
        PC_vectors (numeric 2D array): The principal component vectors.
            Each row represents one principal component.
        reduced_data (numeric 2D array): The original data projected onto
            principal component space.
        variance_norm (numeric array): The normalized variance explained
            by each of the principal components.
    """

    pca = PCA(
        n_vectors,
    )
    reduced_data = pca.fit(data.T).transform(data.T)

    variance = pca.explained_variance_
    variance_norm = variance[0:-1] / np.sum(variance[0:-1])
    PC_vectors = pca.components_

    return PC_vectors, reduced_data, variance_norm


# %%

# wn_low, wn_high = (1400, 1750)
wn_low, wn_high = (1390, 1780)
H2Om_df = pd.read_csv("H2Om1635_BL_Removed.csv", index_col="Wavenumber")

H2Om_df_lim = H2Om_df.loc[wn_low:wn_high]
H2Om_df_i = H2Om_df_lim.values
wavenumber = H2Om_df_lim.loc[wn_low:wn_high].index
wavenumber_full = H2Om_df.loc[1250:2400].index
print(np.shape(H2Om_df))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, H2Om_df_lim)
ax[0].set_title("H$\mathregular{_2}$O$\mathregular{_m}$ Peak")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

basic_data_scale, basic_data_mean = basic_scale_data(H2Om_df_i)
ax[1].plot(wavenumber, basic_data_scale)
ax[1].set_title("H$\mathregular{_2}$O$\mathregular{_m}$ Peak, Rescaled")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
plt.tight_layout()
plt.show()


# %%

mean_subtracted_data, mean_peak = subtract_mean(H2Om_df_i)
mean_peak_reshaped = mean_peak[:, np.newaxis]
add_data = mean_subtracted_data + mean_peak_reshaped

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    wavenumber, mean_peak, "k", label="Mean H$\mathregular{_2}$O$\mathregular{_m}$ Peak"
)
ax.plot(wavenumber, mean_subtracted_data[:, 0], label="Mean Subtracted Data")
ax.plot(wavenumber, mean_subtracted_data)
ax.set_title("H$\mathregular{_2}$O$\mathregular{_m}$ Peak, Mean Subtracted")
ax.legend(prop={"size": 12})
ax.invert_xaxis()
plt.tight_layout()
plt.show()

# %%

PC_vectors, reduced_data, variance_norm = perform_PCA(mean_subtracted_data)
mean_peak_smooth = savgol_filter(mean_peak, 31, 6)
PC1_smooth = savgol_filter(PC_vectors[0], 31, 6)
PC2_smooth = savgol_filter(PC_vectors[1], 31, 6)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(wavenumber, mean_peak, label="Mean H2Om1635")
ax.plot(wavenumber, mean_peak_smooth, label="Mean H2Om1635 smooth")
ax.plot(wavenumber, PC_vectors[0], label="PC1")
ax.plot(wavenumber, PC1_smooth, label="PC1 smooth")
ax.plot(wavenumber, PC_vectors[1], label="PC2")
ax.plot(wavenumber, PC2_smooth, label="PC2 smooth")
ax.set_title("H$\mathregular{_2}$O$\mathregular{_m}$ Peak Principal Components, Smoothed")
ax.set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax.set_ylabel("Absorbance")
ax.set_xlim([1460, 1750])
ax.set_ylim([-0.2, 0.2])
ax.invert_xaxis()
ax.legend(prop={"size": 12})
plt.tight_layout()
plt.show()

# %%

wn_low_init = 1250
wn_high_init = 2400
wn_stitch_low, wn_stitch_high = (1498, 1752)

peak_components = np.array([mean_peak_smooth, PC1_smooth, PC2_smooth])

peak_components_df = pd.DataFrame(
    peak_components.T,
    index=wavenumber,
    columns=[
        "1635_Peak_Mean",
        "1635_Peak_PC1",
        "1635_Peak_PC2",
    ],
)

zeros_df = pd.DataFrame(
    np.zeros_like(wavenumber_full, shape=(len(wavenumber_full), 3)),
    index=wavenumber_full,
    columns=[
        "1635_Peak_Mean",
        "1635_Peak_PC1",
        "1635_Peak_PC2",
    ],
)

peak_components_df_full = pd.concat(
    [
        zeros_df.loc[wn_low_init:wn_stitch_low],
        peak_components_df.loc[wn_stitch_low:wn_stitch_high],
        zeros_df.loc[wn_stitch_high:wn_high_init],
    ],
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    peak_components_df_full.index,
    peak_components_df_full['1635_Peak_Mean'],
    "k",
    label="1635_Peak_Mean",
)
ax.plot(
    peak_components_df_full.index,
    peak_components_df_full["1635_Peak_PC1"],
    label="1635_Peak_PC1",
)
ax.plot(
    peak_components_df_full.index,
    peak_components_df_full["1635_Peak_PC2"],
    label="1635_Peak_PC2",
)
ax.set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax.set_ylabel("Absorbance")
ax.set_xlim([wn_stitch_low-15, wn_stitch_high+15])
ax.set_ylim([-0.025, 0.025])

ax.invert_xaxis()
ax.legend(prop={"size": 12})
plt.tight_layout()
plt.show()

# %% 

mean_peak_smooth1 = savgol_filter(peak_components_df_full["1635_Peak_Mean"], 21, 6)
mean_peak_smooth2 = signal.medfilt(mean_peak_smooth1, 41)
mean_peak_smooth_n = mean_peak_smooth1
mean_peak_smooth_n[125:142] = mean_peak_smooth2[125:142]
mean_peak_smooth_n[247:268] = mean_peak_smooth2[247:268]
mean_peak_smooth_f_i = savgol_filter(mean_peak_smooth_n, 21, 15)
mean_peak_smooth_f = signal.medfilt(mean_peak_smooth_f_i, 5)
mean_peak_smooth_f[194:207] = peak_components_df_full["1635_Peak_Mean"].iloc[194:207]

PC1_smooth1 = savgol_filter(peak_components_df_full["1635_Peak_PC1"], 11, 6)
PC1_smooth2 = signal.medfilt(PC1_smooth1, 45)
PC1_smooth_n = PC1_smooth1
PC1_smooth_n[115:127] = PC1_smooth2[115:127]
PC1_smooth_n[260:270] = PC1_smooth2[260:270]
PC1_smooth_f = savgol_filter(PC1_smooth_n, 21, 15)

PC2_smooth1 = savgol_filter(peak_components_df_full["1635_Peak_PC2"], 21, 6)
PC2_smooth2 = signal.medfilt(PC2_smooth1, 41)
PC2_smooth_n = PC2_smooth1
PC2_smooth_n[130:141] = PC2_smooth2[130:141]
PC2_smooth_n[250:260] = PC2_smooth2[250:260]
PC2_smooth_f_i = savgol_filter(PC2_smooth_n, 21, 15)
PC2_smooth_f = signal.medfilt(PC2_smooth_f_i, 5)
PC2_smooth_f[170:178] = peak_components_df_full["1635_Peak_PC2"].iloc[170:178]
PC2_smooth_f[206:214] = peak_components_df_full["1635_Peak_PC2"].iloc[206:214]
PC2_smooth_f[235:245] = peak_components_df_full["1635_Peak_PC2"].iloc[235:245]

peak_components_f = np.array([mean_peak_smooth_f, PC1_smooth_f, PC2_smooth_f])

# %%

peak_components_df_f = pd.DataFrame(
    peak_components_f.T,
    index=wavenumber_full,
    columns=[
        "1635_Peak_Mean",
        "1635_Peak_PC1",
        "1635_Peak_PC2",
    ],
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    peak_components_df_f.index,
    peak_components_df_f['1635_Peak_Mean'],
    "k",
    label="1635_Peak_Mean",
)
ax.plot(
    peak_components_df_f.index,
    peak_components_df_f["1635_Peak_PC1"],
    label="1635_Peak_PC1",
)
ax.plot(
    peak_components_df_f.index,
    peak_components_df_f["1635_Peak_PC2"],
    label="1635_Peak_PC2",
)
ax.set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax.set_ylabel("Absorbance")
ax.set_xlim([wn_stitch_low-15, wn_stitch_high+15])
ax.set_ylim([-0.025, 0.025])

ax.invert_xaxis()
ax.legend(prop={"size": 12})
plt.tight_layout()
plt.show()

# %%

peak_components_array = peak_components_df_f.reset_index().to_numpy()
# np.savez('../src/PyIRoGlass/H2Om1635PC.npz', data=peak_components_array, columns=peak_components_df_f.columns)


# %%
