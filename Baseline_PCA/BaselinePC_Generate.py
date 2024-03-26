# %%

import sys
import numpy as np
import pandas as pd

sys.path.append("../src/")
import PyIRoGlass as pig

from sklearn.decomposition import PCA
from scipy import signal
from scipy import interpolate

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


def Lorentzian(x, center, half_width, amp=1):
    """
    Return a Lorentzian fit used for characterizing peak shapes.

    Parameters:
        x (numeric): The independent variable, usually representing
            frequency or energy.
        center (numeric): The central position of the peak.
        half_width (numeric): The half-width at half-maximum (HWHM)
            of the peak.
        amp (numeric, optional): The amplitude of the peak. Defaults to 1.

    Returns:
        L (numeric): The Lorentzian fit for the given parameters.
    """

    L = amp * (half_width**2 / (half_width**2 + (2 * x - 2 * center) ** 2))

    return L


def linear(x, m):
    """
    Constructs a linear function and centers it by subtracting its mean.

    Parameters:
        x (numeric): The independent variable.
        m (numeric): The slope of the linear function.

    Returns:
        line (numeric): The centered linear function.
    """

    line = x * m
    line = line - np.mean(line)

    return line


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


def interp_smooth(spectrum, wavenumber):
    """
    Interpolates and smooths a given spectrum using a Univariate Spline.

    Parameters:
        spectrum (numeric array): The spectrum to be interpolated and smoothed.
        wavenumber (numeric array): The wavenumber values corresponding to
            the spectrum.

    Returns:
        interp_spectra (numeric array): The interpolated and smoothed spectrum.
    """

    w = np.ones_like(wavenumber)
    w[0] = 100
    w[-1] = 100
    interp = interpolate.UnivariateSpline(
        x=wavenumber, y=spectrum, k=5, s=0.01, ext=0, w=w
    )

    return interp(wavenumber)


def scale_data(spectrum):
    """
    Scales the data matrix by its range and recenters it by subtracting the
    mean spectrum.

    Parameters:
        spectrum (numeric 2D array): The matrix containing spectral data.

    Returns:
        spectrum_scale (numeric 2D array): The scaled and centered data.
        mean_baseline (numeric array): The mean baseline spectrum used for
            centering.
    """

    data_range = spectrum[0, :] - spectrum[-1, :]
    scaled_data = spectrum / data_range

    spectrum = scaled_data - scaled_data[0, :] + 0.5
    mean_baseline = spectrum.mean(axis=1)
    spectrum_scale = spectrum - np.array([mean_baseline]).T

    return spectrum_scale, mean_baseline


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


def smoothing_protocol(data, wavenumber, split_wn=1500):
    """
    Applies a specialized smoothing protocol to spectral data, which involves
    splitting the spectrum into low and high wavenumber sections, smoothing
    each section individually with different parameters, and then stitching
    them back together. This approach allows for tailored smoothing that accounts
    for the different characteristics typically found in the low and high regions
    of a spectrum.

    Parameters:
        data (numeric 2D array): A matrix where each row corresponds to a
            spectrum and each column corresponds to a spectral intensity at a
            specific wavenumber.
        wavenumber (numeric array): An array containing the wavenumbers
            corresponding to the columns in `data`.
        split_wn (int, optional): The wavenumber at which the spectrum should
            be split into low and high wavenumber sections. Defaults to 1500.

    Returns:
        Smoothed_start (numeric 2D array): The smoothed spectral data obtained
            after applying the smoothing protocol. The data is smoothed in two
            sections, low and high wavenumbers, and then stitched back together.
            Additional smoothing is applied to the entire spectrum to ensure
            continuity and smoothness.
    """

    # Cut spectra at point closest to split_wn to make low and high wavenumber selections.
    section_idx = np.where(np.round(wavenumber) == split_wn)[0][0]

    # Smooth low wavenumbers
    smooth_section1 = np.apply_along_axis(
        savgol_filter,
        0,
        data[0 : section_idx + 50, :],
        smooth_width=71,
        poly_order=3,
    )
    diff = smooth_section1 - data[0 : section_idx + 50, :]
    smooth_diff = np.apply_along_axis(
        interp_smooth, 0, diff, wavenumber=wavenumber[0 : section_idx + 50]
    )

    # Smooth high wavenumbers
    smooth_section1 = smooth_section1 - smooth_diff
    smooth_section2 = np.apply_along_axis(
        savgol_filter,
        0,
        data[section_idx - 50 : None, :],
        smooth_width=121,
        poly_order=3,
    )

    # Cut and Stitch Smoothed Sections
    section_1 = smooth_section1[0:-50, :]
    section_2 = smooth_section2[50:None, :]

    offset_sections = section_1[-1] - section_2[0]
    section_1 = section_1 - offset_sections

    smoothed = np.concatenate([section_1, section_2], axis=0)

    w = np.ones_like(wavenumber)
    w[0:5] = 10
    w[-1:-5] = 10

    interp_data = []

    for array in smoothed.T:
        smoothed_function = interpolate.UnivariateSpline(
            x=wavenumber, y=array, k=5, w=w, s=0.005
        )
        smoothed_array = smoothed_function(wavenumber)
        interp_data.append(smoothed_array)

    return smoothed


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

wn_low, wn_high = (1250, 2400)
H2OCO2_bdl = pd.read_csv("H2O_CO2_BDL.csv", index_col="Wavenumber")
H2OCO2_bdl_df = H2OCO2_bdl.loc[wn_low:wn_high]
H2OCO2_bdl_i = H2OCO2_bdl_df.values
wavenumber = H2OCO2_bdl.loc[wn_low:wn_high].index
print(np.shape(H2OCO2_bdl))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, H2OCO2_bdl)
ax[0].set_title("H$\mathregular{_2}$O and CO$\mathregular{_2}$ BDL")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

basic_data_scale, basic_data_mean = basic_scale_data(H2OCO2_bdl_i)
ax[1].plot(wavenumber, basic_data_scale)
ax[1].set_title("H$\mathregular{_2}$O and CO$\mathregular{_2}$ BDL, Rescaled")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
plt.tight_layout()
plt.show()

# %%

wn_low, wn_high = (1250, 2400)
CO2_bdl = pd.read_csv("All_BDL.csv", index_col="Wavenumber")
CO2_bdl_df = CO2_bdl.loc[wn_low:wn_high]
CO2_bdl_i = CO2_bdl_df.values
CO2_bdl_wavenumber = CO2_bdl_df.index
print(np.shape(CO2_bdl))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, CO2_bdl_i)
ax[0].set_title("H$\mathregular{_2}$O or CO$\mathregular{_2}$ BDL")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

basic_H2O_scale, basic_H2O_mean = basic_scale_data(CO2_bdl_i)
ax[1].plot(wavenumber, basic_H2O_scale)
ax[1].set_title("H$\mathregular{_2}$O or CO$\mathregular{_2}$ BDL, Rescaled")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
plt.tight_layout()
plt.show()

# %%

H2OCO2_smooth_i = smoothing_protocol(H2OCO2_bdl_i, wavenumber, split_wn=1500)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, H2OCO2_smooth_i)
ax[0].set_title("H$\mathregular{_2}$O and CO$\mathregular{_2}$ BDL")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

idx = 8
ax[1].plot(wavenumber, H2OCO2_bdl_i[:, idx], label="Initial Spectrum")
ax[1].plot(wavenumber, H2OCO2_smooth_i[:, idx], label="Smoothed Spectrum")
ax[1].set_title("H$\mathregular{_2}$O and CO$\mathregular{_2}$ BDL, Smoothed")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
ax[1].legend(prop={"size": 12})
plt.tight_layout()
plt.show()

# %%

BDL_data_i, Mean_BL_i = scale_data(H2OCO2_smooth_i)
PC_vectors_i, reduced_data_i, variance_norm_i = perform_PCA(BDL_data_i)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, BDL_data_i, label="")
ax[0].set_title("H$\mathregular{_2}$O or CO$\mathregular{_2}$ BDL, Rescaled")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

ax[1].plot(wavenumber, Mean_BL_i, "k", label="Mean Baseline")
ax[1].plot(wavenumber, PC_vectors_i[0], label="PC1")
ax[1].plot(wavenumber, PC_vectors_i[1], label="PC2")
ax[1].plot(wavenumber, PC_vectors_i[2], label="PC3")
ax[1].plot(wavenumber, PC_vectors_i[3], label="PC4")
ax[1].set_title("Initial Principal Components")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
ax[1].legend(prop={"size": 12})

ax[2].plot(variance_norm_i * 100, marker="o", linestyle="None")
ax[2].set_title("Explained Variance by PCA")
ax[2].set_xlabel("Principal Component")
ax[2].set_ylabel(r"% Variance Explained")
plt.tight_layout()
# plt.savefig('PC_init.png')
plt.show()

# %%

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
ax[0].plot(wavenumber, Mean_BL_i, label="Mean Baseline", linewidth=3)
ax[0].plot(wavenumber, Mean_BL_i + PC_vectors_i[0] * 1, label="Mean Baseline+PC1")
ax[0].plot(wavenumber, Mean_BL_i - PC_vectors_i[0] * 1, label="Mean Baseline-PC1")
ax[0].legend(prop={"size": 12})
ax[0].invert_xaxis()
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")

ax[1].plot(wavenumber, Mean_BL_i, label="Mean Baseline", linewidth=3)
ax[1].plot(wavenumber, Mean_BL_i + PC_vectors_i[1] * 1, label="Mean Baseline+PC2")
ax[1].plot(wavenumber, Mean_BL_i - PC_vectors_i[1] * 1, label="Mean Baseline-PC2")
ax[1].legend(prop={"size": 12})
ax[1].invert_xaxis()
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")

ax[2].plot(wavenumber, Mean_BL_i, label="Mean Baseline", linewidth=3)
ax[2].plot(wavenumber, Mean_BL_i + PC_vectors_i[2] * 1, label="Mean Baseline+PC3")
ax[2].plot(wavenumber, Mean_BL_i - PC_vectors_i[2] * 1, label="Mean Baseline-PC3")
ax[2].legend(prop={"size": 12})
ax[2].invert_xaxis()
ax[2].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[2].set_ylabel("Absorbance")

ax[3].plot(wavenumber, Mean_BL_i, label="Mean Baseline", linewidth=3)
ax[3].plot(wavenumber, Mean_BL_i + PC_vectors_i[3] * 1, label="Mean Baseline+PC4")
ax[3].plot(wavenumber, Mean_BL_i - PC_vectors_i[3] * 1, label="Mean Baseline-PC4")
ax[3].legend(prop={"size": 12})
ax[3].invert_xaxis()
ax[3].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[3].set_ylabel("Absorbance")
plt.tight_layout()
# plt.savefig('PC+mean_plot.png')
plt.show()

# %%


def subtract_H2Om(df, wn_cut_low=1521, wn_cut_high=1730, return_DF=True):
    """
    Removes a specified range of data points from a DataFrame, in this case,
    the H2Om, 1635 peak in transmission FTIR data.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing spectral data.
        wn_cut_low (numeric): The lower bound of the wavenumber range.
        wn_cut_high (numeric): The upper bound of the wavenumber range.
        return_DF (bool, optional): Determines the format of the returned data.

    Returns:
        If return_DF is True:
            No_Peaks_Frame (pandas.DataFrame): The input DataFrame with the
                specified wavenumber range removed.
        If return_DF is False:
            No_Peaks_Wn (numpy.ndarray): An array of wavenumbers excluding the
                specified range.
            No_Peaks_Data (numpy.ndarray): An array of data values corresponding
                to the remaining wavenumbers.
    """

    No_Peaks_Frame = df.drop(df.loc[wn_cut_low:wn_cut_high].index)

    if return_DF == True:
        return No_Peaks_Frame

    No_Peaks_Wn = No_Peaks_Frame.index
    No_Peaks_Data = No_Peaks_Frame.values

    return No_Peaks_Wn, No_Peaks_Data


def No_H2O_fit(spectrum, Avg_BL, PC_DF, wavenumber=wavenumber):
    """
    Fits the given spectrum after excluding the water peak, using a baseline
    correction approach that combines the average baseline, principal component
    analysis (PCA) data, and linear functions to model the baseline.

    Parameters:
        spectrum (pandas.Series or numpy.ndarray): The spectral data to be fitted.
        Avg_BL (pandas.Series): The average baseline data.
        PC_DF (pandas.DataFrame): DataFrame containing PCA vectors.
        wavenumber (numpy.ndarray): An array of wavenumbers corresponding to
            the spectral data.

    Returns:
        numpy.matrix: The baseline matrix used for fitting.
        tuple: Fit parameters from the least squares fit, including the
            solution and residuals.
    """

    offset = pd.Series(np.ones(len(Avg_BL)), index=wavenumber)
    tilt = pd.Series(np.arange(0, len(Avg_BL)), index=wavenumber)
    Baseline_Matrix = pd.concat(
        [
            Avg_BL,
            PC_DF,
            offset,
            tilt,
        ],
        axis=1,
    )
    Baseline_Matrix = np.matrix(Baseline_Matrix)
    fit_param = np.linalg.lstsq(Baseline_Matrix, spectrum, rcond=None)

    return Baseline_Matrix, fit_param


def plot_NoH2O_results(spectrum, Baseline_Matrix, fit_param, wavenumber):
    """
    Plots the spectral data and its corresponding modeled fit after excluding
    the water peak, providing a visual representation of the fit's quality.

    Parameters:
        Spectrum (numpy.ndarray): The original spectral data.
        Baseline_Matrix (numpy.matrix): The baseline matrix used for fitting.
        fit_param (tuple): Fit parameters obtained from the least squares
            fitting process.
        wavenumber (numpy.ndarray): An array of wavenumbers corresponding to
            the spectral data.

    Returns:
        None: This function plots the results and does not return any value.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(wavenumber, spectrum, label="Spectrum")
    plt.plot(
        wavenumber,
        np.matrix(Baseline_Matrix) * np.matrix(fit_param[0]).T,
        label="Modeled Fit",
    )
    ax.set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
    ax.set_ylabel("Absorbance")
    ax.legend(prop={"size": 12})
    ax.invert_xaxis()
    plt.tight_layout()


def Carbonate_baseline_fit(spectrum, Avg_BL, PC_vectors, n_PC_vectors=4):
    """
    Fits a given spectrum to model carbonate baselines using a combination of
    PCA vectors and Gaussian/Lorentzian peaks representing common carbonate
    features.

    Parameters:
        spectrum (numpy.ndarray or pandas.Series): The spectral data to be fitted.
        Avg_BL (pandas.Series): The average baseline data.
        PC_vectors (numpy.ndarray): Array containing PCA vectors.
        n_PC_vectors (int, optional): Number of PCA vectors to use. Default is 4.

    Returns:
        numpy.matrix: The baseline matrix constructed from the input parameters
            and used for fitting.
        tuple: Fit parameters from the least squares fit, including the solution
            and residuals.
    """

    PCA_DF = pd.DataFrame(PC_vectors[0:n_PC_vectors].T, index=wavenumber)
    Peak1 = pd.Series(
        Lorentzian(x=wavenumber, center=1635, half_width=55, amp=1), index=wavenumber
    )
    Peak2 = pd.Series(pig.gauss(x=wavenumber, mu=1430, sd=30, A=1), index=wavenumber)
    Peak3 = pd.Series(pig.gauss(x=wavenumber, mu=1515, sd=30, A=1), index=wavenumber)
    offset = pd.Series(np.ones(len(Peak2)), index=wavenumber)
    tilt = pd.Series(np.arange(0, len(Peak2)), index=wavenumber)

    # This line is only used if we are fitting the Water peak with the CO2 peak.
    Baseline_Matrix = pd.concat(
        [Avg_BL, PCA_DF, offset, tilt, Peak2, Peak3, Peak1], axis=1
    )

    Baseline_Matrix = np.matrix(Baseline_Matrix)

    fit_param = np.linalg.lstsq(Baseline_Matrix, spectrum, rcond=None)

    return Baseline_Matrix, fit_param


# %%  Remove the H2Om, 1635 peak from these CO2-BDL spectra
# This line is a comment indicating the start of a new section in the notebook. The section's goal is to remove the water (H2O) peak and another peak at 1635 wavenumbers from the CO2-bound ligand (BDL) spectra.

# Define the lower and upper bounds of the wavenumber range to be cut out, which includes the water peak and the 1635 peak.
wn_cut_low, wn_cut_high = (1521, 1730)

# Apply the subtract_H2Om function to the CO2_bdl_df DataFrame to remove the specified range of peaks. The function returns a new DataFrame (Full_No_Peaks_DF) with the specified wavenumber range removed, thereby excluding the water and 1635 peaks from the CO2-BDL spectra.
CO2_bdl_df_noH2Om = subtract_H2Om(CO2_bdl_df)

# Apply the subtract_H2Om function to a newly created DataFrame from the first four principal component (PC) vectors, transformed to DataFrame and indexed by wavenumber. This operation removes the specified range of peaks from the PCA data, creating PCA_No_Peaks_DF which is PCA data adjusted for the absence of those peaks.
PCA_DF_i = pd.DataFrame(PC_vectors_i[0:4].T, index=wavenumber)
PCA_DF_noH2Om = subtract_H2Om(PCA_DF_i)

# Create a pandas Series (Avg_BL) representing the average baseline, indexed by wavenumber. This average baseline is derived from Mean_BL_i, which is presumably an array or list containing mean baseline values.
Avg_BL = pd.Series(Mean_BL_i, index=wavenumber)

# Apply the subtract_H2Om function to the Avg_BL Series to remove the specified peak range, resulting in a new Series (Avg_BL_nopeaks) that represents the average baseline adjusted for the absence of the water and 1635 peaks.
Avg_BL_noH2Om = subtract_H2Om(Avg_BL)

# %%

# This line calls the Carbonate_baseline_fit function to fit the spectral data to a model that accounts for carbonate features.
# It inputs a specific spectrum (first column of BDL_data_i), an average baseline (Avg_BL), and PCA vectors (PC_vectors_i).
# It returns a matrix representing the full baseline model (Full_Baseline_Matrix) and parameters of the fit (fit_param).
Full_Baseline_Matrix, fit_param = Carbonate_baseline_fit(
    spectrum=BDL_data_i[:, 0],
    Avg_BL=Avg_BL,
    PC_vectors=PC_vectors_i,
)

# Initialize two empty lists to store fitting parameters and baseline data for each spectrum.
Fits = []
Peak_Strip_Baselines = []

# Iterate over each spectrum in Full_No_Peaks_DF (after water peaks have been removed),
# assuming the spectra are stored in columns (transpose of the original DataFrame).
# for spec in CO2_bdl_df_noH2Om.values.T:

for i in range(0, len(CO2_bdl_df_noH2Om.columns)):
    # For each spectrum, call the No_H2O_fit function to adjust the baseline excluding water peaks.
    # It inputs the current spectrum, an average baseline without water peaks (Avg_BL_nopeaks),
    # a PCA DataFrame adjusted for no water peaks (PCA_No_Peaks_DF), and the wavenumber index.
    # It returns a baseline matrix (Baseline_Matrix) and fitting parameters (fit_param) for the spectrum.
    spectrum = CO2_bdl_df_noH2Om.iloc[:, i]
    Baseline_Matrix, fit_param = No_H2O_fit(
        spectrum=spectrum,
        Avg_BL=Avg_BL_noH2Om,
        PC_DF=PCA_DF_noH2Om,
        wavenumber=Avg_BL_noH2Om.index,
    )

    if i % 20 == 0:
        plot_NoH2O_results(
            spectrum=spectrum,
            Baseline_Matrix=Baseline_Matrix,
            fit_param=fit_param,
            wavenumber=Avg_BL_noH2Om.index,
        )
        plt.plot(
            wavenumber, CO2_bdl_df.iloc[:, i], label="CO$\mathregular{_2}$-BDL Spectrum"
        )
        plt.title(CO2_bdl_df_noH2Om.columns[i] + " Peak Stripping")
        plt.legend(prop={"size": 12})

    # Append the fitting parameters for the current spectrum to the Fits list.
    Fits.append(fit_param)

    # Construct the full baseline for the current spectrum by multiplying the Full_Baseline_Matrix (excluding the last 3 columns)
    # with the first fitting parameter (fit_param[0]), then transpose (.T) and convert to a numpy array.
    # The np.squeeze function removes single-dimensional entries from the shape of the array.
    base_full = np.squeeze(
        np.array(Full_Baseline_Matrix[:, 0:-3] * np.matrix(fit_param[0]).T)
    )

    # Append the constructed full baseline for the current spectrum to the No_H2O_baseline list.
    Peak_Strip_Baselines.append(base_full)

# %%

# Create a DataFrame from the No_H2O_baseline
Peak_Strip_DF = pd.DataFrame(
    np.array(Peak_Strip_Baselines).T,
    index=wavenumber,
    columns=CO2_bdl_df.columns,
)

# Cutout between wn_cut_low:wn_cut_high
cutout = Peak_Strip_DF.loc[wn_cut_low:wn_cut_high]
# Apply Savitzky-Golay filter to smooth the data
smooth_cutout = cutout.apply(func=lambda x: savgol_filter(x, 31, 3))

# Calculate offset by finding difference between values above and below cutout
offset_H2O_frame = (
    CO2_bdl_df.loc[wn_cut_high - 1 : wn_cut_high].values
    - CO2_bdl_df.loc[wn_cut_low : wn_cut_low + 1].values
).ravel()
# Calculate offset in smoothed cutout by finding differences above and below cutout
offset_smooth_cutout = (
    smooth_cutout.loc[wn_cut_high - 1 : wn_cut_high].values
    - smooth_cutout.loc[wn_cut_low : wn_cut_low + 1].values
).ravel()
# Scale smoothed cutout data by ratio of original offset
scale_cutout = smooth_cutout * (offset_H2O_frame / offset_smooth_cutout)
# Calculate offset to align cutout with original data
offset_1800 = (
    CO2_bdl_df.loc[wn_cut_high - 1 : wn_cut_high].values
    - scale_cutout.loc[wn_cut_high - 1 : wn_cut_high].values
)
# Apply offset to adjust
cut_adjusted = scale_cutout + offset_1800

# Concatenate original dataset with cutout data
BDL_baselines_i = pd.concat(
    [
        CO2_bdl_df.drop(CO2_bdl_df.loc[wn_cut_low:wn_cut_high].index),
        cut_adjusted,
    ]
)
# Sort dataset for continuous wavenumbers
BDL_baselines_i.sort_index(inplace=True)

# Apply Savitzky-Golay to smooth dataset
BDL_baselines = BDL_baselines_i.apply(func=lambda x: savgol_filter(x, 31, 3))

# %%

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()

ax[0].plot(Peak_Strip_DF)
ax[0].set_title(str(np.shape(Peak_Strip_DF)[1]))
ax[0].set_title("Initial Peak Stripped Baselines")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

ax[1].plot(BDL_baselines_i)
ax[1].set_title(str(np.shape(BDL_baselines_i)[1]))
ax[1].set_title("Intermediate Peak Stripped Baselines")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()

ax[2].plot(BDL_baselines)
ax[2].set_title(str(np.shape(BDL_baselines)[1]))
ax[2].set_title("Final Peak Stripped Baselines")
ax[2].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[2].set_ylabel("Absorbance")
ax[2].invert_xaxis()
plt.tight_layout()
plt.show()

# %% # Calculates the Principle components

Data, Mean_BL = scale_data(BDL_baselines.values)
PC_vectors, reduced_data, variance_norm = perform_PCA(Data)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()
ax[0].plot(wavenumber, Peak_Strip_DF)
ax[0].set_title("Peak Stripped Baselines, Rescaled")
ax[0].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[0].set_ylabel("Absorbance")
ax[0].invert_xaxis()

# ax[1].plot(Wavenumber, Mean_BL, 'k', label="Mean Baseline")
ax[1].plot(wavenumber, PC_vectors[0], label="PC1")
ax[1].plot(wavenumber, PC_vectors[1], label="PC2")
ax[1].plot(wavenumber, PC_vectors[2], label="PC3")
ax[1].plot(wavenumber, PC_vectors[3], label="PC4")
ax[1].set_title("Final Principal Components")
ax[1].set_xlabel("Wavenumber ($\mathregular{cm^{-1}}$)")
ax[1].set_ylabel("Absorbance")
ax[1].invert_xaxis()
ax[1].legend(prop={"size": 12})

ax[2].plot(variance_norm * 100, marker="o", linestyle="None")
ax[2].set_title("Explained Variance by PCA")
ax[2].set_xlabel("Principal Component")
ax[2].set_ylabel(r"% Variance Explained")
plt.tight_layout()
# plt.savefig('PC_init.png')
plt.show()

# %%
# %%
