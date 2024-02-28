# %%

__author__ = "Sarah Shi"

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import mc3

import matplotlib as mpl
from matplotlib import pyplot as plt
import mc3.plots as mp
import mc3.stats as ms
import mc3.utils as mu

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate


# %% Core functions


class SampleDataLoader:

    """
    A class for efficiently loading and managing spectral data and associated
    metadata from CSV files. This includes spectral data from multiple samples,
    as well as related glass chemistry and thickness information. The class
    supports loading spectral data within a specified wavenumber range from a
    directory containing CSV files and loading chemistry and thickness data
    from a CSV file.

    Attributes:
        spectrum_path (list[str]): The default list of paths where the spectral
            data CSV files are located.
        chemistry_thickness_path (str): The default file path to the CSV file
            containing glass chemistry and thickness data.

    Methods:
        load_spectrum_directory(paths, wn_high, wn_low): Loads
            spectral data from CSV files within a specified wavenumber
            range, ensuring that the wavenumbers are in ascending order and
            skipping headers if present.
        load_chemistry_thickness(chemistry_thickness_path): Loads glass
            chemistry and thickness data from a CSV file, setting the 'Sample'
            column as the index.
        load_all_data(paths, chemistry_thickness_path, wn_high, wn_low):
            Loads both spectral data from CSV files and chemistry thickness
            data from a CSV file.

    """

    def __init__(self,
                 spectrum_path=None,
                 chemistry_thickness_path=None,
                 export_path=None):

        """
        Initializes a SampleDataLoader object with optional default paths for
        the directory containing spectra and the chemistry thickness CSV file.
        """

        self.spectrum_path = spectrum_path
        self.chemistry_thickness_path = chemistry_thickness_path
        self.export_path = export_path
        self.initialize_export_paths(export_path)

    def initialize_export_paths(self, export_path):

        """
        Initializes the export paths for various data outputs, such as figures,
        plot files, and final data. This method creates the necessary
        directories if they do not exist. If an export path is provided, it is
        used as the base directory for all exports. Otherwise, a default
        directory named 'export_data' within the current working directory
        is used.
        """

        output_dirs = [
            "FIGURES", "PLOTFILES", "NPZTXTFILES", "LOGFILES",
            "PKLFILES", "BLPEAKFILES", "FINALDATA"
        ]
        add_dirs = ["TRACE", "HISTOGRAM", "PAIRWISE", "MODELFIT"]

        # Default directory for data export if export_path is not provided
        default_export_dir = "Samples" if export_path is None else export_path

        paths = {}
        for dir_name in output_dirs:
            if dir_name == "FINALDATA":
                full_path = os.path.join(os.getcwd(), dir_name)
            else:
                # Other directories nested in 'export_path'/default directory
                full_path = os.path.join(os.getcwd(), dir_name,
                                         default_export_dir)

            os.makedirs(full_path, exist_ok=True)
            paths[dir_name] = full_path

        plotfile_path = paths["PLOTFILES"]
        for add_dir in add_dirs:
            os.makedirs(os.path.join(plotfile_path, add_dir), exist_ok=True)

        # For 'data_export_path', use 'FINALDATA' without nesting further
        self.data_export_path = os.path.join(paths["FINALDATA"],
                                             default_export_dir)

    def load_spectrum_directory(self, wn_high=5500, wn_low=1000):

        if self.spectrum_path is None:
            raise ValueError("Spectrum path is not provided.")

        paths = sorted(glob.glob(os.path.join(self.spectrum_path, "*")))

        dfs = []
        files = []

        for path in paths:
            file = os.path.split(path)[1][0:-4]

            try:
                df = pd.read_csv(
                    path,
                    names=["Wavenumber", "Absorbance"],
                    dtype={"Wavenumber": float},
                )
            except ValueError:
                df = pd.read_csv(
                    path,
                    skiprows=1,
                    names=["Wavenumber", "Absorbance"],
                    dtype={"Wavenumber": float},
                )

            if df["Wavenumber"].iloc[0] > df["Wavenumber"].iloc[-1]:
                df = df.iloc[::-1]
            df.set_index("Wavenumber", inplace=True)
            spectrum = df.loc[wn_low:wn_high]
            dfs.append(spectrum)
            files.append(file)

        self.files = files
        self.dfs_dict = dict(zip(files, dfs))

        return self.files, self.dfs_dict

    def load_chemistry_thickness(self):

        if (self.chemistry_thickness_path is None or
                not os.path.exists(self.chemistry_thickness_path)):
            raise ValueError("Chemistry thickness path not provided or "
                             "does not exist.")

        chem_thickness = pd.read_csv(self.chemistry_thickness_path)
        chem_thickness.set_index("Sample", inplace=True)

        chemistry = chem_thickness.loc[
            :,
            [
                "SiO2",
                "TiO2",
                "Al2O3",
                "Fe2O3",
                "FeO",
                "MnO",
                "MgO",
                "CaO",
                "Na2O",
                "K2O",
                "P2O5",
            ],
        ].fillna(0)
        thickness = chem_thickness.loc[:, ["Thickness", "Sigma_Thickness"]]

        self.chemistry = chemistry
        self.thickness = thickness

        return self.chemistry, self.thickness

    def load_all_data(self, wn_high=5500, wn_low=1000):

        if self.spectrum_path is None:
            raise ValueError("Spectrum path is not provided.")

        if self.chemistry_thickness_path is None:
            raise ValueError("Chemistry thickness path is not provided.")

        files, dfs_dict = self.load_spectrum_directory(wn_high=wn_high,
                                                       wn_low=wn_low)
        chemistry, thickness = self.load_chemistry_thickness()

        self.files = files
        self.dfs_dict = dfs_dict
        self.chemistry = chemistry
        self.thickness = thickness

        return (
            self.files,
            self.dfs_dict,
            self.chemistry,
            self.thickness,
            self.export_path,
            self.data_export_path,
        )


class VectorLoader:

    """
    A class for loading spectral data vectors, including principal components
    and wavenumbers, from specified NPZ files. It preloads vectors from default
    NPZ files upon instantiation.

    Attributes:
        base_path (str): The base directory path where NPZ files are located.
        Baseline_PC_path (str): The path to the NPZ file containing the
            baseline principal components.
        H2Om_PC_path (str): The path to the NPZ file containing the H2Om
            principal components.
        wn_high (int): The upper limit of the wavenumber range to be
            considered.
        wn_low (int): The lower limit of the wavenumber range to be
            considered.
        Wavenumber (np.ndarray): The array of wavenumbers loaded from the
            baseline NPZ file.
        Baseline_PC (np.ndarray): The matrix of baseline principal components.
        H2Om_PC (np.ndarray): The matrix of H2O-modified principal components.

    Methods:
        load_PC(file_name): Loads baseline principal components from NPZ.
        load_wavenumber(file_name): Loads wavenumbers from NPZ.

    """

    def __init__(self):

        self.base_path = os.path.dirname(__file__)
        self.baseline_PC_path = "BaselineAvgPC.npz"
        self.H2Om_PC_path = "H2Om1635PC.npz"
        self.wn_high = 2400
        self.wn_low = 1250

        # Load default files
        self.wavenumber = self.load_wavenumber(self.baseline_PC_path)
        self.baseline_PC = self.load_PC(self.baseline_PC_path)
        self.H2Om_PC = self.load_PC(self.H2Om_PC_path)

    def load_PC(self, file_name):

        file_path = os.path.join(self.base_path, file_name)
        npz = np.load(file_path)
        PC_DF = pd.DataFrame(npz["data"], columns=npz["columns"])
        PC_DF = PC_DF.set_index("Wavenumber")
        PC_DF = PC_DF.loc[self.wn_low: self.wn_high]
        PC_matrix = PC_DF.to_numpy()

        return PC_matrix

    def load_wavenumber(self, file_name):

        file_path = os.path.join(self.base_path, file_name)
        npz = np.load(file_path)
        wavenumber_DF = pd.DataFrame(npz["data"], columns=npz["columns"])
        wavenumber_DF = wavenumber_DF.set_index("Wavenumber")
        wavenumber_DF = wavenumber_DF.loc[self.wn_low: self.wn_high]
        wavenumber = np.array(wavenumber_DF.index)

        return wavenumber


def gauss(x, mu, sd, A=1):

    """
    Return a Gaussian fit for the CO_3^{2-} doublet peaks at 1515 and
    1430 cm^-1 peaks.

    Parameters:
        x (numeric): The wavenumbers of interest.
        mu (numeric): The center of the peak.
        sd (numeric): The standard deviation (or width of peak).
        A (numeric, optional): The amplitude. Defaults to 1.

    Returns:
        G (np.ndarray): The Gaussian fit.

    """

    G = A * np.exp(-((x - mu) ** 2) / (2 * sd**2))

    return G


def linear(x, m, b):

    """
    Calculate a linear offset for adjusting model data.

    Parameters:
        x (np.ndarray): Input values.
        m (float): Tilt of the linear offset.
        b (float): Offset of the linear offset.

    Returns:
        tilt_offset (np.ndarray): Linear offset.

    """

    tilt_offset = m * np.arange(0, x.size) + b

    return tilt_offset


def carbonate(P, x, PCmatrix, H2Om1635_PCmatrix, Nvectors=5):

    """
    The carbonate function takes in the inputs of fitting parameters P,
    wavenumbers x, principal component matrix, number of principal
    component vectors of interest. The function calculates the
    H2Om_{1635} peak, CO_3^{2-} Gaussian peaks, linear
    offset, and baseline. The function then returns the model data.

    Parameters:
        P (np.ndarray): Fitting parameters including principal component
            and peak weights, Gaussian parameters, linear offset slope
            and intercept.
        x (np.ndarray): Wavenumbers of interest.
        PCmatrix (matrix): Principal components matrix.
        H2Om1635_PCmatrix (matrix): Principal components matrix for the
            1635 peak.
        Nvectors (int, optional): Number of principal components vectors
            of interest. Default is 5.

    Returns:
        model_data (np.ndarray): Model data for the carbonate spectra.

    """

    PC_Weights = np.array([P[0:Nvectors]])
    Peak_Weights = np.array([P[-5:-2]])

    (
        peak_G1430,
        std_G1430,
        G1430_amplitude,
        peak_G1515,
        std_G1515,
        G1515_amplitude,
    ) = P[Nvectors:-5]
    m, b = P[-2:None]

    Peak_1635 = Peak_Weights @ H2Om1635_PCmatrix.T
    G1515 = gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = linear(x, m, b)

    baseline = PC_Weights @ PCmatrix.T
    model_data = baseline + linear_offset + Peak_1635 + G1515 + G1430
    model_data = np.array(model_data)[0, :]

    return model_data


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=5e5,
                 max_iters=15, conv_thresh=1e-5, verbose=False):

    """
    Computes the asymmetric least squares baseline.
    http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf

    Parameters:
        intensities (array-like): Input signal to baseline.
        asymmetry_param (float, optional): Asymmetry parameter (p) that
            controls the weights. Defaults to 0.05.
        smoothness_param (float, optional): Smoothness parameter (s) that
            controls the smoothness of the baseline.
        max_iters (int, optional): Maximum number of iterations.
        conv_thresh (float, optional): Convergence threshold. Iteration
            stops when the change in the weights < threshold.
        verbose (bool, optional): If True, prints the iteration number and
            convergence at each iteration.

    Returns:
        z (np.ndarray): Baseline of the input signal.

    """

    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p to be concise.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])

    for i in range(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p * mask + (1 - p) * (~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print(i + 1, conv)
        if conv < conv_thresh:
            break
        w = new_w

    else:
        print("ALS did not converge in %d iterations" % max_iters)

    return z


class WhittakerSmoother(object):

    """
    Implements the Whittaker smoother for smoothing a signal.
    http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf

    Parameters:
        signal (array-like): Input signal to be smoothed.
        smoothness_param (float): Relative importance of smoothness of the
            predicted response.
        deriv_order (int, optional): Order of the derivative of the identity
            matrix.

    Attributes:
        y (array-like): Input signal to be smoothed.
        upper_bands (array-like): Upper triangular bands of the matrix used
            for smoothing.

    """

    def __init__(self, signal, smoothness_param, deriv_order=1):
        self.y = signal
        assert deriv_order > 0, "deriv_order must be an int > 0"
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), "constant")
            for i in range(1, k + 1)])

        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1:] = ds[::-1][: i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):

        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal

        return solveh_banded(foo, w * self.y, overwrite_ab=True,
                             overwrite_b=True)


def NIR_process(data, wn_low, wn_high, peak):

    """
    The NIR_process function processes Near-IR absorbance data and returns
    relevant information, such as the fit absorbance data, kriged data, peak
    height, and signal to noise ratio. The function first filters the data,
    then fits a baseline and subtracts it to determine the peak absorbance.
    Next, the data are kriged to further reduce noise and obtain peak height.
    Finally, the signal to noise ratio is calculated, and a warning is issued
    if the ratio is high. The function is used three times with different
    H2Om wavenumber ranges for uncertainty assessment.

    Parameters:
        data (pd.DataFrame): A DataFrame of absorbance data.
        wn_low (int): The lower bound wavenumber for NIR H2Om or OH.
        wn_high (int): The higher bound wavenumber for NIR H2Om or OH.
        peak (str): The H2Om or OH peak of interest.

    Returns:
        peak_fit (pd.DataFrame): A DataFrame of the absorbance data in
            the region of interest, median filtered data, baseline
            subtracted absorbance, and the subtracted peak.
        krige_out (pd.DataFrame): A DataFrame of kriged data output.
        PH_krige (float): The peak height obtained after kriging.
        STN (float): The signal to noise ratio.


    """

    data_H2O = data.loc[wn_low:wn_high]
    data_output = pd.DataFrame(
        columns=["Absorbance", "Absorbance_Filt",
                 "Baseline_NIR", "Peak_Subtract"],
        index=data_H2O.index,
    )

    data_output["Absorbance"] = data_H2O
    data_output["Absorbance_Filt"] = signal.medfilt(data_H2O["Absorbance"], 5)
    data_output["Baseline_NIR"] = als_baseline(
        data_output["Absorbance_Filt"],
        asymmetry_param=0.001,
        smoothness_param=1e9,
        max_iters=10,
        conv_thresh=1e-5,
    )
    data_output["Peak_Subtract"] = (
        data_output["Absorbance_Filt"] - data_output["Baseline_NIR"]
    )

    krige_wn_range = np.linspace(wn_low - 5, wn_high + 5,
                                 wn_high - wn_low + 11)
    krige_peak = OrdinaryKriging(
        data_H2O.index,
        np.zeros(data_output["Peak_Subtract"].shape),
        data_output["Peak_Subtract"],
        variogram_model="gaussian",
    )
    krige_abs, _ = krige_peak.execute("grid", krige_wn_range,
                                      np.array([0.0]))
    krige_out = pd.DataFrame(
        {
            "Absorbance": krige_abs.squeeze(),
        },
        index=krige_wn_range,
    )

    if peak == "OH":  # 4500 peak
        pr_low, pr_high = 4400, 4600
    elif peak == "H2Om":  # 5200 peak
        pr_low, pr_high = 5100, 5300
    else:
        raise ValueError(f"Invalid peak type: {peak}")

    PH_max = data_output["Peak_Subtract"].loc[pr_low:pr_high].max()
    PH_krige = (
        krige_out["Absorbance"].loc[pr_low:pr_high].max()
        - krige_out["Absorbance"].min()
    )

    PH_krige_index = int(
        data_output["Peak_Subtract"][
            data_output["Peak_Subtract"] == PH_max
        ].index.to_numpy()[0]
    )
    PH_std = (
        data_output["Peak_Subtract"]
        .loc[PH_krige_index - 50: PH_krige_index + 50]
        .std()
    )
    STN = PH_krige / PH_std

    return data_output, krige_out, PH_krige, STN


def MIR_process(data, wn_low, wn_high):

    """
    The MIR_process function processes the Mid-IR H2Ot, 3550 peak and
    returns the fit absorbance data, kriged data, peak height. The
    function first filters the data, then fits a baseline and subtracts
    it to determine the peak absorbance. Next, the data are kriged to
    further reduce noise and obtain peak height. The function is used
    three times with different H2Ot wavenumber ranges for uncertainty
    assessment.

    Parameters:
        data (pd.DataFrame): A DataFrame of absorbance data.
        wn_low (int): The lower bound wavenumber for MIR H2Ot, 3550.
        wn_high (int): The higher bound wavenumber for MIR H2Ot, 3550.

    Returns:
        data_output (pd.DataFrame): A DataFrame of absorbance data,
            median filtered data, baseline subtracted absorbance,
            and the subtracted peak.
        krige_out (pd.DataFrame): A DataFrame of kriged data output.
        PH_krige (float): The peak height obtained after kriging.

    """

    data_H2Ot_3550 = data.loc[wn_low:wn_high]
    data_output = pd.DataFrame(
        columns=["Absorbance", "Baseline_MIR",
                 "Peak_Subtract", "Peak_Subtract_Filt"],
        index=data_H2Ot_3550.index,
    )

    data_output["Absorbance"] = data_H2Ot_3550["Absorbance"]
    data_output["Baseline_MIR"] = als_baseline(
        data_H2Ot_3550["Absorbance"],
        asymmetry_param=0.0010,
        smoothness_param=1e11,
        max_iters=20,
        conv_thresh=1e-7,
    )
    data_output["Peak_Subtract"] = (
        data_H2Ot_3550["Absorbance"] - data_output["Baseline_MIR"]
    )
    data_output["Peak_Subtract_Filt"] = signal.medfilt(
        data_output["Peak_Subtract"], 21)

    peak_wn_low, peak_wn_high = 3300, 3600
    plot_output = data_output.loc[peak_wn_low:peak_wn_high]
    PH_3550 = np.max(plot_output["Peak_Subtract_Filt"])
    plotindex = np.argmax(plot_output["Absorbance"].index.to_numpy() > 3400)

    return data_output, plot_output, PH_3550, plotindex


def MCMC(data, uncert, indparams, log, savefile):

    """
    Runs Monte Carlo-Markov Chain and outputs the best fit parameters and
    standard deviations.

    Parameters:
        data (np.ndarray): 1D array of data points.
        uncert (np.ndarray): 1D array of uncertainties on the data.
        indparams (np.ndarray): 2D array of independent parameters. Each row
            corresponds to one data point and each column corresponds to a
            different independent parameter.
        log (bool, optional): Flag indicating whether to log the output or not.
            Defaults to False.
        savefile (str, optional): File name to save output to. If None, output
            is not saved. Defaults to None.

    Returns:
        mc3_output (mc3.output): The output of the Monte Carlo-Markov Chain.

    """

    # Define initial values, limits, and step sizes for parameters
    func = carbonate

    params = np.array([1.25, 2.00, 0.25, 0.005, 0.001, 1430, 30.0, 0.01,
                       1510, 30.0, 0.01, 0.100, 0.020, 0.01, 5e-4, 0.50])
    pmin = np.array([0.00, -3.000, -2.00, -0.600, -0.300,  1415,  22.5,  0.00,
                     1500,  22.50,  0.00,  0.000, -2.000, -2.00, -5e-1, -1.00])
    pmax = np.array([4.00, 3.00, 2.00, 0.600, 0.300, 1445, 40.0, 3.00,
                     1535, 40.0, 3.00, 3.000, 2.000, 2.00, 5e-1, 3.00])
    pstep = np.abs(pmin - pmax) * 0.01

    # Define prior limits for parameters
    prior = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 30.0, 0.0, 0.0,
                      30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorlow = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 5.00, 0.0, 0.0,
                         5.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorup = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 5.00, 0.0, 0.0,
                        5.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pnames = ['B_mean', 'B_PC1', 'B_PC2', 'B_PC3', 'B_PC4',
              'G1430_peak', 'G1430_std', 'G1430_amp',
              'G1515_peak', 'G1515_std', 'G1515_amp',
              'H1635_mean', 'H1635_PC1', 'H1635_PC2', 'm', 'b']

    texnames = ['$\\overline{B}$', '$\\overline{B}_{PC1}$',
                '$\\overline{B}_{PC2}$', '$\\overline{B}_{PC3}$',
                '$\\overline{B}_{PC4}$',
                '$\\mu_{1430}$', '$\\sigma_{1430}$', '$a_{1430}$',
                '$\\mu_{1515}$', '$\\sigma_{1515}$', '$a_{1515}$',
                '$\\overline{H_{1635}}$',
                '$\\overline{H_{1635}}_{PC1}$',
                '$\\overline{H_{1635}}_{PC2}$', '$m$', '$b$']

    mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
                            indparams=indparams, pmin=pmin, pmax=pmax,
                            pstep=pstep, prior=prior, priorlow=priorlow,
                            priorup=priorup, pnames=pnames, texnames=texnames,
                            sampler='snooker', rms=False, nsamples=1e6,
                            nchains=9, ncpu=4, burnin=2e4, thinning=5,
                            leastsq='trf', chisqscale=False, grtest=True,
                            grbreak=1.01, grnmin=0.5, hsize=10,
                            kickoff='normal', wlike=False, plots=False,
                            log=log, savefile=savefile)

    return mc3_output


def calculate_baselines(dfs_dict, export_path):

    """
    The calculate_baselines function processes a collection of spectral data
    to fit baselines to all H2O and CO2 related peaks in basaltic-andesitic
    spectra. The function inputs the dictionary of DataFrames that were
    created by the SampleDataLoad class and determines best-fit (and
    standard deviations) baselines, peak heights, peak locations, peak
    widths, and principal component vectors used to fit the spectra. These
    values are exported in a csv file and figures are created for each
    individual sample. Optionally, it generates visual representations of
    the spectra and analysis outcomes, alongside saving detailed analysis
    results in various file formats for further review.

    Parameters:
        dfs_dict (dict): A dictionary where keys are file identifiers and
            values are DataFrames containing the spectral data for each
            sample. The spectral data is expected to have columns for
            wavenumbers and absorbance values.
        export_path (str, None): The directory path where the output files
            (CSVs, figures, logs, etc.) should be saved. If None, no
            files will be saved.

    Returns:
        data_output (pd.DataFrame): A DataFrame of absorbance data,
            median filtered data, baseline subtracted absorbance,
            and the subtracted peak.
        failures (list): A list of file identifiers for which the analysis
            failed, possibly due to data issues or processing errors.

    """

    path_beg = os.getcwd() + "/"

    # Load files with PC vectors for the baseline and H2Om, 1635 peak.
    vector_loader = VectorLoader()
    wavenumber = vector_loader.wavenumber
    PCmatrix = vector_loader.baseline_PC
    H2Om1635_PCmatrix = vector_loader.H2Om_PC
    Nvectors = 5
    indparams = [wavenumber, PCmatrix, H2Om1635_PCmatrix, Nvectors]

    # Create DataFrames to store peak height data:
    # P_ = peak_, _BP = best parameter, #_STD = _stdev
    H2O_3550_PH = pd.DataFrame(columns=['PH_3550_M', 'PH_3550_STD',
                                        'H2Ot_3550_MAX',
                                        'BL_H2Ot_3550_MAX',
                                        'H2Ot_3550_SAT'])
    DF_Output = pd.DataFrame(columns=['PH_1635_BP', 'PH_1635_STD',
                                      'PH_1515_BP', 'PH_1515_STD',
                                      'P_1515_BP', 'P_1515_STD',
                                      'STD_1515_BP', 'STD_1515_STD',
                                      'PH_1430_BP', 'PH_1430_STD',
                                      'P_1430_BP', 'P_1430_STD',
                                      'STD_1430_BP', 'STD_1430_STD'])
    PC_Output = pd.DataFrame(columns=['AVG_BL_BP', 'AVG_BL_STD',
                                      'PC1_BP', 'PC1_STD',
                                      'PC2_BP', 'PC2_STD',
                                      'PC3_BP', 'PC3_STD',
                                      'PC4_BP', 'PC4_STD',
                                      'm_BP', 'm_STD', 'b_BP', 'b_STD',
                                      'PH_1635_PC1_BP', 'PH_1635_PC1_STD',
                                      'PH_1635_PC2_BP', 'PH_1635_PC2_STD'])
    NEAR_IR_PH = pd.DataFrame(columns=['PH_5200_M', 'PH_5200_STD',
                                       'PH_4500_M', 'PH_4500_STD',
                                       'STN_P5200', 'ERR_5200',
                                       'STN_P4500', 'ERR_4500'])

    # Initialize lists for failures and errors
    failures = []
    error_3550 = []
    error_4500 = []
    error_5200 = []

    # Determine best-fit baselines for all peaks with ALS (H2Om_{5200},
    # OH_{4500}, H2Ot_{3550}) and PyIRoGlass mc3 (H2Om_{1635}, CO3^{2-})
    for files, data in dfs_dict.items():
        try:
            # Three repeat baselines for the OH_{4500}
            OH_4500_peak_ranges = [(4250, 4675), (4225, 4650), (4275, 4700)]
            OH_4500_results = list(map(lambda peak_range: {
                'peak_fit': (result := NIR_process(data,
                                                   peak_range[0],
                                                   peak_range[1],
                                                   'OH'))[0],
                'peak_krige': result[1],
                'PH_krige': result[2],
                'STN': result[3]
            }, OH_4500_peak_ranges))

            # Three repeat baselines for the H2Om_{5200}
            H2Om_5200_peak_ranges = [(4875, 5400), (4850, 5375), (4900, 5425)]
            H2Om_5200_results = list(map(lambda peak_range: {
                'peak_fit': (result := NIR_process(data,
                                                   peak_range[0],
                                                   peak_range[1],
                                                   'H2Om'))[0],
                'peak_krige': result[1],
                'PH_krige': result[2],
                'STN': result[3]
            }, H2Om_5200_peak_ranges))

            # Kriged peak heights
            PH_4500_krige = [result["PH_krige"] for result in
                             OH_4500_results]
            PH_4500_krige_M, PH_4500_krige_STD = (
                np.mean(PH_4500_krige),
                np.std(PH_4500_krige),
            )
            PH_5200_krige = [result["PH_krige"] for result in
                             H2Om_5200_results]
            PH_5200_krige_M, PH_5200_krige_STD = (
                np.mean(PH_5200_krige),
                np.std(PH_5200_krige),
            )

            # Calculate signal to noise ratio
            STN_4500_M = np.mean([result["STN"] for result in
                                  OH_4500_results])
            STN_5200_M = np.mean([result["STN"] for result in
                                  H2Om_5200_results])

            # Consider strength of signal
            error_4500 = "-" if STN_4500_M >= 4.0 else "*"
            error_5200 = "-" if STN_5200_M >= 4.0 else "*"

            # Save NIR peak heights
            NEAR_IR_PH.loc[files] = pd.Series(
                {
                    "PH_5200_M": PH_5200_krige_M,
                    "PH_5200_STD": PH_5200_krige_STD,
                    "PH_4500_M": PH_4500_krige_M,
                    "PH_4500_STD": PH_4500_krige_STD,
                    "STN_P5200": STN_5200_M,
                    "ERR_5200": error_5200,
                    "STN_P4500": STN_4500_M,
                    "ERR_4500": error_4500,
                }
            )

            # Three repeat baselines for the H2Ot_{3550}
            H2Ot_3550_peak_ranges = [(1900, 4400), (2100, 4200), (2300, 4000)]
            H2Ot_3550_results = list(map(lambda peak_range: {
                'peak_fit': (result := MIR_process(data,
                                                   peak_range[0],
                                                   peak_range[1]))[0],
                'plot_output': result[1],
                'PH': result[2],
            }, H2Ot_3550_peak_ranges))

            PH_3550 = [result["PH"] for result in H2Ot_3550_results]
            PH_3550_M, PH_3550_STD = np.mean(PH_3550), np.std(PH_3550)

            # Determine maximum absorbances for H2Ot_{3550} and H2Om_{1635}
            # to check for saturation
            data_H2Ot_3550_1 = H2Ot_3550_results[0]["peak_fit"]
            MAX_3550_ABS = data_H2Ot_3550_1[data_H2Ot_3550_1.index > 3550][
                "Absorbance"
            ].iloc[0]
            BL_MAX_3550_ABS = data_H2Ot_3550_1[data_H2Ot_3550_1.index > 3550][
                "Baseline_MIR"
            ].iloc[0]

            # Assign * marker to saturated samples H2Ot_{3550} absorbance > 2
            error_3550 = "-" if MAX_3550_ABS < 2 else "*"

            # Save H2Om_{3550} peak heights and saturation information
            H2O_3550_PH.loc[files] = pd.Series(
                {
                    "PH_3550_M": PH_3550_M,
                    "PH_3550_STD": PH_3550_STD,
                    "H2Ot_3550_MAX": MAX_3550_ABS,
                    "BL_H2Ot_3550_MAX": BL_MAX_3550_ABS,
                    "H2Ot_3550_SAT": error_3550,
                }
            )

            # Initialize PyIRoGlass mc3 fit for H2Om_{1635} and CO3^{2-}
            df_length = np.shape(wavenumber)[0]
            CO2_wn_high, CO2_wn_low = 2400, 1250
            spec = data.loc[CO2_wn_low:CO2_wn_high]

            # Interpolate data to wavenumber spacing, to prepare for mc3
            if spec.shape[0] != df_length:
                interp_wn = np.linspace(spec.index[0],
                                        spec.index[-1],
                                        df_length)
                interp_abs = interpolate.interp1d(
                    spec.index, spec['Absorbance'])(interp_wn)
                spec = spec.reindex(index=interp_wn)
                spec["Absorbance"] = interp_abs
                spec_mc3 = spec["Absorbance"].to_numpy()
            elif spec.shape[0] == df_length:
                spec_mc3 = spec["Absorbance"].to_numpy()

            # Set uncertainty
            uncert = np.ones_like(spec_mc3) * 0.01

            # Run PyIRoGlass mc3!!!
            if export_path is not None:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # Create output directories for resulting files
                output_dirs = [
                    "FIGURES",
                    "PLOTFILES",
                    "NPZTXTFILES",
                    "LOGFILES",
                    "BLPEAKFILES",
                    "PKLFILES",
                ]
                paths = {}
                for dir_name in output_dirs:
                    full_path = os.path.join(path_beg, dir_name, export_path)
                    paths[dir_name] = full_path

                # Create additional directories under 'PLOTFILES'
                plotfile_path = paths["PLOTFILES"]

                # Construct file paths
                fpath = os.path.join(paths["FIGURES"], "")
                ppath = os.path.join(plotfile_path, "")
                sfpath = os.path.join(paths["NPZTXTFILES"], "")
                lpath = os.path.join(paths["LOGFILES"], "")
                pklpath = os.path.join(paths["PKLFILES"], "")

                als_bls = {
                    "OH_4500_results": OH_4500_results,
                    "H2Om_5200_results": H2Om_5200_results,
                    "H2Ot_3550_results": H2Ot_3550_results,
                }

                with open(pklpath + files + ".pkl", "wb") as handle:
                    pickle.dump(als_bls, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

                mc3_output = MCMC(
                    data=spec_mc3,
                    uncert=uncert,
                    indparams=indparams,
                    log=lpath + files + ".log",
                    savefile=sfpath + files + ".npz",
                )

                texnames = [
                    "$\\overline{B}$",
                    "$\\overline{B}_{PC1}$",
                    "$\\overline{B}_{PC2}$",
                    "$\\overline{B}_{PC3}$",
                    "$\\overline{B}_{PC4}$",
                    "$\\mu_{1430}$",
                    "$\\sigma_{1430}$",
                    "$a_{1430}$",
                    "$\\mu_{1515}$",
                    "$\\sigma_{1515}$",
                    "$a_{1515}$",
                    "$\\overline{H_{1635}}$",
                    "$\\overline{H_{1635}}_{PC1}$",
                    "$\\overline{H_{1635}}_{PC2}$",
                    "$m$",
                    "$b$",
                ]

                posterior, _, _ = mu.burn(mc3_output)
                post = mp.Posterior(posterior, texnames)

                post.plot()
                plt.suptitle(files)
                plt.savefig(ppath + "PAIRWISE/" + files + "_pairwise.pdf")
                plt.close("all")

                post.plot_histogram(nx=4, ny=4)
                plt.suptitle(files, y=1.015)
                plt.savefig(ppath + "HISTOGRAM/" + files + "_histogram.pdf",
                            bbox_inches="tight")
                plt.close("all")

                plot_trace(
                    mc3_output["posterior"],
                    title=files,
                    zchain=mc3_output["zchain"],
                    burnin=mc3_output["burnin"],
                    pnames=texnames,
                    savefile=(ppath + "TRACE/" + files + "_trace.pdf"),
                )
                plt.close("all")

                plot_modelfit(
                    spec_mc3,
                    uncert,
                    indparams[0],
                    mc3_output["best_model"],
                    title=files,
                    savefile=(ppath + "MODELFIT/" + files + "_modelfit.pdf"),
                )
                plt.close("all")

            else:
                mc3_output = MCMC(
                    data=spec_mc3,
                    uncert=uncert,
                    indparams=indparams,
                    log=None,
                    savefile=None,
                )

            # Initialize plotting, create subplots of H2Om_{5200} and OH_{4500}
            # baselines and peak fits
            if export_path is not None:
                warnings.filterwarnings("ignore", module="matplotlib\\..*")
                fig = plt.figure(figsize=(26, 8))
                ax1 = plt.subplot2grid((2, 3), (0, 0), fig=fig)
                ax2 = plt.subplot2grid((2, 3), (1, 0), fig=fig)
                ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, fig=fig)
                ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, fig=fig)

                # Create subplot of H2Om_{5200}, OH_{4500} baselines/peak fits
                plot_H2Om_OH(data, files, als_bls, ax_top=ax1, ax_bot=ax2)
                # Create subplot of H2Ot_{3550} baselines/peak fits
                plot_H2Ot_3550(data, files, als_bls, ax=ax3)
                # Create subplot of CO_3^{2-} baselines/peak fits
                plot_carbonate(data, files, mc3_output, ax=ax4)
                plt.tight_layout()
                plt.savefig(fpath + files + ".pdf")
                plt.close("all")

        except Exception as e:
            failures.append(files)
            print(f"{files} failed. Reason: {str(e)}")

        else:
            pass

            # Create DataFrame of best fit parameters and standard deviations
            DF_Output.loc[files] = pd.Series(
                {
                    "PH_1635_BP": mc3_output["bestp"][-5],
                    "PH_1635_STD": mc3_output["stdp"][-5],
                    "PH_1515_BP": mc3_output["bestp"][-6],
                    "PH_1515_STD": mc3_output["stdp"][-6],
                    "P_1515_BP": mc3_output["bestp"][-8],
                    "P_1515_STD": mc3_output["stdp"][-8],
                    "STD_1515_BP": mc3_output["bestp"][-7],
                    "STD_1515_STD": mc3_output["stdp"][-7],
                    "PH_1430_BP": mc3_output["bestp"][-9],
                    "PH_1430_STD": mc3_output["stdp"][-9],
                    "P_1430_BP": mc3_output["bestp"][-11],
                    "P_1430_STD": mc3_output["stdp"][-11],
                    "STD_1430_BP": mc3_output["bestp"][-10],
                    "STD_1430_STD": mc3_output["stdp"][-10],
                }
            )

            PC_Output.loc[files] = pd.Series(
                {
                    "AVG_BL_BP": mc3_output["bestp"][0],
                    "AVG_BL_STD": mc3_output["stdp"][0],
                    "PC1_BP": mc3_output["bestp"][1],
                    "PC1_STD": mc3_output["stdp"][1],
                    "PC2_BP": mc3_output["bestp"][2],
                    "PC2_STD": mc3_output["stdp"][2],
                    "PC3_BP": mc3_output["bestp"][3],
                    "PC3_STD": mc3_output["stdp"][3],
                    "PC4_BP": mc3_output["bestp"][4],
                    "PC4_STD": mc3_output["stdp"][4],
                    "m_BP": mc3_output["bestp"][-2],
                    "m_STD": mc3_output["stdp"][-2],
                    "b_BP": mc3_output["bestp"][-1],
                    "b_STD": mc3_output["stdp"][-1],
                    "PH_1635_PC1_BP": mc3_output["bestp"][-4],
                    "PH_1635_PC1_STD": mc3_output["stdp"][-4],
                    "PH_1635_PC2_BP": mc3_output["bestp"][-3],
                    "PH_1635_PC2_STD": mc3_output["stdp"][-3],
                }
            )

    Volatile_PH = pd.concat([H2O_3550_PH, DF_Output, NEAR_IR_PH, PC_Output],
                            axis=1)

    return Volatile_PH, failures


def beer_lambert(molar_mass, absorbance, density, thickness, epsilon):

    """
    Applies the Beer-Lambert Law to calculate concentration from given inputs.

    Parameters:
        molar_mass (float): The molar mass of the substance in grams per mole.
        absorbance (float): The absorbance of the substance, measured in
            optical density units.
        density (float): The density of the substance in grams per cubic
            centimeter.
        thickness (float): The thickness of the sample in centimeters.
        epsilon (float): The molar extinction coefficient, measured in liters
            per mole per centimeter.

    Returns:
        pd.Series: pandas series containing the concentration in wt.%.

    Notes:
        The Beer-Lambert Law states that the absorbance of a substance is
        proportional to its concentration. The formula for calculating
        concentration from absorbance is:
        concentration = (1e6*molar_mass*absorbance)/(density*thickness*epsilon)

        # https://sites.fas.harvard.edu/~scphys/nsta/error_propagation.pdf

    """

    concentration = pd.Series(dtype='float')
    concentration = ((1e6 * molar_mass * absorbance) /
                     (density * thickness * epsilon))

    return concentration


def beer_lambert_error(N, molar_mass,
                       absorbance, sigma_absorbance,
                       density, sigma_density,
                       thickness, sigma_thickness,
                       epsilon, sigma_epsilon):

    """
    Applies a Monte Carlo simulation to estimate the uncertainty in
    concentration calculated using the Beer-Lambert Law.

    Parameters:
        N (int): The number of Monte Carlo samples to generate.
        molar_mass (float): The molar mass of the substance in grams
            per mole.
        absorbance (float): The absorbance of the substance, measured
            in optical density units.
        sigma_absorbance (float): The uncertainty associated with the
            absorbance measurement.
        density (float): The density of the substance in grams per
            cubic centimeter.
        sigma_density (float): The uncertainty associated with the
            density measurement.
        thickness (float): The thickness of the sample in centimeters.
        sigma_thickness (float): The uncertainty associated with the
            sample thickness measurement.
        epsilon (float): The molar extinction coefficient, measured
            in liters per mole per centimeter.
        sigma_epsilon (float): The uncertainty associated with the molar
            extinction coefficient measurement.

    Returns:
        float: The estimated uncertainty in concentration in wt.%.

    Notes:
        The Beer-Lambert Law states that the absorbance of a substance is
        proportional to its concentration. The formula for calculating
        concentration from absorbance is:
        concentration = (1e6*molar_mass*absorbance)/(density*thickness*epsilon)
        This function estimates the uncertainty in concentration using a
        Monte Carlo simulation. Absorbance, density, and thickness are assumed
        to follow Gaussian distributions with means from the input values
        and standard deviations from the input uncertainties. The absorbance
        coefficient is assumed to follow a uniform distribution with a mean
        given by the input value and a range from the input uncertainty.
        The function returns the standard deviation of the resulting
        concentration distribution.

        # https://astrofrog.github.io/py4sci/_static/Practice%20Problem%20-%20Monte-Carlo%20Error%20Propagation%20-%20Sample%20Solution.html

    """

    gaussian_concentration = (
        1e6 * molar_mass * np.random.normal(absorbance, sigma_absorbance,
                                            N)
    ) / (
        np.random.normal(density, sigma_density, N)
        * np.random.normal(thickness, sigma_thickness, N)
        * np.random.uniform(epsilon - sigma_epsilon, epsilon + sigma_epsilon,
                            N)
    )

    concentration_std = np.std(gaussian_concentration)

    return concentration_std


def calculate_density(composition, T, P, model="LS"):

    """
    The calculate_density function inputs the MI composition file and outputs
    the glass density at the temperature and pressure of analysis. The mole
    fraction is calculated. The total molar volume xivibari is determined from
    sum of the mole fractions of each oxide * partial molar volume at room
    temperature and pressure of analysis. The gram formula weight gfw is then
    determined by summing the mole fractions*molar masses. The density is
    finally determined by dividing gram formula weight by total molar volume.

    Parameters:
        composition (pd.DataFrame): Dataframe containing oxide weight
            percentages for the glass composition.
        T (float): temperature at which the density is calculated (in Celsius)
        P (float): pressure at which the density is calculated (in bars)
        model (str): Choice of density model. 'LS' for Lesher and Spera (2015),
            'IT' for Iacovino and Till (2019). Default is 'LS'.

    Returns:
        mol (pd.DataFrame): DataFrame containing the oxide mole fraction for
            the glass composition
        density (float): glass density at room temperature and pressure
            (in kg/m^3)

    """

    # Define a dictionary of molar masses for each oxide
    molar_mass = {
        "SiO2": 60.08,
        "TiO2": 79.866,
        "Al2O3": 101.96,
        "Fe2O3": 159.69,
        "FeO": 71.844,
        "MnO": 70.9374,
        "MgO": 40.3044,
        "CaO": 56.0774,
        "Na2O": 61.9789,
        "K2O": 94.2,
        "P2O5": 141.9445,
        "H2O": 18.01528,
        "CO2": 44.01,
    }

    # Convert room temperature from Celsius to Kelvin
    T_K = T + 273.15

    if model == "LS":
        # Define a dictionary of partial molar volumes for each oxide, based on
        # data compiled from Lesher and Spera (2015).
        # Tref = 1400C, Pref = 10**-4 GPa (1 bar).
        par_molar_vol = {
            "SiO2": (26.86 - 1.89 * P / 1000),
            "TiO2": (23.16 + 7.24 * (T_K - 1673) / 1000 - 2.31 * P / 1000),
            "Al2O3": (37.42 - 2.26 * P / 1000),
            "Fe2O3": (42.13 + 9.09 * (T_K - 1673) / 1000 - 2.53 * P / 1000),
            "FeO": (13.65 + 2.92 * (T_K - 1673) / 1000 - 0.45 * P / 1000),
            "MgO": (11.69 + 3.27 * (T_K - 1673) / 1000 + 0.27 * P / 1000),
            "CaO": (16.53 + 3.74 * (T_K - 1673) / 1000 + 0.34 * P / 1000),
            "Na2O": (28.88 + 7.68 * (T_K - 1673) / 1000 - 2.4 * P / 1000),
            "K2O": (45.07 + 12.08 * (T_K - 1673) / 1000 - 6.75 * P / 1000),
            "H2O": (26.27 + 9.46 * (T_K - 1673) / 1000 - 3.15 * P / 1000),
        }
    else:
        # data compiled from Iacovino and Till (2019)
        par_molar_vol = {
            "SiO2": (26.86 - 1.89 * P / 1000),
            "TiO2": (28.32 + 7.24 * (T_K - 1773) / 1000 - 2.31 * P / 1000),
            "Al2O3": (37.42 + 2.62 * (T_K - 1773) / 1000 - 2.26 * P / 1000),
            "Fe2O3": (41.5 - 2.53 * P / 1000),
            "FeO": (12.68 + 3.69 * (T_K - 1723) / 1000 - 0.45 * P / 1000),
            "MgO": (12.02 + 3.27 * (T_K - 1773) / 1000 + 0.27 * P / 1000),
            "CaO": (16.9 + 3.74 * (T_K - 1773) / 1000 + 0.34 * P / 1000),
            "Na2O": (29.65 + 7.68 * (T_K - 1773) / 1000 - 2.4 * P / 1000),
            "K2O": (47.28 + 12.08 * (T_K - 1773) / 1000 - 6.75 * P / 1000),
            "H2O": (22.9 + 9.5 * (T_K - 1273) / 1000 - 3.20 * P / 1000),
        }

    # Create an DataFrame to store oxide moles for composition
    mol = pd.DataFrame()
    # Calculate oxide moles by dividing weight percentage by molar mass
    for oxide in composition:
        mol[oxide] = composition[oxide] / molar_mass[oxide]

    # Calculate the total moles for the MI composition
    mol_tot = mol.sum(axis=1)

    # Create empty DataFrames to store the partial molar volume, gram formula
    # weight, oxide density, and the product of mole fraction and partial
    # molar volume for each oxide
    xivbari = pd.DataFrame()
    gfw = pd.DataFrame()

    for oxide in composition:
        # If the oxide is included in the partial molar volume dictionary,
        # calculate its partial molar volume and gram formula weight
        if oxide in par_molar_vol:
            # partial molar volume
            xivbari[oxide] = mol[oxide] / mol_tot * par_molar_vol[oxide]
            # gram formula weight
            gfw[oxide] = mol[oxide] / mol_tot * molar_mass[oxide]

    xivbari_tot = xivbari.sum(axis=1)
    gfw_tot = gfw.sum(axis=1)

    # Calculate density of glass, convert by *1000 for values in kg/m^3
    density = 1000 * gfw_tot / xivbari_tot

    return mol, density


def calculate_epsilon(composition, T, P):

    """
    The calculate_epsilon function computes the extinction coefficients
    and their uncertainties for various molecular species in a given MI
    or glass composition dataset.

    Parameters:
        composition (dictionary): Dictionary containing the weight percentages
            of each oxide in the glass composition
        T (int): temperature at which the density is calculated (in Celsius)
        P (int): pressure at which the density is calculated (in bars)

    Returns:
        mol (pd.DataFrame): Dataframe of the mole fraction of each oxide in
            the glass composition
        density (pd.DataFrame): Dataframe of glass density at room temperature
            and pressure (in kg/m^3)
    """

    epsilon = pd.DataFrame(
        columns=[
            "Tau",
            "Eta",
            "epsilon_H2Ot_3550",
            "sigma_epsilon_H2Ot_3550",
            "epsilon_H2Om_1635",
            "sigma_epsilon_H2Om_1635",
            "epsilon_CO2",
            "sigma_epsilon_CO2",
            "epsilon_H2Om_5200",
            "sigma_epsilon_H2Om_5200",
            "epsilon_OH_4500",
            "sigma_epsilon_OH_4500",
        ]
    )

    mol, _ = calculate_density(composition, T, P)

    # Calculate extinction coefficient
    cation_tot = (mol.sum(axis=1) +
                  mol['Al2O3'] + mol['Na2O'] + mol['K2O'] + mol['P2O5'])
    Na_NaCa = (2 * mol["Na2O"]) / ((2 * mol["Na2O"]) + mol["CaO"])
    SiAl_tot = (mol["SiO2"] + (2 * mol["Al2O3"])) / cation_tot

    # Set up extinction coefficient inversion best-fit parameters and
    # covariance matrices
    mest_3550 = np.array([15.725557, 71.368691])
    mest_1635 = np.array([-50.397564, 124.250534])
    mest_CO2 = np.array([426.66290034, -334.45444392])
    covm_est_3550 = np.diag([38.4640, 77.8597])
    covm_est_1635 = np.diag([20.8503, 39.3875])
    covm_est_CO2 = np.diag([93.85345732, 359.94988573])
    mest_4500 = np.array([-1.632730, 3.532522])
    mest_5200 = np.array([-2.291420, 4.675528])
    covm_est_4500 = np.diag([0.0329, 0.0708])
    covm_est_5200 = np.diag([0.0129, 0.0276])

    # Set up matrices for calculating uncertainties on extinction coefficients
    G_SiAl = np.ones((2, 1))
    G_NaCa = np.ones((2, 1))
    covz_error_SiAl = np.zeros((2, 2))
    covz_error_NaCa = np.zeros((2, 2))

    # Loop through and calculate for all MI or glass compositions.
    for i in composition.index:
        # Calculate extinction coefficients with best-fit parameters
        epsilon_H2Ot_3550 = mest_3550[0] + (mest_3550[1] * SiAl_tot[i])
        epsilon_H2Om_1635 = mest_1635[0] + (mest_1635[1] * SiAl_tot[i])
        epsilon_CO2 = mest_CO2[0] + (mest_CO2[1] * Na_NaCa[i])
        epsilon_H2Om_5200 = mest_5200[0] + (mest_5200[1] * SiAl_tot[i])
        epsilon_OH_4500 = mest_4500[0] + (mest_4500[1] * SiAl_tot[i])

        # Calculate extinction coefficient uncertainties
        G_SiAl[1, 0] = SiAl_tot[i]
        G_NaCa[1, 0] = Na_NaCa[i]
        covz_error_SiAl[1, 1] = SiAl_tot[i] * 0.01  # 1 sigma
        covz_error_NaCa[1, 1] = Na_NaCa[i] * 0.01

        CT_int_3550 = (G_SiAl * covm_est_3550 * np.transpose(G_SiAl)) + (
            mest_3550 * covz_error_SiAl * np.transpose(mest_3550)
        )
        CT68_3550 = (np.mean(np.diag(CT_int_3550))) ** (1 / 2)

        CT_int_1635 = (G_SiAl * covm_est_1635 * np.transpose(G_SiAl)) + (
            mest_1635 * covz_error_SiAl * np.transpose(mest_1635)
        )
        CT68_1635 = (np.mean(np.diag(CT_int_1635))) ** (1 / 2)

        CT_int_CO2 = (G_NaCa * covm_est_CO2 * np.transpose(G_NaCa)) + (
            mest_CO2 * covz_error_NaCa * np.transpose(mest_CO2)
        )
        CT68_CO2 = (np.mean(np.diag(CT_int_CO2))) ** (1 / 2)

        CT_int_5200 = (G_SiAl * covm_est_5200 * np.transpose(G_SiAl)) + (
            mest_5200 * covz_error_SiAl * np.transpose(mest_5200)
        )
        CT68_5200 = (np.mean(np.diag(CT_int_5200))) ** (1 / 2)

        CT_int_4500 = (G_SiAl * covm_est_4500 * np.transpose(G_SiAl)) + (
            mest_4500 * covz_error_SiAl * np.transpose(mest_4500)
        )
        CT68_4500 = (np.mean(np.diag(CT_int_4500))) ** (1 / 2)

        # Save outputs of extinction coefficients to DataFrame epsilon
        epsilon.loc[i] = pd.Series(
            {
                "Tau": SiAl_tot[i],
                "Na/Na+Ca": Na_NaCa[i],
                "epsilon_H2Ot_3550": epsilon_H2Ot_3550,
                "sigma_epsilon_H2Ot_3550": CT68_3550,
                "epsilon_H2Om_1635": epsilon_H2Om_1635,
                "sigma_epsilon_H2Om_1635": CT68_1635,
                "epsilon_CO2": epsilon_CO2,
                "sigma_epsilon_CO2": CT68_CO2,
                "epsilon_H2Om_5200": epsilon_H2Om_5200,
                "sigma_epsilon_H2Om_5200": CT68_5200,
                "epsilon_OH_4500": epsilon_OH_4500,
                "sigma_epsilon_OH_4500": CT68_4500,
            }
        )

    return epsilon


def calculate_concentrations(Volatile_PH, composition, thickness,
                             N=500000, T=25, P=1):

    """
    The calculate_concentrations function calculates the concentrations
    and uncertainties of volatile components (H2O peak (3550 cm^-1),
    molecular H2O peak (1635 cm^-1), and carbonate peaks (1515 and
    1430 cm^-1) in a glass sample based on peak height data, sample
    composition, and wafer thickness. This function uses the Beer-Lambert
    law for absorbance to estimate concentrations and applies Monte Carlo
    simulations to quantify uncertainties. It iteratively adjusts for the
    effect of water content on glass density to improve accuracy.

    Parameters:
        Volatile_PH (pd.DataFrame): DataFrame with columns for peak heights
            of total H2O (3550 cm^-1), molecular H2O (1635 cm^-1), and
            carbonate peaks (1515 and 1430 cm^-1). Each row represents a
            different sample.
        composition (dictionary): Dictionary with keys as oxide names and
            values as their weight percentages in the glass composition.
        thickness (pd.DataFrame): DataFrame with 'Thickness' column
            indicating wafer thickness in micrometers (µm) for each sample.
        N (int): Number of Monte Carlo simulations to perform for uncertainty
            estimation. Default is 500,000.
        T (int): Temperature in Celsius at which the density is calculated.
            Default is 25°C.
        P (int): Pressure in bars at which the density is calculated.
            Default is 1 bar.

    Returns:
        concentrations_df (pd.DataFrame): DataFrame containing calculated
            volatile concentrations and their uncertainties for each sample,
            including columns for mean and standard deviation of H2O and CO2
            species concentrations. ALso contains density ('Density' column)
            and extinction coefficient ('epsilon' column) for each sample,
            providing insight into the properties of the glass under analysis.

    Note:
        The function assumes that the input composition includes all relevant
        oxides and that the Volatile_PH DataFrame contains peak height data for
        all specified peaks. Errors in input data or missing values may affect
        the accuracy of the results.
    """

    # Define a dictionary of molar masses for each oxide
    molar_mass = {
        "SiO2": 60.08,
        "TiO2": 79.866,
        "Al2O3": 101.96,
        "Fe2O3": 159.69,
        "FeO": 71.844,
        "MnO": 70.9374,
        "MgO": 40.3044,
        "CaO": 56.0774,
        "Na2O": 61.9789,
        "K2O": 94.2,
        "P2O5": 141.9445,
        "H2O": 18.01528,
        "CO2": 44.01,
    }

    # Create DataFrames to store volatile data:
    concentrations = pd.DataFrame(
        columns=[
            "H2Ot_MEAN",
            "H2Ot_STD",
            "H2Ot_3550_M",
            "H2Ot_3550_STD",
            "H2Ot_3550_SAT",
            "H2Om_1635_BP",
            "H2Om_1635_STD",
            "CO2_MEAN",
            "CO2_STD",
            "CO2_1515_BP",
            "CO2_1515_STD",
            "CO2_1430_BP",
            "CO2_1430_STD",
            "H2Om_5200_M",
            "H2Om_5200_STD",
            "OH_4500_M",
            "OH_4500_STD",
        ]
    )

    # Dataframe for saturated concentrations
    concentrations_sat = pd.DataFrame(columns=concentrations.columns)

    # Dataframe for storing glass density data
    density_df = pd.DataFrame(columns=["Density"])
    # Dataframe for storing saturated glass density data
    density_sat_df = pd.DataFrame(columns=["Density_Sat"])
    # Dataframe for storing mean volatile data
    mean_vol = pd.DataFrame(columns=["H2Ot_MEAN", "H2Ot_STD",
                                     "CO2_MEAN", "CO2_STD"])
    # Dataframe for storing signal-to-noise error data
    stnerror = pd.DataFrame(
        columns=["PH_5200_STN", "ERR_5200", "PH_4500_STN", "ERR_4500"]
    )

    # Initialize density calculation with 0 wt.% H2O.
    composition["H2O"] = 0
    _, density = calculate_density(composition, T, P)
    epsilon = calculate_epsilon(composition, T, P)

    # Doing density-H2O iterations:
    for jj in range(10):
        H2Ot_3550_I = beer_lambert(
            molar_mass["H2O"],
            Volatile_PH["PH_3550_M"],
            density,
            thickness["Thickness"],
            epsilon["epsilon_H2Ot_3550"],
        )
        composition["H2O"] = H2Ot_3550_I
        _, density = calculate_density(composition, T, P)

    # Doing density-H2O iterations:
    for kk in Volatile_PH.index:
        # Calculate volatile species concentrations
        H2Ot_3550_M = beer_lambert(
            molar_mass["H2O"],
            Volatile_PH["PH_3550_M"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_H2Ot_3550"][kk],
        )
        H2Om_1635_BP = beer_lambert(
            molar_mass["H2O"],
            Volatile_PH["PH_1635_BP"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_H2Om_1635"][kk],
        )
        CO2_1515_BP = beer_lambert(
            molar_mass["CO2"],
            Volatile_PH["PH_1515_BP"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_CO2"][kk],
        )
        CO2_1430_BP = beer_lambert(
            molar_mass["CO2"],
            Volatile_PH["PH_1430_BP"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_CO2"][kk],
        )
        H2Om_5200_M = beer_lambert(
            molar_mass["H2O"],
            Volatile_PH["PH_5200_M"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_H2Om_5200"][kk],
        )
        OH_4500_M = beer_lambert(
            molar_mass["H2O"],
            Volatile_PH["PH_4500_M"][kk],
            density[kk],
            thickness["Thickness"][kk],
            epsilon["epsilon_OH_4500"][kk],
        )
        # Multiply by 1e4 to convert CO2 concentrations to ppm
        CO2_1515_BP *= 10000
        CO2_1430_BP *= 10000

        # Calculate volatile species concentration uncertainties
        H2Ot_3550_M_STD = beer_lambert_error(
            N,
            molar_mass["H2O"],
            Volatile_PH["PH_3550_M"][kk],
            Volatile_PH["PH_3550_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_H2Ot_3550"][kk],
            epsilon["sigma_epsilon_H2Ot_3550"][kk],
        )
        H2Om_1635_BP_STD = beer_lambert_error(
            N,
            molar_mass["H2O"],
            Volatile_PH["PH_1635_BP"][kk],
            Volatile_PH["PH_1635_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_H2Om_1635"][kk],
            epsilon["sigma_epsilon_H2Om_1635"][kk],
        )
        CO2_1515_BP_STD = beer_lambert_error(
            N,
            molar_mass["CO2"],
            Volatile_PH["PH_1515_BP"][kk],
            Volatile_PH["PH_1515_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_CO2"][kk],
            epsilon["sigma_epsilon_CO2"][kk],
        )
        CO2_1430_BP_STD = beer_lambert_error(
            N,
            molar_mass["CO2"],
            Volatile_PH["PH_1430_BP"][kk],
            Volatile_PH["PH_1430_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_CO2"][kk],
            epsilon["sigma_epsilon_CO2"][kk],
        )
        H2Om_5200_M_STD = beer_lambert_error(
            N,
            molar_mass["H2O"],
            Volatile_PH["PH_5200_M"][kk],
            Volatile_PH["PH_5200_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_H2Om_5200"][kk],
            epsilon["sigma_epsilon_H2Om_5200"][kk],
        )
        OH_4500_M_STD = beer_lambert_error(
            N,
            molar_mass["H2O"],
            Volatile_PH["PH_4500_M"][kk],
            Volatile_PH["PH_4500_STD"][kk],
            density[kk],
            density[kk] * 0.025,
            thickness["Thickness"][kk],
            thickness["Sigma_Thickness"][kk],
            epsilon["epsilon_OH_4500"][kk],
            epsilon["sigma_epsilon_OH_4500"][kk],
        )
        # Multiply by 1e4 to convert CO2 uncertainties to ppm
        CO2_1515_BP_STD *= 10000
        CO2_1430_BP_STD *= 10000

        # Save volatile concentrations and uncertainties to DataFrame
        density_df.loc[kk] = pd.Series({"Density": density[kk]})
        concentrations.loc[kk] = pd.Series(
            {
                "H2Ot_3550_M": H2Ot_3550_M,
                "H2Ot_3550_STD": H2Ot_3550_M_STD,
                "H2Ot_3550_SAT": Volatile_PH["H2Ot_3550_SAT"][kk],
                "H2Om_1635_BP": H2Om_1635_BP,
                "H2Om_1635_STD": H2Om_1635_BP_STD,
                "CO2_1515_BP": CO2_1515_BP,
                "CO2_1515_STD": CO2_1515_BP_STD,
                "CO2_1430_BP": CO2_1430_BP,
                "CO2_1430_STD": CO2_1430_BP_STD,
                "H2Om_5200_M": H2Om_5200_M,
                "H2Om_5200_STD": H2Om_5200_M_STD,
                "OH_4500_M": OH_4500_M,
                "OH_4500_STD": OH_4500_M_STD,
            }
        )

    # Loops through for samples to perform two operations depending on
    # saturation. For unsaturated samples, take concentrations and
    # uncertainties from above. For saturated samples, perform the
    # Beer-Lambert calculation again, and have total H2O = H2Om + OH-
    for ll in Volatile_PH.index:
        if Volatile_PH["H2Ot_3550_SAT"][ll] == "-":
            H2Ot_3550_M = concentrations["H2Ot_3550_M"][ll]
            H2Om_1635_BP = concentrations["H2Om_1635_BP"][ll]
            CO2_1515_BP = concentrations["CO2_1515_BP"][ll]
            CO2_1430_BP = concentrations["CO2_1430_BP"][ll]
            H2Om_5200_M = concentrations["H2Om_5200_M"][ll]
            OH_4500_M = concentrations["OH_4500_M"][ll]

            H2Ot_3550_M_STD = concentrations["H2Ot_3550_STD"][ll]
            H2Om_1635_BP_STD = concentrations["H2Om_1635_STD"][ll]
            CO2_1515_BP_STD = concentrations["CO2_1515_STD"][ll]
            CO2_1430_BP_STD = concentrations["CO2_1430_STD"][ll]
            H2Om_5200_M_STD = concentrations["H2Om_5200_STD"][ll]
            OH_4500_M_STD = concentrations["OH_4500_STD"][ll]
            density_sat = density_df["Density"][ll]

        elif Volatile_PH["H2Ot_3550_SAT"][ll] == "*":
            sat_composition = composition.copy()
            for m in range(20):
                H2Om_1635_BP = beer_lambert(
                    molar_mass["H2O"],
                    Volatile_PH["PH_1635_BP"][ll],
                    density[ll],
                    thickness["Thickness"][ll],
                    epsilon["epsilon_H2Om_1635"][ll],
                )
                OH_4500_M = beer_lambert(
                    molar_mass["H2O"],
                    Volatile_PH["PH_4500_M"][ll],
                    density[ll],
                    thickness["Thickness"][ll],
                    epsilon["epsilon_OH_4500"][ll],
                )
                sat_composition.loc[ll, "H2O"] = H2Om_1635_BP + OH_4500_M
                mol_sat, density_sat = calculate_density(sat_composition, T, P)
            density_sat = density_sat[ll]

            H2Ot_3550_M = beer_lambert(
                molar_mass["H2O"],
                Volatile_PH["PH_3550_M"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_H2Ot_3550"][ll],
            )
            H2Om_1635_BP = beer_lambert(
                molar_mass["H2O"],
                Volatile_PH["PH_1635_BP"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_H2Om_1635"][ll],
            )
            CO2_1515_BP = beer_lambert(
                molar_mass["CO2"],
                Volatile_PH["PH_1515_BP"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_CO2"][ll],
            )
            CO2_1430_BP = beer_lambert(
                molar_mass["CO2"],
                Volatile_PH["PH_1430_BP"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_CO2"][ll],
            )
            H2Om_5200_M = beer_lambert(
                molar_mass["H2O"],
                Volatile_PH["PH_5200_M"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_H2Om_5200"][ll],
            )
            OH_4500_M = beer_lambert(
                molar_mass["H2O"],
                Volatile_PH["PH_4500_M"][ll],
                density_sat,
                thickness["Thickness"][ll],
                epsilon["epsilon_OH_4500"][ll],
            )
            CO2_1515_BP *= 10000
            CO2_1430_BP *= 10000

            H2Ot_3550_M_STD = beer_lambert_error(
                N,
                molar_mass["H2O"],
                Volatile_PH["PH_3550_M"][ll],
                Volatile_PH["PH_3550_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_H2Ot_3550"][ll],
                epsilon["sigma_epsilon_H2Ot_3550"][ll],
            )
            H2Om_1635_BP_STD = beer_lambert_error(
                N,
                molar_mass["H2O"],
                Volatile_PH["PH_1635_BP"][ll],
                Volatile_PH["PH_1635_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_H2Om_1635"][ll],
                epsilon["sigma_epsilon_H2Om_1635"][ll],
            )
            CO2_1515_BP_STD = beer_lambert_error(
                N,
                molar_mass["CO2"],
                Volatile_PH["PH_1515_BP"][ll],
                Volatile_PH["PH_1515_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_CO2"][ll],
                epsilon["sigma_epsilon_CO2"][ll],
            )
            CO2_1430_BP_STD = beer_lambert_error(
                N,
                molar_mass["CO2"],
                Volatile_PH["PH_1430_BP"][ll],
                Volatile_PH["PH_1430_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_CO2"][ll],
                epsilon["sigma_epsilon_CO2"][ll],
            )
            H2Om_5200_M_STD = beer_lambert_error(
                N,
                molar_mass["H2O"],
                Volatile_PH["PH_5200_M"][ll],
                Volatile_PH["PH_5200_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_H2Om_5200"][ll],
                epsilon["sigma_epsilon_H2Om_5200"][ll],
            )
            OH_4500_M_STD = beer_lambert_error(
                N,
                molar_mass["H2O"],
                Volatile_PH["PH_4500_M"][ll],
                Volatile_PH["PH_4500_STD"][ll],
                density_sat,
                density_sat * 0.025,
                thickness["Thickness"][ll],
                thickness["Sigma_Thickness"][ll],
                epsilon["epsilon_OH_4500"][ll],
                epsilon["sigma_epsilon_OH_4500"][ll],
            )
            CO2_1515_BP_STD *= 10000
            CO2_1430_BP_STD *= 10000

        density_sat_df.loc[ll] = pd.Series({"Density_Sat": density_sat})
        concentrations_sat.loc[ll] = pd.Series(
            {
                "H2Ot_3550_M": H2Ot_3550_M,
                "H2Ot_3550_SAT": Volatile_PH["H2Ot_3550_SAT"][ll],
                "H2Ot_3550_STD": H2Ot_3550_M_STD,
                "H2Om_1635_BP": H2Om_1635_BP,
                "H2Om_1635_STD": H2Om_1635_BP_STD,
                "CO2_1515_BP": CO2_1515_BP,
                "CO2_1515_STD": CO2_1515_BP_STD,
                "CO2_1430_BP": CO2_1430_BP,
                "CO2_1430_STD": CO2_1430_BP_STD,
                "H2Om_5200_M": H2Om_5200_M,
                "H2Om_5200_STD": H2Om_5200_M_STD,
                "OH_4500_M": OH_4500_M,
                "OH_4500_STD": OH_4500_M_STD,
            }
        )
        stnerror.loc[ll] = pd.Series(
            {
                "PH_5200_STN": Volatile_PH["STN_P5200"][ll],
                "PH_4500_STN": Volatile_PH["STN_P4500"][ll],
                "ERR_5200": Volatile_PH["ERR_5200"][ll],
                "ERR_4500": Volatile_PH["ERR_4500"][ll],
            }
        )

    # Create final spreadsheet
    concentrations_df = pd.concat(
        [concentrations_sat, stnerror,
         density_df, density_sat_df, epsilon],
        axis=1
    )

    # Output different values depending on saturation.
    for m in concentrations.index:
        if concentrations["H2Ot_3550_SAT"][m] == "*":
            H2O_mean = (concentrations["H2Om_1635_BP"][m] +
                        concentrations["OH_4500_M"][m])
            H2O_std = (
                (concentrations["H2Om_1635_STD"][m] ** 2)
                + (concentrations["OH_4500_STD"][m] ** 2)
            ) ** (1 / 2) / 2

        elif concentrations["H2Ot_3550_SAT"][m] == "-":
            H2O_mean = concentrations["H2Ot_3550_M"][m]
            H2O_std = concentrations["H2Ot_3550_STD"][m]
        mean_vol.loc[m] = pd.Series({"H2Ot_MEAN": H2O_mean,
                                     "H2Ot_STD": H2O_std})
    mean_vol["CO2_MEAN"] = (concentrations["CO2_1515_BP"] +
                            concentrations["CO2_1430_BP"]) / 2
    mean_vol["CO2_STD"] = (
        (concentrations["CO2_1515_STD"] ** 2) +
        (concentrations["CO2_1430_STD"] ** 2)
    ) ** (1 / 2) / 2

    concentrations_df["H2Ot_MEAN"] = mean_vol["H2Ot_MEAN"]
    concentrations_df["H2Ot_STD"] = mean_vol["H2Ot_STD"]
    concentrations_df["CO2_MEAN"] = mean_vol["CO2_MEAN"]
    concentrations_df["CO2_STD"] = mean_vol["CO2_STD"]

    return concentrations_df


# %% Plotting Functions


def plot_H2Om_OH(data, files, als_bls, ax_top=None, ax_bot=None):

    """
    Visualizes NIR spectrum data along with baseline-subtracted and kriged
    peak fits for H2Om and OH components in spectral data. The function plots
    the original NIR spectrum and overlays the median-filtered peak fits and
    baseline fits for H2Om (5200 cm^-1) and OH- (4500 cm^-1) peaks on the top
    axis. The bottom axis displays the baseline-subtracted peaks and the
    kriged peak fits, emphasizing the isolated peak shapes and their
    respective signal-to-noise ratios.

    Parameters:
        data (pd.DataFrame): DataFrame containing spectral data with
            'Absorbance' values indexed by wavenumber.
        files (str): The identifier for the current sample set being
            processed, used for titling the plot.
        als_bls (dict): A dictionary containing results from baseline
            fitting and peak analysis. Expected keys include
            'H2Om_5200_results' and 'OH_4500_results', each storing a
            list of result dictionaries from the ALS (Asymmetric Least
            Squares) processing for respective peaks.
        ax_top (matplotlib.axes.Axes, optional): The top subplot axis
            for plotting the NIR spectrum and peak fits. If not provided,
            a new figure with subplots will be created.
        ax_bot (matplotlib.axes.Axes, optional): The bottom subplot
            axis for plotting baseline-subtracted peaks and kriged peak
            fits. If not provided, it will be created along with `ax_top`
            in a new figure.

    Returns:
        None: This function does not return any value. It generates a plot
        visualizing the spectral data, baseline fits, and peak fits for the
        provided sample.

    Note:
        The function is designed to work within a larger analytical framework,
        where spectral preprocessing and peak fitting have been previously
        conducted. It expects specific data structures from these analyses as
        input. The function modifies the provided `ax_top` and `ax_bot`
        axes in place if they are provided; otherwise, it creates a new
        figure and axes for plotting.

    """

    if ax_top is None or ax_bot is None:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8))
        ax_top.set_title(files)

    H2Om_5200_results = als_bls["H2Om_5200_results"]
    OH_4500_results = als_bls["OH_4500_results"]

    # Calculate peak heights
    PH_4500_krige = [result["PH_krige"] for result in OH_4500_results]
    PH_4500_krige_M, PH_4500_krige_STD = (np.mean(PH_4500_krige),
                                          np.std(PH_4500_krige))
    PH_5200_krige = [result["PH_krige"] for result in H2Om_5200_results]
    PH_5200_krige_M, PH_5200_krige_STD = (np.mean(PH_5200_krige),
                                          np.std(PH_5200_krige))

    # Calculate signal to noise ratio
    STN_4500_M = np.mean([result["STN"] for result in OH_4500_results])
    STN_5200_M = np.mean([result["STN"] for result in H2Om_5200_results])

    warnings.filterwarnings("ignore", module="matplotlib\\..*")
    warnings.filterwarnings("ignore", category=UserWarning)

    ax_top.plot(data.index, data["Absorbance"], "k",
                linewidth=1.5, label="NIR Spectrum")

    for result in H2Om_5200_results:
        ax_top.plot(result["peak_fit"].index,
                    result["peak_fit"]["Absorbance_Filt"],
                    "tab:blue",
                    label=(r"$\mathregular{H_2O_{m, 5200}}$ Median Filtered" if
                           result is H2Om_5200_results[0] else "_"))
    for result in OH_4500_results:
        ax_top.plot(result["peak_fit"].index,
                    result["peak_fit"]["Absorbance_Filt"],
                    "tab:orange",
                    label=(r"$\mathregular{OH^{-}_{4500}}$ Median Filtered" if
                           result is OH_4500_results[0] else "_"))
    for result in H2Om_5200_results:
        ax_top.plot(result["peak_fit"].index,
                    result["peak_fit"]["Baseline_NIR"],
                    "lightsteelblue",
                    label=r"$\mathregular{H_2O_{m, 5200}}$ Baseline" if
                    result is H2Om_5200_results[0] else "_")
    for result in OH_4500_results:
        ax_top.plot(result["peak_fit"].index,
                    result["peak_fit"]["Baseline_NIR"],
                    "bisque",
                    label=r"$\mathregular{OH^{-}_{4500}}$ Baseline" if
                    result is OH_4500_results[0] else "_")

    handles_top, labels_top = ax_top.get_legend_handles_labels()
    filtered_handles_top = [h_t for h_t, l_t in
                            zip(handles_top, labels_top)
                            if not l_t.startswith('_')]
    filtered_labels_top = [l_t for l_t in labels_top if not
                           l_t.startswith('_')]
    ax_top.legend(filtered_handles_top, filtered_labels_top, prop={"size": 10})
    ax_top.annotate(
        r"$\mathregular{H_2O_{m, 5200}}$ Peak Height: "
        + f"{PH_5200_krige_M:.4f} "
        + r"± "
        + f"{PH_5200_krige_STD:.4f}, S2N={STN_5200_M:.2f}",
        (0.025, 0.9),
        xycoords="axes fraction",
    )
    ax_top.annotate(
        r"$\mathregular{OH^{-}_{4500}}$ Peak Height: "
        + f"{PH_4500_krige_M:.4f} "
        + r"± "
        + f"{PH_4500_krige_STD:.4f}, S2N={STN_4500_M:.2f}",
        (0.025, 0.8),
        xycoords="axes fraction",
    )

    plotmin = np.round(np.min(data.loc[4250:5400]["Absorbance"]), decimals=1)
    plotmax = np.round(np.max(data.loc[4250:5400]["Absorbance"]), decimals=1)
    ax_top.set_xlim([4200, 5400])
    ax_top.set_ylim([plotmin - 0.075, plotmax + 0.075])
    ax_top.tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax_top.tick_params(axis="y", direction="in", length=5, pad=6.5)
    ax_top.invert_xaxis()

    warnings.filterwarnings("ignore", module="matplotlib\\..*")
    warnings.filterwarnings("ignore", category=UserWarning)
    for result in H2Om_5200_results:
        baseline_subtracted = (result["peak_fit"]["Peak_Subtract"] -
                               np.min(result["peak_krige"]["Absorbance"]))
        ax_bot.plot(result["peak_fit"].index, baseline_subtracted, "k",
                    label=(r"$\mathregular{H_2O_{m,5200}}$ Baseline Subtracted"
                           if result is H2Om_5200_results[0] else "_"))

    for result in OH_4500_results:
        baseline_subtracted = (result["peak_fit"]["Peak_Subtract"] -
                               np.min(result["peak_krige"]["Absorbance"]))
        ax_bot.plot(result["peak_fit"].index, baseline_subtracted, "k",
                    label=(r"$\mathregular{OH^{-}_{4500}}$ Baseline Subtracted"
                           if result is OH_4500_results[0] else "_"))

    for result in H2Om_5200_results:
        kriged_peak = (result["peak_krige"]["Absorbance"] -
                       np.min(result["peak_krige"]["Absorbance"]))
        ax_bot.plot(result["peak_krige"].index, kriged_peak, "tab:blue",
                    label=(r"$\mathregular{H_2O_{m,5200}}$ Kriged Peak"
                           if result is H2Om_5200_results[0] else "_"))

    for result in OH_4500_results:
        kriged_peak = (result["peak_krige"]["Absorbance"] -
                       np.min(result["peak_krige"]["Absorbance"]))
        ax_bot.plot(result["peak_krige"].index, kriged_peak, "tab:orange",
                    label=(r"$\mathregular{OH^{-}_{4500}}$ Kriged Peak"
                           if result is OH_4500_results[0] else "_"))

    # Handling legend entries to avoid duplicates
    handles_bottom, labels_bottom = ax_bot.get_legend_handles_labels()
    filtered_handles_bottom = [h_b for h_b, l_b in
                               zip(handles_bottom, labels_bottom)
                               if not l_b.startswith('_')]
    filtered_labels_bottom = [l_b for l_b in labels_bottom if not
                              l_b.startswith('_')]
    ax_bot.legend(filtered_handles_bottom, filtered_labels_bottom,
                  prop={"size": 10})

    ax_bot.set_xlim([4200, 5400])
    plotmax = np.round(
        np.max(OH_4500_results[0]["peak_fit"]["Peak_Subtract"]), decimals=1
    )
    ax_bot.set_ylim([0, plotmax + 0.05])
    ax_bot.tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax_bot.tick_params(axis="y", direction="in", length=5, pad=6.5)
    ax_bot.invert_xaxis()


def plot_H2Ot_3550(data, files, als_bls, ax=None):

    """
    Plots Mid Infrared (MIR) spectral data along with the baseline and
    filtered peak for the total water (H2Ot) peak at 3550 cm^-1. This
    function visualizes the original MIR spectrum, the baseline fit, and
    the baseline-subtracted and filtered peak for H2Ot, highlighting the
    isolated peak shape and its characteristics.

    Parameters:
        data (pd.DataFrame): DataFrame containing spectral data with
            'Absorbance' values indexed by wavenumber. Expected to cover
            the MIR range relevant for the H2Ot peak analysis.
        files (str): Identifier for the current sample set being processed,
            used for titling the plot.
        als_bls (dict): Dictionary containing results from baseline fitting
            and peak analysis. Expected to have a key 'H2Ot_3550_results'
            which stores a list of dictionaries with results from the
            processing for the H2Ot peak.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis object where
            the plot will be drawn. If None, a new figure and axis will be
            created.

    Returns:
        None: This function does not return any value. It generates a plot
            visualizing the MIR spectral data, baseline fit, and filtered
            peak fit for the H2Ot peak at 3550 cm^-1.

    Note:
        The function is designed to work as part of a larger spectroscopic
        analysis workflow, where spectral data preprocessing, baseline fitting,
        and peak analysis have been previously conducted. It expects specific
        data structures from these analyses as input. If 'ax' is not provided,
        the function creates a new figure and axis for plotting, which might
        not be ideal for integrating this plot into multi-panel figures or
        more complex visual layouts.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    H2Ot_3550_results = als_bls["H2Ot_3550_results"]

    # Calculate peak heights
    PH_3550 = [result["PH"] for result in H2Ot_3550_results]
    PH_3550_M, PH_3550_STD = np.mean(PH_3550), np.std(PH_3550)

    ax.plot(data.index, data["Absorbance"], "k")
    ax.plot(
        H2Ot_3550_results[0]["peak_fit"]["Absorbance"].index,
        H2Ot_3550_results[0]["peak_fit"]["Baseline_MIR"],
        "silver",
        label=r"$\mathregular{H_2O_{t, 3550}}$ Baseline",
    )
    ax.plot(
        H2Ot_3550_results[1]["peak_fit"]["Absorbance"].index,
        H2Ot_3550_results[1]["peak_fit"]["Baseline_MIR"],
        "silver",
    )
    ax.plot(
        H2Ot_3550_results[2]["peak_fit"]["Absorbance"].index,
        H2Ot_3550_results[2]["peak_fit"]["Baseline_MIR"],
        "silver",
    )
    ax.plot(
        H2Ot_3550_results[0]["plot_output"].index,
        (
            H2Ot_3550_results[0]["plot_output"]["Peak_Subtract_Filt"]
            + H2Ot_3550_results[0]["plot_output"]["Baseline_MIR"]
        ),
        "r",
        linewidth=2,
    )
    ax.plot(
        H2Ot_3550_results[1]["plot_output"].index,
        (
            H2Ot_3550_results[1]["plot_output"]["Peak_Subtract_Filt"]
            + H2Ot_3550_results[1]["plot_output"]["Baseline_MIR"]
        ),
        "r",
        linewidth=2,
    )
    ax.plot(
        H2Ot_3550_results[2]["plot_output"].index,
        (
            H2Ot_3550_results[2]["plot_output"]["Peak_Subtract_Filt"]
            + H2Ot_3550_results[2]["plot_output"]["Baseline_MIR"]
        ),
        "r",
        linewidth=2,
    )
    ax.set_title(files)
    ax.annotate(
        r"$\mathregular{H_2O_{t, 3550}}$  Peak Height: "
        + f"{PH_3550_M:.4f} ± {PH_3550_STD:.4f}",
        (0.025, 0.95),
        xycoords="axes fraction",
    )
    ax.set_xlabel(r"Wavenumber $(\mathregular{cm^{-1}})$")
    ax.set_xlim([1250, 4000])
    ax.set_ylabel("Absorbance")
    plotmax = np.round(np.max(data.loc[1250:4000]["Absorbance"].to_numpy()),
                       decimals=0)
    plotmin = np.round(np.min(data.loc[1250:4000]["Absorbance"].to_numpy()),
                       decimals=0)
    ax.set_ylim([plotmin - 0.25, plotmax + 0.5])
    ax.tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax.tick_params(axis="y", direction="in", length=5, pad=6.5)
    ax.legend(loc="upper right", prop={"size": 10})
    ax.invert_xaxis()


def derive_carbonate(data, files, mc3_output, export_path=None):

    """
    Derives and saves carbonate region baseline and peak fits from FTIR
    spectral data using the PyIRoGlass model outputs. This function
    interpolates the spectral data to match the wavenumber spacing
    required by the model, calculates the baseline and peak fits for
    carbonate at 1430 and 1515 cm^-1, and water at 1635 cm^-1. It also
    generates an ensemble of baseline fits from the posterior distributions
    to estimate uncertainties. The results, including best fits and
    baselines, can optionally be saved to CSV files.

    Parameters:
        data (pd.DataFrame): DataFrame containing FTIR spectral data with
            'Absorbance' values indexed by wavenumber.
        file (str): Spectrum sample name.
        mc3_output (dict): Dictionary containing results from PyIRoGlass
            fitting, including best fit parameters, posterior distributions,
            and other model outputs relevant for calculating peak fits.
        savefile (bool): If True, the function will save the best fits and
            baselines data to CSV files named according to the 'file'
            parameter. If False, no files will be saved.

    Returns:
        bestfits (pd.DataFrame): DataFrame containing the original spectral
            data, the complete spectrum fit from PyIRoGlass, individual peak
            fits for carbonate and water, and the calculated baseline.
            Indexed by wavenumber.
        baselines (pd.DataFrame): DataFrame containing an ensemble of
            baseline fits derived from the posterior distribution to represent
            uncertainty in the baseline estimation. Indexed by wavenumber.

    """

    vector_loader = VectorLoader()
    wavenumber = vector_loader.wavenumber
    PCmatrix = vector_loader.baseline_PC
    H2Om1635_PCmatrix = vector_loader.H2Om_PC
    Nvectors = 5

    df_length = np.shape(wavenumber)[0]
    CO2_wn_high, CO2_wn_low = 2400, 1250
    spec = data.loc[CO2_wn_low:CO2_wn_high]

    # Interpolate data to wavenumber spacing, to prepare for mc3
    if spec.shape[0] != df_length:
        interp_wn = np.linspace(spec.index[0],
                                spec.index[-1],
                                df_length)
        interp_abs = interpolate.interp1d(
            spec.index, spec['Absorbance'])(interp_wn)
        spec = spec.reindex(index=interp_wn)
        spec["Absorbance"] = interp_abs
        spec_mc3 = spec["Absorbance"].to_numpy()
    elif spec.shape[0] == df_length:
        spec_mc3 = spec["Absorbance"].to_numpy()

    Spectrum_Solve_BP = carbonate(
        mc3_output["meanp"], wavenumber, PCmatrix, H2Om1635_PCmatrix, Nvectors
    )
    Baseline_Solve_BP = (
        np.asarray(mc3_output["bestp"][0:Nvectors] @ PCmatrix.T).ravel() +
        linear(wavenumber, mc3_output["bestp"][14], mc3_output["bestp"][15])
        )

    H1635_BP = np.asarray(mc3_output["bestp"][11:14] @
                          H2Om1635_PCmatrix.T).ravel()
    CO2P1430_BP = gauss(
        wavenumber,
        mc3_output["bestp"][5],
        mc3_output["bestp"][6],
        A=mc3_output["bestp"][7],
    )
    CO2P1515_BP = gauss(
        wavenumber,
        mc3_output["bestp"][8],
        mc3_output["bestp"][9],
        A=mc3_output["bestp"][10],
    )
    H1635_SOLVE = H1635_BP + Baseline_Solve_BP
    CO2P1515_SOLVE = CO2P1515_BP + Baseline_Solve_BP
    CO2P1430_SOLVE = CO2P1430_BP + Baseline_Solve_BP

    posterior = mc3_output["posterior"]
    mask = mc3_output["zmask"]
    masked_posterior = posterior[mask]
    samplingerror = masked_posterior[:, 0:5]
    samplingerror = samplingerror[
        0: np.shape(masked_posterior[:, :])[0]: int(
            np.shape(masked_posterior[:, :])[0] / 100
        ),
        :,
    ]
    lineerror = masked_posterior[:, -2:None]
    lineerror = lineerror[
        0: np.shape(masked_posterior[:, :])[0]: int(
            np.shape(masked_posterior[:, :])[0] / 100
        ),
        :,
    ]
    Baseline_Array = np.array(samplingerror @ PCmatrix[:, :].T)

    for i in range(np.shape(Baseline_Array)[0]):
        linearray = linear(wavenumber, lineerror[i, 0], lineerror[i, 1])
        Baseline_Array[i, :] += linearray

    best_fits_data = {
        "Wavenumber": wavenumber,
        "Spectrum": spec_mc3,
        "Spectrum_Fit": Spectrum_Solve_BP,
        "Baseline": Baseline_Solve_BP,
        "H2Om_1635": H1635_SOLVE,
        "CO2_1515": CO2P1515_SOLVE,
        "CO2_1430": CO2P1430_SOLVE,
    }
    bestfits = pd.DataFrame(best_fits_data, index=wavenumber)

    baselines = pd.DataFrame(
        Baseline_Array.T,
        index=wavenumber,
        columns=[f"Baseline_{i}" for i in range(Baseline_Array.shape[0])],
    )

    if export_path is not None:
        path_beg = os.getcwd() + "/"
        output_dirs = ["BLPEAKFILES"]
        paths = {}
        for dir_name in output_dirs:
            full_path = os.path.join(path_beg, dir_name, export_path)
            paths[dir_name] = full_path

        # Construct file paths
        bppath = os.path.join(paths["BLPEAKFILES"], "")
        bestfits.to_csv(bppath + files + "_bestfits.csv")
        baselines.to_csv(bppath + files + "_baselines.csv")

    return bestfits, baselines


def plot_carbonate(data, files, mc3_output, ax=None):

    """
    Plots the FTIR spectrum along with baseline fits, and model fits for
    carbonate and water peaks. This function visualizes the original FTIR
    spectrum, the baseline determined by PCA, and model fits for CO_3^{2-}
    doublet, and the H2Om_1635 peak, generated from PyIRoGlass fitting.

    Parameters:
        data (pd.DataFrame): DataFrame containing FTIR spectral data with
            'Absorbance' values indexed by wavenumber. Expected to cover
            the range relevant for carbonate and water peak analysis.
        files (str): Identifier for the current sample set being processed,
            used for titling the plot.
        mc3_output (dict): Dictionary containing results from PyIRoGlass
            fitting, including best fit parameters, standard deviations,
            posterior distributions, and other model outputs.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis object where the
            plot will be drawn. If None, a new figure and axis will be created.

    Returns:
        None: This function does not return any value. It generates a plot
            visualizing the FTIR spectral data, baseline fit, and model fits
            for carbonate and water peaks.

    Note:
        The function is designed to work as part of a larger spectroscopic
        analysis workflow, where spectral data preprocessing, baseline fitting,
        and peak modeling have been previously conducted. It expects specific
        data structures from these analyses as input. If 'ax' is not provided,
        the function creates a new figure and axis for plotting, which might
        not be ideal for integrating this plot into multi-panel figures or
        more complex visual layouts.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(files)

    bestfits, baselines = derive_carbonate(data, files, mc3_output)
    ax.plot(baselines.index, baselines.to_numpy(), "dimgray", linewidth=0.25)
    ax.plot(
        bestfits.index,
        bestfits.Spectrum,
        "tab:blue",
        linewidth=2.5,
        label="FTIR Spectrum",
    )
    ax.plot(
        bestfits.index,
        bestfits.H2Om_1635,
        "tab:orange",
        linewidth=1.5,
        label=r"$\mathregular{H_2O_{m, 1635}}$",
    )
    ax.plot(
        bestfits.index,
        bestfits.CO2_1515,
        "tab:green",
        linewidth=2.5,
        label=r"$\mathregular{CO_{3, 1515}^{2-}}$",
    )
    ax.plot(
        bestfits.index,
        bestfits.CO2_1430,
        "tab:red",
        linewidth=2.5,
        label=r"$\mathregular{CO_{3, 1430}^{2-}}$",
    )
    ax.plot(
        bestfits.index,
        bestfits.Spectrum_Fit,
        "tab:purple",
        linewidth=1.5,
        label="PyIRoGlass Fit",
    )
    ax.plot(bestfits.index, bestfits.Baseline,
            "k", linewidth=1.5, label="Baseline")
    ax.annotate(
        r"$\mathregular{H_2O_{m, 1635}}$ Peak Height: "
        + f"{mc3_output['bestp'][11]:.3f} ± {mc3_output['stdp'][11]:.3f}",
        (0.025, 0.95),
        xycoords="axes fraction",
    )
    ax.annotate(
        r"$\mathregular{CO_{3, 1515}^{2-}}$ Peak Height: "
        + f"{mc3_output['bestp'][10]:.3f} ± {mc3_output['stdp'][10]:.3f}",
        (0.025, 0.90),
        xycoords="axes fraction",
    )
    ax.annotate(
        r"$\mathregular{CO_{3, 1430}^{2-}}$ Peak Height: "
        + f"{mc3_output['bestp'][7]:.3f} ± {mc3_output['stdp'][7]:.3f}",
        (0.025, 0.85),
        xycoords="axes fraction",
    )
    ax.set_xlim([1250, 2400])
    ax.set_xlabel(r"Wavenumber $(\mathregular{cm^{-1}})$")
    ax.set_ylabel("Absorbance")
    ax.legend(loc="upper right", prop={"size": 10})
    ax.tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax.tick_params(axis="y", direction="in", length=5, pad=6.5)
    ax.invert_xaxis()


def plot_trace(posterior, title, zchain=None, pnames=None, thinning=50,
               burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=12):

    """
    Plot parameter trace MCMC sampling.

    Parameters:
        posterior (2D np.ndarray): MCMC posterior sampling with dimension:
            [nsamples, npars].
        zchain (1D np.ndarray): Chain index for each posterior sample.
        pnames (str): Label names for parameters.
        thinning (int): Thinning factor for plotting (plot every thinning-th
            value).
        burnin (int): Thinned burn-in number of iteration (only used when
            zchain is not None).
        fignum (int): The figure number.
        savefile (bool): Name of file to save the plot if not none
        fmt (string): The format string for the line and marker.
        ms (float): Marker size.
        fs (float): Fontsize of texts.

    Returns:
        axes (1D list of matplotlib.axes.Axes): List of axes containing
            the marginal posterior distributions.

    """

    # Get indices for samples considered in final analysis:
    if zchain is not None:
        nchains = np.amax(zchain) + 1
        good = np.zeros(len(zchain), bool)
        for c in range(nchains):
            good[np.where(zchain == c)[0][burnin:]] = True
        # Values accepted for posterior stats:
        posterior = posterior[good]
        zchain = zchain[good]
        # Sort the posterior by chain:
        zsort = np.lexsort([zchain])
        posterior = posterior[zsort]
        zchain = zchain[zsort]
        # Get location for chains separations:
        xsep = np.where(np.ediff1d(zchain[0::thinning]))[0]

    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)
    # Number of samples (thinned):
    xmax = len(posterior[0::thinning])

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    npanels = 16  # Max number of panels per page
    npages = int(1 + (npars - 1) / npanels)

    # Make the trace plot:
    axes = []
    ipar = 0
    for page in range(npages):
        fig = plt.figure(fignum + page, figsize=(8.5, 11.0))
        plt.clf()
        plt.subplots_adjust(left=0.15, right=0.95,
                            bottom=0.05, top=0.95, hspace=0.15)
        while ipar < npars:
            ax = plt.subplot(npanels, 1, ipar % npanels + 1)
            axes.append(ax)
            ax.plot(posterior[0::thinning, ipar],
                    fmt, ms=ms, c="#46A4F5", rasterized=True)
            yran = ax.get_ylim()
            if zchain is not None:
                ax.vlines(xsep, yran[0], yran[1], "0.5")
            # Y-axis adjustments:
            ax.set_ylim(yran)
            ax.locator_params(axis="y", nbins=5, tight=True)
            ax.tick_params(labelsize=fs - 1, direction="in",
                           top=True, right=True)
            ax.set_ylabel(pnames[ipar], size=fs, multialignment="center")
            # X-axis adjustments:
            ax.set_xlim(0, xmax)
            ax.get_xaxis().set_visible(False)
            ipar += 1
            if ipar % npanels == 0:
                break
        ax.set_xlabel("Thinned MCMC Sample", size=fs)
        ax.get_xaxis().set_visible(True)

        if savefile is not None:
            if npages > 1:
                sf = os.path.splitext(savefile)
                try:
                    bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.1)
                    bbox_points = bbox.get_points()
                    bbox_points[:, 0] = 0.0, 8.5
                    bbox.set_points(bbox_points)
                except Exception:
                    ylow = 9.479 - 0.862 * np.amin(
                        [npanels - 1, npars - npanels * page - 1]
                    )
                    bbox = mpl.transforms.Bbox([[0.0, ylow], [8.5, 11]])

                fig.savefig(f"{sf[0]}_page{page:02d}{sf[1]}", bbox_inches=bbox)
            else:
                fig.suptitle(title, fontsize=fs + 1)
                plt.ioff()
                fig.savefig(savefile, bbox_inches="tight")

    return axes


def plot_modelfit(data, uncert, indparams, model, title, nbins=75,
                  fignum=1400, savefile=None):

    """
    Plot the binned dataset with given uncertainties and model curves as
    a function of indparams. In a lower panel, plot the residuals between
    the data and model.

    Parameters:
        data (1D np.ndarray): Input data set.
        uncert (1D np.ndarray): One-sigma uncertainties of the data points.
        indparams (1D np.ndarray): Independent variable (X axis) of the
            data points.
        model (1D np.ndarray): Model of data.
        nbins (int): Number of bins in the output plot.
        fignum (int): Figure number.
        savefile (bool): Name of file to save the plot, if not none.

    Returns:
        ax (matplotlib.axes.Axes): Axes instance containing the marginal
            posterior distributions.

    """

    # Bin down array:
    binsize = int((np.size(data) - 1) / nbins + 1)
    binindp = ms.bin_array(indparams, binsize)
    binmodel = ms.bin_array(model, binsize)
    bindata, binuncert = ms.bin_array(data, binsize, uncert)
    fs = 12

    plt.figure(fignum, figsize=(8, 6))
    plt.clf()

    # Residuals:
    rax = plt.axes([0.15, 0.1, 0.8, 0.2])
    rax.errorbar(binindp, bindata - binmodel, binuncert, fmt="ko", ms=4)
    rax.plot([indparams[0], indparams[-1]], [0, 0], "k:", lw=1.5)
    rax.tick_params(labelsize=fs - 1, direction="in", top=True, right=True)
    rax.set_xlabel("Wavenumber (cm^-1)", fontsize=fs)
    rax.set_ylabel("Residual", fontsize=fs)
    rax.invert_xaxis()

    # Data and Model:
    ax = plt.axes([0.15, 0.35, 0.8, 0.575])
    ax.errorbar(binindp, bindata, binuncert,
                fmt="ko", ms=4, label="Binned data")
    ax.plot(indparams, data, "tab:blue", lw=3, label="Data")
    ax.plot(indparams, model, "tab:purple",
            linestyle="-.", lw=3, label="Best Fit")
    ax.set_xticklabels([])
    ax.invert_xaxis()
    ax.tick_params(labelsize=fs - 1, direction="in", top=True, right=True)
    ax.set_ylabel("Absorbance", fontsize=fs)
    ax.legend(loc="best")

    if savefile is not None:
        plt.suptitle(title, fontsize=fs + 1)
        plt.ioff()
        plt.savefig(savefile, bbox_inches="tight")

    return ax, rax
