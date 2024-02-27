# %%

__author__ = 'Sarah Shi'

import numpy as np
import pandas as pd
import scipy.signal as signal

from matplotlib import pyplot as plt

# %% Reflectance FTIR - Interference Fringe Processing for Thicknesses

def datacheck_peakdetect(x_axis, y_axis):

    """
    Check and prepare data for peak detection analysis.

    This function ensures that the input data for peak detection is in the
    correct format and that the x- and y-axis data are of equal lengths. If
    the x-axis data is not provided, it generates an x-axis as a range of
    integers with the same length as the y-axis data.

    Parameters:
        x_axis (np.ndarray or None): A 1D array containing the x-axis data or
            None if an x-axis is to be generated.
        y_axis (np.ndarray): A 1D array containing the y-axis data.

    Returns:
        Tuple containing two numpy arrays:
            x_axis (np.ndarray): The checked or generated x-axis data.
            y_axis (np.ndarray): The checked y-axis data.

    Raises:
        ValueError: If the lengths of the x-axis and y-axis data do not match.

    Notes:
        This function comes from https://github.com/avhn/peakdetect. I pulled
        this for peak fitting, as the repository is no longer maintained and
        installation no longer works.
    """

    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
            "Input vectors y_axis and x_axis must have same length")

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0):

    """
    Detect local maxima and minima in a signal based on a MATLAB script
    (http://billauer.co.il/peakdet.html).

    This function identifies peaks by searching for values which are surrounded
    by lower or higher values for maxima and minima, respectively.

    Parameters:
        y_axis (list or np.ndarray): A list or 1D numpy array containing the
            signal over which to find peaks.
        x_axis (list or np.ndarray, optional): A list or 1D numpy array whose
            values correspond to the y_axis list and is used in the return to
            specify the position of the peaks. If omitted, an index of the
            y_axis is used. Defaults to None.
        lookahead (int, optional): Distance to look ahead from a peak candidate
            to determine if it is the actual peak. Defaults to 200. A good
            value might be '(samples / period) / f' where '4 >= f >= 1.25'.
        delta (float, optional): Specifies a minimum difference between a peak
            and the following points, before a peak may be considered a peak.
            Useful to hinder the function from picking up false peaks towards
            the end of the signal. To work well, delta should be set to
            delta >= RMSnoise x 5. Defaults to 0. When omitted, it can decrease
            the speed by 20%, but when used correctly, it can double the speed
            of the function.

    Returns:
        A tuple of two lists ([max_peaks, min_peaks]) containing the positive
        and negative peaks, respectively. Each element of the lists is a
        tuple of (position, peak_value). To get the average peak value,
        use: np.mean(max_peaks, 0)[1]. To unpack one of the lists into x, y
        coordinates, use: x, y = zip(max_peaks).

    Notes:
        This function comes from https://github.com/avhn/peakdetect. I pulled
        this for peak fitting, as the repository is no longer maintained and
        installation no longer works.
    """

    max_peaks = []
    min_peaks = []
    dump = []

    # Check input data
    x_axis, y_axis = datacheck_peakdetect(x_axis, y_axis)
    # Store data length for later use
    length = len(y_axis)

    # Perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # Maxima (mx) and minima (mn) candidates are temporarily stored
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        # Look for max
        if y < mx-delta and mx != np.Inf:
            # Maxima peak candidate found
            # Look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # Set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    # End is within lookahead no more peaks can be found
                    break
                continue

        # Look for min
        if y > mn+delta and mn != -np.Inf:
            # Minima peak candidate found
            # Look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # Set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        pass

    return [max_peaks, min_peaks]



def peakID(ref_spec, wn_high, wn_low, peak_heigh_min_delta, peak_search_width,
           savgol_filter_width, smoothing_wn_width=None, remove_baseline=True,
           plotting=False, filename=None):

    """
    Identifies peaks based on the peakdetect package which
    identifies local maxima and minima in noisy signals.
    Based on: https://github.com/avhn/peakdetect

    Parameters:
        ref_spec (pd.DataFrame): A Pandas DataFrame indexed by wavenumber
            and containing absorbance values.
        wn_high (int): The upper wavenumber limit for the analysis.
        wn_low (int): The lower wavenumber limit for the analysis.
        smoothing_wn_width (int): The window size for the Savitzky-Golay
            smoothing filter. Default is None.
        remove_baseline (bool): Whether to remove the baseline from the
            spectrum. Default is False.
        peak_heigh_min_delta (float): Minimum difference between a peak and
            its neighboring points for it to be considered a peak.
            Default is 0.008.
        peak_search_width (int): The size of the region around each point
            to search for a peak. Default is 50.
        plotting (bool): Whether to create a plot of the spectrum with
            identified peaks and troughs. Default is False.
        filename (str): The name of the plot file. If None, the plot is not
            saved. Default is None.

    Returns:
        Tuple containing the following elements:
            Peaks and troughs identified as local maxima and minima.
    """

    spec = ref_spec.loc[wn_low:wn_high].copy()  # df indexed by wavenumber
    spec_filt = pd.DataFrame(columns=['Wavenumber', 'Absorbance'])
    baseline = 0

    spec_filter = signal.medfilt(spec.Absorbance, 3)

    if remove_baseline is True:
        baseline = signal.savgol_filter(spec_filter, savgol_filter_width, 3)
        spec_filter = spec_filter - baseline

    if smoothing_wn_width is not None:
        spec_filter = signal.savgol_filter(spec_filter, smoothing_wn_width, 3)

    spec_filt['Absorbance'] = spec_filter
    spec_filt.index = spec.index
    spec['Subtracted'] = spec['Absorbance'] - baseline

    pandt = peakdetect(spec_filt.Absorbance, spec_filt.index,
                       lookahead=peak_search_width,
                       delta=peak_heigh_min_delta)
    peaks = np.array(pandt[0])
    troughs = np.array(pandt[1])

    if plotting is False:
        pass
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(spec.index, spec['Subtracted'], linewidth=1)
        ax.plot(spec_filt.index, spec_filt.Absorbance)
        ax.plot(peaks[:, 0], peaks[:, 1], 'ro')
        ax.plot(troughs[:, 0], troughs[:, 1], 'ko')
        ax.set_title(filename)
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance')
        ax.invert_xaxis()

    return peaks, troughs


def calculate_thickness(n, positions):

    """
    Calculates thicknesses of glass wafers based on the refractive index of the
    glass and the positions of the peaks or troughs in the FTIR spectrum.

    Parameters:
        n (float): Refractive index of the glass.
        positions (np.ndarray): Array of positions of the peaks or troughs in
            the FTIR spectrum.

    Returns:
        np.ndarray: Array of thicknesses of glass wafers.
    """

    return 1/(2 * n * np.abs(np.diff(positions)))


def calculate_mean_thickness(dfs_dict, n, wn_high, wn_low,
                             remove_baseline=True, plotting=False, phaseol=True):

    """
    Calculates thickness of glass wafers based on the refractive index of
    the glass and the positions of the peaks or troughs in the FTIR spectrum.
    Thicknesses for each interference fringe, starting at both the peaks
    and troughs of the fringes are determined. These thicknesses are then
    averaged over the interval of interest.

    Parameters:
        dfs_dict (dictionary): dictionary containing FTIR data for each
            file
        n (float): refractive index of the glass
        wn_high (float): the high wavenumber cutoff for the analysis
        wn_low (float): the low wavenumber cutoff for the analysis
        remove_baseline (bool): whether or not to remove the baseline
            from the data
        plotting (bool): whether or not to plot the data and detected
            peaks and troughs

    Returns:
        ThickDF (pd.DataFrame): a dataframe containing the thickness
        calculations for each file. 

    Notes:
        smoothing_wn_width (float): Width of the Savitzky-Golay smoothing
        window, if not used, set to None. 
        peak_heigh_min_delta (float): Minimum height difference between a
        peak and its surrounding points.
        peak_search_width (float): Distance (in wavenumbers) to look on
        either side of a peak to find the corresponding trough.
    """

    ThickDF = pd.DataFrame(
        columns=[
            'Thickness_M',
            'Thickness_STD',
            'Peak_Thicknesses',
            'Peak_Thickness_M',
            'Peak_Thickness_STD',
            'Trough_Thicknesses',
            'Trough_Thickness_M',
            'Trough_Thickness_STD'
            ]
            )

    failures = []

    # If phase is olivine, set these parameters.
    if phaseol is True:
        savgol_filter_width = 99
        smoothing_wn_width = 15
        peak_heigh_min_delta = 0.002
        peak_search_width = 10
    # If phase glass, set other parameters.
    else:
        savgol_filter_width = 449
        smoothing_wn_width = 71
        peak_heigh_min_delta = 0.008
        peak_search_width = 50

    for filename, data in dfs_dict.items():
        try:
            peaks, troughs = peakID(data, wn_high, wn_low, filename=filename,
                                    plotting=plotting,
                                    savgol_filter_width=savgol_filter_width,
                                    smoothing_wn_width=smoothing_wn_width,
                                    remove_baseline=True,
                                    peak_heigh_min_delta=peak_heigh_min_delta,
                                    peak_search_width=peak_search_width)
            peaks_loc = peaks[:, 0].round(2)
            troughs_loc = troughs[:, 0].round(2)
            peaks_diff = np.diff(peaks[:, 0]).round(2)
            troughs_diff = np.diff(troughs[:, 0]).round(2)

            peaks_loc_filt = np.array([x for x in peaks_loc if
                                       abs(x - np.mean(peaks_loc))
                                       < 2 * np.std(peaks_loc)])
            troughs_loc_filt = np.array([x for x in troughs_loc if
                                         abs(x - np.mean(troughs_loc))
                                         < 2 * np.std(troughs_loc)])
            peaks_diff_filt = np.array([x for x in peaks_diff if
                                        abs(x - np.mean(peaks_diff))
                                        < 2 * np.std(peaks_diff)])
            troughs_diff_filt = np.array([x for x in troughs_diff if
                                          abs(x - np.mean(troughs_diff))
                                          < 2 * np.std(troughs_diff)])

            t_peaks = (calculate_thickness(n, peaks[:, 0]) * 1e4).round(2)
            t_peaks_filt = np.array([x for x in t_peaks if
                                     abs(x - np.mean(t_peaks))
                                     < np.std(t_peaks)])
            mean_t_peaks_filt = np.mean(t_peaks_filt).round(2)
            std_t_peaks_filt = np.std(t_peaks_filt).round(2)

            t_troughs = (calculate_thickness(n, troughs[:, 0]) * 1e4).round(2)
            t_troughs_filt = [x for x in t_troughs if
                              abs(x - np.mean(t_troughs))
                              < np.std(t_troughs)]
            mean_t_troughs_filt = np.mean(t_troughs_filt).round(2)
            std_t_troughs_filt = np.std(t_troughs_filt).round(2)

            mean_t = np.mean(
                np.concatenate([t_peaks_filt, t_troughs_filt])
                ).round(2)
            std_t = np.std(
                np.concatenate([t_peaks_filt, t_troughs_filt])
                ).round(2)

            ThickDF.loc[f"{filename}"] = pd.Series({
                'Thickness_M': mean_t,
                'Thickness_STD': std_t,
                'Peak_Thicknesses': t_peaks_filt,
                'Peak_Thickness_M': mean_t_peaks_filt,
                'Peak_Thickness_STD': std_t_peaks_filt,
                'Peak_Loc': peaks_loc_filt,
                'Peak_Diff': peaks_diff_filt,
                'Trough_Thicknesses': t_troughs_filt,
                'Trough_Thickness_M': mean_t_troughs_filt,
                'Trough_Thickness_STD': std_t_troughs_filt,
                'Trough_Loc': troughs_loc_filt,
                'Trough_Diff': troughs_diff_filt
                })

        except Exception as e:
            print(f"Error: {e}")
            print(e)
            failures.append(filename)
            ThickDF.loc[filename] = pd.Series({
                'V1': np.nan, 'V2': np.nan,
                'Thickness': np.nan
                })

    return ThickDF


def reflectance_index(XFo):

    """
    Calculates the reflectance index for a given forsterite composition.
    The reflectance index is calculated based on values from Deer, Howie,
    and Zussman, 3rd Edition.

    Parameters:
        XFo (float): The mole fraction of forsterite in the sample.

    Returns:
        n (float): The calculated reflectance index.
    """

    n_alpha = 1.827 - 0.192*XFo
    n_beta = 1.869 - 0.218*XFo
    n_gamma = 1.879 - 0.209*XFo
    n = (n_alpha+n_beta+n_gamma) / 3

    return n