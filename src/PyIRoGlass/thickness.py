# %%

__author__ = "Sarah Shi"

import numpy as np
import pandas as pd
import scipy.signal as signal

from matplotlib import pyplot as plt

# %% Reflectance FTIR - Interference Fringe Processing for Thicknesses


def peakID(
    ref_spec,
    wn_high,
    wn_low,
    peak_search_width,
    savgol_filter_width,
    smoothing_wn_width=None,
    auto_smoothing_width=True,
    auto_savgol_width=True,
    auto_search_width=True,
    n_sigma=1.5,
    plotting=False,
    filename=None,
):

    """
    Identifies peaks using scipy.signal.find_peaks with a single
    per-file threshold, n_sigma * std(whole baseline-subtracted signal).
    A single global delta matched manually-verified reference peak
    counts across every test file, while still adapting the threshold
    to each file's own amplitude scale.

    Fringe period varies severalfold across samples (e.g. ~7-8 points
    for a thick wafer vs. ~180 points for a thin one). Fixed
    savgol_filter_width/smoothing_wn_width/peak_search_width values
    tuned for one fringe density don't transfer: a savgol_filter_width
    comparable to or narrower than the true fringe period makes the
    baseline fit track the real oscillation instead of just the slow
    background, and a too-narrow peak_search_width lets find_peaks
    count small ripples between real fringes as separate peaks (roughly
    doubling the count). With auto_savgol_width/auto_smoothing_width/
    auto_search_width=True (default), a fringe period is first
    estimated via a deliberately wide bootstrap baseline (see
    bootstrap_fringe_period), then used to scale all three widths for
    this sample specifically.

    Parameters:
        ref_spec (pd.DataFrame): A Pandas DataFrame indexed by wavenumber
            and containing absorbance values.
        wn_high (int): The upper wavenumber limit for the analysis.
        wn_low (int): The lower wavenumber limit for the analysis.
        peak_search_width (int): The size of the region around each point
            to search for a peak.
        savgol_filter_width (int): The window size for the baseline
            Savitzky-Golay filter.
        smoothing_wn_width (int): The window size for the second
            Savitzky-Golay smoothing filter. Default is None.
        auto_smoothing_width (bool): If True (default), smoothing_wn_width
            is capped based on this sample's own estimated fringe period
            rather than used as a fixed value.
        auto_savgol_width (bool): If True (default), savgol_filter_width
            is raised toward this sample's own estimated fringe period if
            the requested value is too narrow for it.
        auto_search_width (bool): If True (default), peak_search_width
            is raised toward this sample's own estimated fringe period if
            the requested value is too narrow for it.
        n_sigma (float): Multiplier on std(whole signal) used as the
            find_peaks prominence threshold. Default is 1.5.
        plotting (bool): Whether to create a plot of the spectrum with
            identified peaks and troughs. Default is False.
        filename (str): The name of the plot title. Default is None.

    Returns:
        Tuple containing the following elements:
            Peaks and troughs identified as local maxima and minima.
    """

    spec = ref_spec.loc[wn_low:wn_high].copy()
    n_points = len(spec)

    medfilt_absorbance = signal.medfilt(spec.Absorbance, 3)

    bootstrap_period = None
    if auto_savgol_width or auto_search_width:
        bootstrap_period = bootstrap_fringe_period(medfilt_absorbance, n_points)

    if auto_savgol_width and bootstrap_period is not None:
        savgol_filter_width = safe_savgol_width(
            n_points, max(savgol_filter_width, int(bootstrap_period * 2.5))
        )
    if auto_search_width and bootstrap_period is not None:
        peak_search_width = max(peak_search_width, int(bootstrap_period * 0.6))

    baseline = signal.savgol_filter(medfilt_absorbance, savgol_filter_width, 3)
    spec_filter = medfilt_absorbance - baseline
    subtracted = spec_filter.copy()  # pre-second-smoothing, for refinement

    if smoothing_wn_width is not None:
        if auto_smoothing_width:
            smoothing_wn_width = safe_smoothing_width(
                subtracted, smoothing_wn_width
            )
        spec_filter = signal.savgol_filter(spec_filter, smoothing_wn_width, 3)

    spec_filt = pd.DataFrame({"Absorbance": spec_filter}, index=spec.index)
    spec["Subtracted"] = subtracted

    delta = n_sigma * np.std(spec_filter)
    wn_arr = spec_filt.index.values

    pk, _ = signal.find_peaks(
        spec_filter, distance=peak_search_width, prominence=delta
    )
    tr, _ = signal.find_peaks(
        -spec_filter, distance=peak_search_width, prominence=delta
    )

    def refine(idx_arr, find_max):
        # Second smoothing pass is used to decide locations of
        # candidate peaks/troughs; attenuates and can shift extrema.
        # Candidates are snapped to the true local extrema on the
        # less-smoothed 'subtracted' signal within one
        # peak_search_width. Report that position and value.
        refined_idx = []
        for i in idx_arr:
            lo = max(0, i - peak_search_width)
            hi = min(len(subtracted), i + peak_search_width + 1)
            window = subtracted[lo:hi]
            refined_idx.append(lo + (np.argmax(window) if find_max
                                      else np.argmin(window)))
        return np.array(refined_idx, dtype=int)

    pk = refine(pk, find_max=True)
    tr = refine(tr, find_max=False)

    peaks = (np.column_stack([wn_arr[pk], subtracted[pk]])
             if len(pk) else np.empty((0, 2)))
    troughs = (np.column_stack([wn_arr[tr], subtracted[tr]])
               if len(tr) else np.empty((0, 2)))

    if plotting is False:
        pass
    else:
        # Only the less-smoothed subtracted signal is shown: peak/trough
        # values and positions are read from it, so the smoothed curve
        # used only to decide candidate locations would undersell the
        # true peak amplitude without adding information.
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(spec.index, spec["Subtracted"], color='#0C7BDC', linewidth=1)
        if len(peaks):
            ax.scatter(peaks[:, 0], peaks[:, 1], marker='^', color='white',
                       ec='k', lw=1.5, zorder=30)
        if len(troughs):
            ax.scatter(troughs[:, 0], troughs[:, 1], marker='v', color='white',
                       ec='k', lw=1.5, zorder=30)
        ax.set_title(filename)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Absorbance")
        ax.invert_xaxis()

    return peaks, troughs


def safe_savgol_width(n_points, requested_width, min_width=5):

    """
    Clamps a Savitzky-Golay window length to fit within available data.

    Parameters:
        n_points (int): Number of points available to filter.
        requested_width (int): Desired window length.
        min_width (int): Minimum window length to fall back to.
            Default is 5.

    Returns:
        int: A valid, odd window length no larger than n_points.
    """

    max_width = n_points - 1 if n_points % 2 == 0 else n_points
    width = min(requested_width, max_width)
    if width % 2 == 0:
        width -= 1
    return max(width, min_width)


def estimate_fringe_period_points(subtracted, min_period=3, n_sigma=1.2):

    """
    Roughly estimates the fringe period, in points, from a
    baseline-subtracted signal, using true extrema filtered by a
    prominence threshold relative to the signal's own amplitude.

    A distance-only constraint (no prominence) undercounts the fringe
    period whenever noise amplitude is comparable to real fringe
    amplitude at short lags: small noise wiggles between real fringes
    get counted as extrema too, badly underestimating the period (e.g.
    16 points instead of ~178 for a widely-spaced fringe pattern with a
    similar-scale noise floor, or worse if there's a real secondary
    ripple superimposed on the main fringe). Filtering by prominence
    rejects those small wiggles and recovers the true spacing between
    the dominant real extrema.

    Parameters:
        subtracted (array-like): Baseline-subtracted signal.
        min_period (int): Minimum spacing, in points, enforced between
            extrema. Default is 3.
        n_sigma (float): Multiplier on std(subtracted) used as the
            find_peaks prominence threshold. Default is 1.2.

    Returns:
        float or None: Estimated fringe period in points, or None if
            fewer than 2 extrema are found.
    """

    prominence = n_sigma * np.std(subtracted)
    pk, _ = signal.find_peaks(subtracted, distance=min_period,
                               prominence=prominence)
    tr, _ = signal.find_peaks(-subtracted, distance=min_period,
                               prominence=prominence)
    extrema = np.sort(np.concatenate([pk, tr]))
    if len(extrema) < 2:
        return None
    # consecutive extrema (peak-to-trough) are ~half a fringe period apart
    return 2 * np.median(np.diff(extrema))


def bootstrap_fringe_period(medfilt_absorbance, n_points,
                             n_widths=16, plateau_len=4, tol=0.15):

    """
    Estimates the fringe period from a medfilt'd absorbance array by
    scanning a range of Savitzky-Golay baseline widths (geometrically
    spaced from narrow up to 90% of the window) and returning the
    period from the longest run of consecutive widths whose estimates
    agree within tol, so the estimate itself doesn't depend on already
    having chosen the right savgol_filter_width.

    A single fixed bootstrap width doesn't work across very different
    fringe densities: a width comparable to or narrower than the true
    fringe period lets the baseline track the real oscillation itself
    (destroying the signal a period estimate needs), while a width
    that's a large fraction of a SHORT window can be nearly as wide as
    the whole window, over-smoothing away real narrow fringes and
    leaving only one dominant broad envelope (which looks like a valid
    but spurious "fringe" of its own). Scanning widths and requiring a
    multi-width plateau finds the width range where the true fringe
    period genuinely dominates the residual, for either regime.

    Parameters:
        medfilt_absorbance (array-like): Median-filtered absorbance
            (signal.medfilt(spec.Absorbance, 3), before any real
            baseline subtraction).
        n_points (int): Number of points in the window.
        n_widths (int): Number of candidate baseline widths to scan,
            geometrically spaced. Default is 16.
        plateau_len (int): Minimum number of consecutive candidate
            widths whose period estimates must agree (within tol) to be
            accepted as the true period. Default is 4.
        tol (float): Relative tolerance for two period estimates to be
            considered part of the same plateau. Default is 0.15.

    Returns:
        float or None: Estimated fringe period in points, or None if a
            period can't be estimated.
    """

    max_width = max(int(n_points * 0.9), 9)
    candidate_widths = sorted(set(
        safe_savgol_width(n_points, w)
        for w in np.unique(np.geomspace(9, max_width, n_widths).astype(int))
    ))

    periods = []
    for w in candidate_widths:
        baseline = signal.savgol_filter(medfilt_absorbance, w, 3)
        residual = medfilt_absorbance - baseline
        periods.append(estimate_fringe_period_points(residual))

    best_len, best_val = 0, None
    i = 0
    while i < len(periods):
        if periods[i] is None:
            i += 1
            continue
        j = i
        while (j + 1 < len(periods) and periods[j + 1] is not None and
               abs(periods[j + 1] - periods[i]) /
               max(periods[i], periods[j + 1]) < tol):
            j += 1
        run_len = j - i + 1
        # >= (not >): on a tie, prefer the run found at a WIDER baseline
        # width, scanned later. Coincidental agreement between a few
        # narrow-width noise-driven estimates is common; a plateau that
        # holds at wide scales is the more reliable signal of the true
        # period.
        if run_len >= best_len:
            best_len = run_len
            best_val = np.mean(periods[i:j + 1])
        i = j + 1

    if best_len >= plateau_len:
        return best_val
    valid = [p for p in periods if p is not None]
    return valid[-1] if valid else None


def safe_smoothing_width(subtracted, requested_width, min_width=5,
                          max_fraction_of_period=0.5):

    """
    Caps smoothing_wn_width at a fraction of the sample's own estimated
    fringe period, so the second Savitzky-Golay smoothing pass denoises
    without averaging together adjacent real fringes.

    Parameters:
        subtracted (array-like): Baseline-subtracted signal.
        requested_width (int): Desired smoothing window length.
        min_width (int): Minimum window length to fall back to. Default
            is 5, the floor for savgol_filter's polyorder=3.
        max_fraction_of_period (float): Fraction of the estimated fringe
            period the smoothing window is capped at. Default is 0.5.

    Returns:
        int: A valid, odd smoothing window length. Falls back to
            requested_width if a fringe period can't be estimated.
    """

    period = estimate_fringe_period_points(subtracted)
    if period is None:
        return requested_width
    width = min(requested_width, int(period * max_fraction_of_period))
    if width % 2 == 0:
        width -= 1
    return max(width, min_width)


def safe_search_width(subtracted, requested_width,
                       min_fraction_of_period=0.6):

    """
    Raises peak_search_width toward the sample's own estimated fringe
    period if the requested value is too small relative to it.

    find_peaks' distance parameter (passed as peak_search_width) has to
    be comparable to the true spacing between same-type extrema, or
    find_peaks accepts small ripples/noise between real fringes as
    separate peaks. A fixed peak_search_width tuned for a
    narrow-fringed sample (e.g. ~7-30 points) is far too small for a
    sample with a much wider fringe period (e.g. ~90 points), and can
    roughly double the peak count with spurious detections.

    Parameters:
        subtracted (array-like): Baseline-subtracted signal.
        requested_width (int): Desired peak_search_width.
        min_fraction_of_period (float): Minimum fraction of the
            estimated fringe period peak_search_width is raised to.
            Default is 0.6.

    Returns:
        int: requested_width, or the estimated fringe period scaled by
            min_fraction_of_period if that's larger. Falls back to
            requested_width if a fringe period can't be estimated.
    """

    period = estimate_fringe_period_points(subtracted)
    if period is None:
        return requested_width
    return max(requested_width, int(period * min_fraction_of_period))


def calculate_thickness(n, positions):

    """
    Calculates thicknesses of the wafer(s) based on the refractive index of the
    wafer and the positions of the peaks or troughs in the FTIR spectrum.

    Parameters:
        n (float): Refractive index of the wafer.
        positions (np.ndarray): Array of positions of the peaks or troughs in
            the FTIR spectrum.

    Returns:
        np.ndarray: Array of thicknesses of glass wafers.
    """

    return 1 / (2 * n * np.abs(np.diff(positions)))


def calculate_mean_thickness(
    dfs_dict, n, wn_high, wn_low,
    savgol_filter_width=99, smoothing_wn_width=15, peak_search_width=5,
    n_sigma=1.5,
    plotting=False,
):

    """
    Calculates thickness of the wafer(s) based on the refractive index of
    the wafer and the positions of the peaks or troughs in the FTIR spectrum.
    Thicknesses for each interference fringe, starting at both the peaks
    and troughs of the fringes are determined. These thicknesses are then
    averaged over the interval of interest.

    savgol_filter_width/smoothing_wn_width/peak_search_width previously
    were chosen from a phaseol flag (olivine vs. glass defaults).
    smoothing_wn_width is made adaptive here via auto_smoothing_width in
    peakID, and the former glass-phase savgol_filter_width/
    peak_search_width defaults (449/50) were tuned for a much wider,
    coarser-fringed acquisition window than typical narrow-window
    samples -- they clamp down to nearly the entire window and can drop
    most real peaks. Pass the olivine-style values that work across the
    datasets tested directly, and override them if a specific sample
    needs something different.

    Parameters:
        dfs_dict (dictionary): dictionary containing FTIR data for each
            file
        n (float): refractive index of the wafer
        wn_high (float): the high wavenumber cutoff for the analysis
        wn_low (float): the low wavenumber cutoff for the analysis
        savgol_filter_width (int): The window size for the baseline
            Savitzky-Golay filter. Default is 99.
        smoothing_wn_width (int): The window size for the second
            Savitzky-Golay smoothing filter, before being capped by
            auto_smoothing_width. Default is 15.
        peak_search_width (int): The size of the region around each
            point to search for a peak. Default is 5.
        n_sigma (float): adaptive peak height threshold multiplier.
            Default is 1.5.
        plotting (bool): whether or not to plot the data and detected
            peaks and troughs

    Returns:
        ThickDF (pd.DataFrame): a dataframe containing the thickness
        calculations for each file.
    """

    ThickDF = pd.DataFrame(
        columns=[
            "Thickness_M",
            "Thickness_STD",
            "Peak_Thicknesses",
            "Peak_Thickness_M",
            "Peak_Thickness_STD",
            "Trough_Thicknesses",
            "Trough_Thickness_M",
            "Trough_Thickness_STD",
        ]
    )

    failures = []

    for filename, data in dfs_dict.items():
        try:
            n_points = len(data.loc[wn_low:wn_high])
            safe_width = safe_savgol_width(n_points, savgol_filter_width)
            safe_smooth = safe_savgol_width(n_points, smoothing_wn_width)

            peaks, troughs = peakID(
                data,
                wn_high,
                wn_low,
                filename=filename,
                plotting=plotting,
                savgol_filter_width=safe_width,
                smoothing_wn_width=safe_smooth,
                peak_search_width=peak_search_width,
                n_sigma=n_sigma,
            )
            peaks_loc = peaks[:, 0].round(2)
            troughs_loc = troughs[:, 0].round(2)
            peaks_diff = np.diff(peaks[:, 0]).round(2)
            troughs_diff = np.diff(troughs[:, 0]).round(2)

            peaks_loc_filt = np.array(
                [
                    x
                    for x in peaks_loc
                    if (abs(x - np.mean(peaks_loc)) <=
                        2 * np.std(peaks_loc))
                ]
            )
            troughs_loc_filt = np.array(
                [
                    x
                    for x in troughs_loc
                    if (abs(x - np.mean(troughs_loc)) <=
                        2 * np.std(troughs_loc))
                ]
            )
            peaks_diff_filt = np.array(
                [
                    x
                    for x in peaks_diff
                    if (abs(x - np.mean(peaks_diff)) <=
                        2 * np.std(peaks_diff))
                ]
            )
            troughs_diff_filt = np.array(
                [
                    x
                    for x in troughs_diff
                    if (abs(x - np.mean(troughs_diff)) <=
                        2 * np.std(troughs_diff))
                ]
            )

            t_peaks = (calculate_thickness(n, peaks[:, 0]) * 1e4).round(2)
            t_peaks_filt = np.array(
                [x for x in t_peaks if (abs(x - np.mean(t_peaks)) <=
                                        np.std(t_peaks))]
            )
            mean_t_peaks_filt = np.mean(t_peaks_filt).round(2)
            std_t_peaks_filt = np.std(t_peaks_filt).round(2)

            t_troughs = (calculate_thickness(n, troughs[:, 0]) * 1e4).round(2)
            t_troughs_filt = np.array(
                [x for x in t_troughs if (abs(x - np.mean(t_troughs)) <=
                                          np.std(t_troughs))]
            )
            mean_t_troughs_filt = np.mean(t_troughs_filt).round(2)
            std_t_troughs_filt = np.std(t_troughs_filt).round(2)

            mean_t = np.mean(np.concatenate([t_peaks_filt,
                                             t_troughs_filt])).round(2)
            std_t = np.std(np.concatenate([t_peaks_filt,
                                           t_troughs_filt])).round(2)

            ThickDF.loc[f"{filename}"] = pd.Series(
                {
                    "Thickness_M": mean_t,
                    "Thickness_STD": std_t,
                    "Peak_Thicknesses": t_peaks_filt,
                    "Peak_Thickness_M": mean_t_peaks_filt,
                    "Peak_Thickness_STD": std_t_peaks_filt,
                    "Peak_Loc": peaks_loc_filt,
                    "Peak_Diff": peaks_diff_filt,
                    "Trough_Thicknesses": t_troughs_filt,
                    "Trough_Thickness_M": mean_t_troughs_filt,
                    "Trough_Thickness_STD": std_t_troughs_filt,
                    "Trough_Loc": troughs_loc_filt,
                    "Trough_Diff": troughs_diff_filt,
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            print(e)
            failures.append(filename)
            ThickDF.loc[filename] = pd.Series(
                {"V1": np.nan, "V2": np.nan, "Thickness": np.nan}
            )

    return ThickDF


def reflectance_index_ol(XFo):

    """
    Calculates the reflectance index of olivine for a given XFo composition.
    The reflectance index is calculated based on values from Deer, Howie,
    and Zussman, 3rd Edition.

    Parameters:
        XFo (float): The mole fraction of forsterite in the sample.

    Returns:
        n (float): The calculated reflectance index.
    """

    n_alpha = 1.827 - 0.192 * XFo
    n_beta = 1.869 - 0.218 * XFo
    n_gamma = 1.879 - 0.209 * XFo
    n = (n_alpha + n_beta + n_gamma) / 3

    return n


def reflectance_index_opx(XMg):

    """
    Calculates the reflectance index of orthopyroxene for a given XMg composition.
    The reflectance index is calculated based on enstatite-ferrosilite 
    values from Deer, Howie, and Zussman, 3rd Edition.

    Parameters:
        XMg (float): The mole fraction of Mg/(Mg+Fe+Mn) in the sample.

    Returns:
        n (float): The calculated reflectance index.
    """

    n_alpha = 1.768 - 0.118 * XMg
    n_beta = 1.770 - 0.117 * XMg
    n_gamma = 1.788 - 0.130 * XMg
    n = (n_alpha + n_beta + n_gamma) / 3

    return n


def reflectance_index_cpx(XMg):

    """
    Calculates the reflectance index of crthopyroxene for a given XMg composition.
    The reflectance index is calculated based on diopside-hedenbergite
    values from Deer, Howie, and Zussman, 3rd Edition. 

    Parameters:
        XMg (float): The mole fraction of Mg/(Mg+Fe+Mn) in the sample.

    Returns:
        n (float): The calculated reflectance index.
    """

    n_alpha = 1.732 - 0.068 * XMg
    n_beta = 1.730 - 0.058 * XMg
    n_gamma = 1.755 - 0.061 * XMg
    n = (n_alpha + n_beta + n_gamma) / 3

    return n


# %%
