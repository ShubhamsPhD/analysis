import warnings
import mdtraj as md
import numpy as np
from analysis.smoothing import savitzky_golay
from scipy.signal import find_peaks

__all__ = ["calc_peaks", "calc_height"]


def calc_peaks(z, z_range, weights=None, n_layers=0, window=41, smooth=True, **kwargs):
    """ Calculate the locations of peaks in 1-D mass density
    Calculates a mass-weighted density histogram along the z-dimension

    Parameters:
    -----------
    z : numpy.array
        1-D points
    z_range : tuple
        The [min, max] z to use for creating the histogram
    n_layers : int,
        the expected number of peaks to find.
    window : int, default=41
        Window size for Savizky-Golay smoothing. Must be odd, positive
    smooth : bool, default=True
        Smooth the histogram before finding peaks. This is useful when
        we have noisy data.
    **kwargs
        Keyword arguments to pass to the scipy.signal.find_peaks()
        function.

    Returns:
    --------
    peaks : list
        list of z-coordinates at which there are peaks in the mass
        density histogram
    """

    # Create histogram
    if weights is None:
        weights = np.ones_like(z)

    hist, edges = np.histogram(
        z, weights=weights, range=z_range, bins=400
    )
    bins = (edges[1:] + edges[:-1]) * 0.5

    # Smoothing via Savitzky-Golay
    if smooth:
        hist = savitzky_golay(hist, window, 5)

    # Gets peak indices
    # Prominance: https://en.wikipedia.org/wiki/Topographic_prominence
    if "prominence" not in kwargs:
        kwargs["prominence"] = np.max(hist) * 0.1
    if "distance" not in kwargs:
        kwargs["distance"] = 20
    if "threshold" not in kwargs:
        kwargs["threshold"] = [0, n_layers]
    peaks, _ = find_peaks(hist, **kwargs)
    peaks = np.sort(peaks)
    peaks = bins[peaks]
"""
    # Warns if there is an unequal number of peaks and layers
    if len(peaks) != n_layers:
        warnings.warn(
            "There is an unequal number of peaks "
            + "({}) and layers ({})".format(len(peaks), n_layers)
        )
        # remove the last few peaks if there are too many
        if len(peaks) > n_layers:
            peaks = peaks[:n_layers]
        # adds peaks via linear interpolation if there are too few
        else:
            for _ in range(n_layers - len(peaks)):
                try:
                    np.append(peaks, 2 * peaks[-1] - peaks[-2])
                except IndexError:
                    np.append(peaks, peaks[-1])
"""

    zmin = np.min(z)
    zmax = np.max(z)

    # Check if the number of detected peaks matches the expected number
    if len(peaks) != n_layers:
        if len(peaks) > n_layers:
            peaks = peaks[:n_layers]

        elif len(peaks) < n_layers:
            if len(peaks) == 0:
                # If no peaks are detected, evenly space the expected peaks
                peaks = np.linspace(zmin, zmax, n_layers)


            elif len(peaks) == 1:
                if n_layers == 2:
                    # For 1 detected peak and 2 expected layers, interpolate
                    predicted_peaks = np.array([peaks[0], peaks[0]])
                    peaks = np.sort(predicted_peaks)
                elif n_layers > 2:
                    # For more than 2 layers, linearly interpolate
                    predicted_peaks = np.linspace(zmin, zmax, n_layers)
                    closest_idx = np.argmin(np.abs(predicted_peaks - peaks[0]))
                    predicted_peaks[closest_idx] = peaks[0]
                    peaks = np.sort(predicted_peaks)

            elif len(peaks) > 2:
                # If more than 2 peaks are detected, we can linearly interpolate the expected number of layers
                predicted_peaks = np.linspace(peaks[0], peaks[-1], n_layers)
                peaks = np.unique(np.concatenate([peaks, predicted_peaks]))[:n_layers]
    return peaks


def calc_height(frame, atoms, window=41):
    """ Calculate the height of layers in frame
    Obtains peak locations the calc_peaks function and takes the
    difference in adjacent peak locations to get the heights.

    Parameters:
    -----------
    frame : analysis.Frame
        The frame to be analyzed
    atoms : list
        A list of atom indices. These are used to create the
        mass density histogram
    window : int, default=41
        Window size for Savizky-Golay smoothing. Must be odd, positive

    Returns:
    --------
    height : list
        list of heights for each layer (see above for n_layers)
    """

    atoms = np.array(atoms)

    # Heuristic for getting the n_layers from n_leaflets
    n_layers = int(frame.n_leaflets / 2 + 1)
    box_length = frame.unitcell_lengths[2]

    # Collect centered z coordinates and box dimensions
    z = frame.xyz[atoms, 2].reshape(-1) - np.mean(frame.xyz[atoms, 2])
    z_range = [-box_length * 0.5 - 0.01, box_length * 0.5 + 0.01]

    # Get weighting for histogram
    weights=frame.masses.take(atoms)

    peaks = calc_peaks(z, z_range, weights=weights,
                       n_layers=n_layers, window=window)
    peaks = np.sort(peaks)
    height = peaks[1:] - peaks[:-1]
    return height
