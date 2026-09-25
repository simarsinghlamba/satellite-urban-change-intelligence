import numpy as np

from processing.indices import calculate_ndvi
from processing.change_detection import detect_change


def run_change_detection(
    nir_before,
    red_before,
    nir_after,
    red_after,
    threshold=0.2
):
    """
    Run the complete urban-change detection pipeline.

    Parameters
    ----------
    nir_before : numpy.ndarray
        NIR band from the earlier satellite image.

    red_before : numpy.ndarray
        Red band from the earlier satellite image.

    nir_after : numpy.ndarray
        NIR band from the later satellite image.

    red_after : numpy.ndarray
        Red band from the later satellite image.

    threshold : float
        Minimum NDVI difference considered significant.

    Returns
    -------
    dict
        Contains the before/after NDVI maps and binary change mask.
    """

    # Validate inputs
    arrays = [
        nir_before,
        red_before,
        nir_after,
        red_after
    ]

    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise TypeError("All image bands must be NumPy arrays.")

    if not (
        nir_before.shape == red_before.shape
        == nir_after.shape == red_after.shape
    ):
        raise ValueError("All input bands must have the same shape.")

    # Step 1: Calculate NDVI for the earlier image
    ndvi_before = calculate_ndvi(
        nir_before,
        red_before
    )

    # Step 2: Calculate NDVI for the later image
    ndvi_after = calculate_ndvi(
        nir_after,
        red_after
    )

    # Step 3: Detect significant change
    change_mask = detect_change(
        ndvi_before,
        ndvi_after,
        threshold=threshold
    )

    # Step 4: Return pipeline results
    return {
        "ndvi_before": ndvi_before,
        "ndvi_after": ndvi_after,
        "change_mask": change_mask
    }