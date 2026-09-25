import numpy as np


def detect_change(index_before, index_after, threshold=0.2):
    """
    Detect significant change between two spectral-index images.

    Parameters
    ----------
    index_before : numpy.ndarray
        Spectral index from the earlier image.
    index_after : numpy.ndarray
        Spectral index from the later image.
    threshold : float
        Minimum absolute difference considered significant.

    Returns
    -------
    numpy.ndarray
        Binary change mask.
    """

    difference = np.abs(
        index_after.astype(np.float32)
        - index_before.astype(np.float32)
    )

    change_mask = difference >= threshold

    return change_mask.astype(np.uint8)


def calculate_change_percentage(change_mask):
    """
    Calculate the percentage of pixels classified as changed.
    """

    total_pixels = change_mask.size

    if total_pixels == 0:
        return 0.0

    changed_pixels = np.count_nonzero(change_mask)

    return (changed_pixels / total_pixels) * 100