import numpy as np


def calculate_ndvi(nir, red):
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    nir = nir.astype(np.float32)
    red = red.astype(np.float32)

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir, dtype=np.float32),
        where=denominator != 0
    )

    return ndvi


def calculate_ndbi(swir, nir):
    """
    Calculate Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """

    swir = swir.astype(np.float32)
    nir = nir.astype(np.float32)

    denominator = swir + nir

    ndbi = np.divide(
        swir - nir,
        denominator,
        out=np.zeros_like(swir, dtype=np.float32),
        where=denominator != 0
    )

    return ndbi