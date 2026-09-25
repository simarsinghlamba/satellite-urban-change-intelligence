import numpy as np

from processing.indices import calculate_ndvi, calculate_ndbi
from processing.change_detection import (
    detect_change,
    calculate_change_percentage,
)
from processing.io import validate_image, normalize_image


def test_ndvi():
    nir = np.array([[0.8, 0.6]])
    red = np.array([[0.2, 0.3]])

    result = calculate_ndvi(nir, red)

    assert result.shape == nir.shape
    assert np.all(result <= 1)
    assert np.all(result >= -1)


def test_ndbi():
    swir = np.array([[0.7, 0.8]])
    nir = np.array([[0.3, 0.4]])

    result = calculate_ndbi(swir, nir)

    assert result.shape == swir.shape


def test_change_detection():
    before = np.array([[0.1, 0.2, 0.3]])
    after = np.array([[0.1, 0.5, 0.3]])

    mask = detect_change(before, after, threshold=0.2)

    assert mask.shape == before.shape
    assert mask[0, 1] == 1


def test_change_percentage():
    mask = np.array([[1, 0], [0, 1]])

    percentage = calculate_change_percentage(mask)

    assert percentage == 50.0


def test_image_validation():
    image = np.ones((10, 10))

    assert validate_image(image) is True


def test_normalization():
    image = np.array([[0, 5, 10]])

    normalized = normalize_image(image)

    assert normalized.min() == 0
    assert normalized.max() == 1