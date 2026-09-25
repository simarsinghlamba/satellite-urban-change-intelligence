import numpy as np

from pipeline.run_pipeline import run_change_detection


def test_change_detection_pipeline():

    nir_before = np.array([
        [0.8, 0.8],
        [0.8, 0.8]
    ])

    red_before = np.array([
        [0.2, 0.2],
        [0.2, 0.2]
    ])

    nir_after = np.array([
        [0.2, 0.8],
        [0.8, 0.2]
    ])

    red_after = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])

    result = run_change_detection(
        nir_before,
        red_before,
        nir_after,
        red_after
    )

    assert "ndvi_before" in result
    assert "ndvi_after" in result
    assert "change_mask" in result

    assert result["ndvi_before"].shape == (2, 2)
    assert result["ndvi_after"].shape == (2, 2)
    assert result["change_mask"].shape == (2, 2)