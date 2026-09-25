import numpy as np


def validate_image(image):
    """
    Validate that an image is a non-empty NumPy array.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    return True


def normalize_image(image):
    """
    Normalize image values to the range 0-1.
    """

    image = image.astype(np.float32)

    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_value) / (max_value - min_value)