"""Module for generating synthetic data for experiments."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Config:
    """Configuration for experiments.

    Attributes
    ----------
    method : str
        Method to experiment. Must be one of 'adaptive', 'naive', 'bonferroni',
        'permutation', 'combination', 'fixed'. Defaults to 'adaptive'.
    data_type : str
        Type of data. Must be one of 'image', 'series'. Defaults to 'image'.
    noise_type : str
        Type of noise. Must be one of 'iid', 'corr', 'estimated',
        'skewnorm', 'exponnorm', 'gennormsteep', 'gennormflat', 't'.
        Defaults to 'iid'.
    data_size : int
        Size of data. Must be one of 8, 16, 32, 64 for image data
        and 64, 128, 256, 512 for series data.
        If set to -1, it is set to 16 for image and 256 for series. Defaults to -1.
    architecture_size : str
        Size of architecture. Must be one of 'small', 'base', 'large', 'huge'.
        Defaults to 'base'.
    signal : float
        Signal strength for the data. Defaults to 0.0.
    hypothesis : str
        Hypothesis to test. Must be one of 'background', 'neighbor' 'reference'.
        Defaults to 'background'.
    deviation_from_gaussian : float
        Deviation from Gaussian noise. Defaults to 0.0.
    """

    method: str = "adaptive"
    data_type: str = "image"
    noise_type: str = "iid"
    data_size: int = -1
    architecture_size: str = "base"
    signal: float = 0.0
    hypothesis: str = "background"
    deviation_from_gaussian: float = 0.0

    def __post_init__(self) -> None:
        """Initialize the data size if not provided."""
        if self.data_size == -1:
            match self.data_type:
                case "image":
                    self.data_size = 16
                case "series":
                    self.data_size = 256

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return "_".join(str(value) for value in asdict(self).values())

    @property
    def unique_integer(self) -> int:
        """Return the unique integer of the configuration excluding the method for seeding."""
        hash_value = hashlib.sha512(str(self).encode()).hexdigest()
        return int(hash_value, 16)


def create_cov_matrix(data_type: str, data_size: int, rho: float = 0.5) -> np.ndarray:
    """Create a covariance matrix.

    It creates a covariance matrix for a one-order autoregressive process.

    Parameters
    ----------
    data_type : str
        Type of data. Must be one of 'image', 'series'.
    data_size : int
        Size of the data.
    rho : float
        Correlation coefficient. Defaults to 0.5.

    Returns
    -------
    np.ndarray
        Covariance matrix.
    """
    matrix = np.power(rho, np.abs(np.arange(data_size) - np.arange(data_size)[:, None]))
    match data_type:
        case "image":
            return np.kron(matrix, matrix)
        case "series":
            return matrix
        case _:
            raise ValueError


def _sample_normal(
    rng: np.random.Generator,
    numel: int,
    cov: float | np.ndarray,
    num: int,
) -> np.ndarray:
    """Sample from a Gaussian distribution.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    numel : int
        Number of elements in the sample.
    cov : float | np.ndarray
        Variance scalar or covariance matrix.
    num : int
        Number of samples to generate.

    Returns
    -------
    np.ndarray
        Sampled data, shape (num, numel).
    """
    cov = np.array(cov)
    if cov.shape == ():
        return rng.normal(0, cov, (num, numel))
    return rng.multivariate_normal(np.zeros(numel), cov, num, method="cholesky")


def make_image(
    rng: np.random.Generator,
    image_size: int,
    cov: float | np.ndarray = 1.0,
    signal: float | tuple[float, float] = 0.0,
    num: int = 1,
) -> np.ndarray:
    """Make multiple images.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    image_size : int
        Size of the image.
    cov : float | np.ndarray
        Variance scalar or covariance matrix. Defaults to 1.0.
    signal : float | tuple[float, float]
        Signal strength for the image. If set to a tuple, it generates signals
        uniformly from the range for each image. Defaults to 0.0.
    num : int
        Number of images to generate. Defaults to 1.

    Returns
    -------
    np.ndarray
        Generated images, shape (num, 1, image_size, image_size).
    """
    x = _sample_normal(rng, image_size * image_size, cov, num)
    x = np.reshape(x, (num, 1, image_size, image_size))

    if signal == 0.0:
        return x
    if isinstance(signal, float):
        signals = signal * np.ones(num)
    else:
        signals = rng.uniform(signal[0], signal[1], num)
    a = image_size // 3
    for i in range(num):
        abnormal_x = rng.integers(0, image_size - a)
        abnormal_y = rng.integers(0, image_size - a)
        x[i, 0, abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a] += signals[i]
    return x


def make_series(
    rng: np.random.Generator,
    series_size: int,
    cov: float | np.ndarray = 1.0,
    signal: float = 0.0,
    num: int = 1,
) -> np.ndarray:
    """Make multiple series.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    series_size : int
        Size of the series.
    cov : float | np.ndarray
        Variance scalar or covariance matrix. Defaults to 1.0.
    signal : float | tuple[float, float]
        Signal strength for the series. If set to a tuple, it generates signals
        uniformly from the range for each series. Defaults to 0.0.
    num : int
        Number of series to generate. Defaults to 1.

    Returns
    -------
    np.ndarray
        Generated series, shape (num, 1, series_size).
    """
    x = _sample_normal(rng, series_size, cov, num)
    x = np.reshape(x, (num, 1, series_size))

    if signal == 0.0:
        return x
    if isinstance(signal, float):
        signals = signal * np.ones(num)
    else:
        signals = rng.uniform(signal[0], signal[1], num)
    a = series_size // 10
    for i in range(num):
        abnormal_x = rng.integers(0, series_size - a)
        x[i, 0, abnormal_x : abnormal_x + a] += signals[i]
    return x
