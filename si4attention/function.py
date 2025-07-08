"""Module for providing an interface to conduct inference."""

from typing import Literal

import torch
from torch import nn

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)

from si4attention.abstract import SupportsExtractWeights
from si4attention.batch_adaptive import BatchAdaptiveGridInference
from si4attention.grid import AdaptiveGrid
from si4attention.hypothesis import Background, Hypothesis, Neighbor, Reference
from si4attention.inference import GridBasedSelectiveInference


def test_attention_map(
    model: SupportsExtractWeights,
    data: torch.Tensor,
    cov: float | torch.Tensor | None = None,
    hypothesis: Literal["background", "neighbor", "reference"] = "background",
    threshold: float = 0.6,
    radius: int | None = None,
    reference: torch.Tensor | None = None,
    min_width: float = 1e-4,
    max_width: float = 1e-1,
    batch: int = 64,
    *,
    progress: bool = False,
) -> tuple[float, float, torch.Tensor]:
    """Conduct a statistical test on the attention map of a transformer model.

    This function performs a statistical test to analyze the attention map
    of a given transformer model. It supports different hypotheses
    (`'background'`, `'neighbor'`, `'reference'`) to test different aspects
    of the attention map.

    Parameters
    ----------
    model : SupportsExtractWeights
        Any transformer model that inherits `nn.Module` and supports the `extract_weights` method.
        This method must accept an input tensor of shape
        `(b, c, h, w)` for image data or `(b, c, t)` for series data,
        and return the attention weights of all multi-head attention layers
        as a tensor of shape `(b, l, n, n)`,
        where `b` is the batch size, `l` is the number of layers,
        and `n` is the number of patches (including the class token).
    data : torch.Tensor
        Input tensor of shape `(1, 1, h, w)` for image data or `(1, 1, t)` for series data.
    cov : float | torch.Tensor | None, optional
        Covariance matrix of the input data.
        If `float`, the covariance matrix is set to the scalar times the identity matrix.
        If `torch.Tensor`, it must be a 2D tensor and represent the covariance matrix.
        If `None`, the data is assumed to be independent and identical distributed, and
        the variance is estimated from the input data. Defaults to `None`.
    hypothesis : Literal['background', 'neighbor', 'reference'], optional
        The hypothesis to test.
        Must be one of `'background'`, `'neighbor'`, `'reference'`
        Defaults to `'background'`.
    threshold : float, optional
        The threshold value to define the high attention region. Defaults to 0.6.
    radius : int | None, optional
        Only active when the hypothesis is set to `'neighbor'`.
        The radius of the neighborhood.
        If `None`, the radius is automatically determined based on the data size.
        Defaults to `None`.
    reference : torch.Tensor | None, optional
        Only active when the hypothesis is set to `'reference'`.
        The reference data, which must have the same shape and covariances as the input data.
        Defaults to `None`.
    min_width : float, optional
        The minimum width of the grid for adaptive grid search.
        Defaults to 1e-4.
    max_width : float, optional
        The maximum width of the grid for adaptive grid search.
        Defaults to 1e-1.
    batch: int, optional
        The batch size for the adaptive grid search. Defaults to 64.
    progress : bool, optional
        Whether to show the progress bar. Defaults to `False`.

    Returns
    -------
    tuple[float, float, torch.Tensor]
        A tuple containing the selective p-value, test statistic and the attention map.

    Raises
    ------
    TypeError
        If `model` is not an instance of `nn.Module`.
    ValueError
        If `data` is not a 3D tensor (series data) or a 4D tensor (image data).
    """
    if not isinstance(model, nn.Module):
        raise TypeError

    model.eval()
    model.double()

    data = data.double()

    if cov is None:
        cov = torch.var(data).double()
    if isinstance(cov, float):
        cov = torch.tensor(cov).double()

    data_size = data.shape[2]
    match len(data.shape):
        case 4:
            data_type = "image"
        case 3:
            data_type = "series"
        case _:
            raise ValueError

    hypothesis_: Hypothesis
    match hypothesis:
        case "background":
            hypothesis_ = Background(tau=threshold)
        case "neighbor":
            hypothesis_ = Neighbor(
                data_type=data_type,
                data_size=data_size,
                tau=threshold,
                radius=radius,
            )
        case "reference":
            hypothesis_ = Reference(tau=threshold)

    si: GridBasedSelectiveInference | BatchAdaptiveGridInference
    if batch > 1:
        si = BatchAdaptiveGridInference(
            data_type=data_type,
            data_size=data_size,
            hypothesis=hypothesis_,
            model=model,
            batch=batch,
            min_width=min_width,
            max_width=max_width,
            progress=progress,
        )
    else:
        si = GridBasedSelectiveInference(
            data_type=data_type,
            data_size=data_size,
            grid=AdaptiveGrid(min_width=min_width, max_width=max_width),
            hypothesis=hypothesis_,
            model=model,
            progress=progress,
        )
    si.construct_hypothesis(cov, data, reference)

    return si.selective_p_value, si.stat, si.attention_map
