"""Module for loading and processing the model for inference."""

import math
from typing import Protocol

import torch
from torch import nn

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)


class SupportsExtractWeights(Protocol):
    """Protocol for the model which supports the `extract_weights` method.

    The model must have the `extract_weights` method
    which returns the attention weights of all multi-head attention layers
    as a tensor of shape `(b, l, n, n)`, where
    `b` is the batch size, `l` is the number of layers, and `n` is the number of patches.
    """

    def extract_weights(self, input: torch.Tensor) -> torch.Tensor:
        """Extract the attention weights of all multi-head attention layers.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape
            `(b, c, h, w)` for image data or
            `(b, c, t)` for series data,
            where `b` is the batch size and `c` is the number of channels,

        Returns
        -------
        torch.Tensor
            Attention weights of all multi-head attention layers
            as a tensor of shape `(b, l, n, n)`, where
            `b` is the batch size, `l` is the number of layers, and
            `n` is the number of patches (including the class token).
        """


class Transformer:
    """Class for the transformer model to conduct inference.

    It employs the instance of the `SupportsExtractWeights` protocol.
    Based on the `extract_weights` method, it implements the `attention_map`.
    """

    def __init__(
        self,
        model: SupportsExtractWeights,
        data_type: str,
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        model : SupportsExtractWeights
            Any transformer model which supports the `extract_weights` method.
            This method must take an input tensor of shape
            `(b, c, h, w)` for image data or `(b, c, t)` for series data,
            and return the attention weights of all multi-head attention layers
            as a tensor of shape `(b, l, n, n)`,
            where `b` is the batch size, `l` is the number of layers,
            and `n` is the number of patches (including the class token).
        data_type : str
            Type of the input data. Must be 'image' or 'series'.
        """
        self.model = model
        match data_type:
            case "image":
                self.mode = "bilinear"
            case "series":
                self.mode = "linear"

    def attention_map(self, input: torch.Tensor) -> torch.Tensor:
        """Compute attention map for the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, c, h, w)` for image data or `(b, c, t)` for series data,
            where `b` is the batch size and `c` is the number of channels.

        Returns
        -------
        torch.Tensor
            Attention map as a tensor of
            shape `(b, h, w)` for image data or `(b, t)` for series data,
            where `b` is the batch size.
        """
        spatial_dim = 4
        # (b, c, h, w) -> (b, l, n, n)
        attention_scores = self.model.extract_weights(input)

        # (b, l, n, n) -> (b, l, n, n)
        weights = attention_scores + torch.eye(attention_scores.shape[2])

        # (b, l, n, n) -> (b, l, n, n)
        weights = weights / torch.sum(weights, dim=[2, 3], keepdim=True)

        # (b, l, n, n) -> (b, n, n)
        v = weights[:, -1, :, :]
        for n in range(1, weights.shape[1]):
            v = torch.matmul(v, weights[:, -(n + 1), :, :])

        # (b, n, n) -> (b, num_patches)
        weights = v[:, 0, 1:]

        size = weights.shape[1]
        shape: tuple[int, ...] = (
            (-1, 1, math.isqrt(size), math.isqrt(size))
            if len(input.shape) == spatial_dim
            else (-1, 1, size)
        )

        # (b, num_patches) -> (b, 1, h', w') or (b, 1, t')
        weights = weights.reshape(shape)
        # (b, 1, h', w') -> (b, 1, h, w) or (b, 1, t') -> (b, 1, t)
        weights_: torch.Tensor = nn.Upsample(
            size=input.shape[2:],
            mode=self.mode,
            align_corners=True,
        )(weights)
        # (b, 1, h, w) -> (b, h, w) or (b, 1, t) -> (b, t)
        weights_ = weights_.squeeze(1)

        shape_: tuple[int, ...] = (
            (-1, 1, 1) if len(weights_.shape) == spatial_dim - 1 else (-1, 1)
        )
        min_values = weights_.flatten(1).min(dim=1).values.reshape(shape_)
        max_values = weights_.flatten(1).max(dim=1).values.reshape(shape_)

        return (weights_ - min_values) / (max_values - min_values)
