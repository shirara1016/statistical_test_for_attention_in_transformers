"""Module for defining Vision Transformer and Time Transformer."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    """Dataclass for transformer configuration.

    Attributes
    ----------
    data_type : str
        Type of data. Must be one of 'image', 'series'.
    data_size : int
        Size of data.
    patch_size : int
        Patch size for embedding.
    num_layers : int
        Number of encoder blocks. Defaults to 4.
    num_heads : int
        Number of attention heads. Defaults to 4.
    embed_dim : int
        Embedding dimension. Defaults to 64.
    mlp_dim : int
        Dimension of hidden states in the multilayer perceptron. Defaults to 256.
    num_classes : int
        Number of classes. Defaults to 1.
    dropout : float
        Dropout rate. Defaults to 0.1.
    """

    data_type: str
    data_size: int
    patch_size: int
    num_layers: int = 8
    num_heads: int = 4
    embed_dim: int = 64
    mlp_dim: int = 256
    num_classes: int = 1
    dropout: float = 0.1


def setup_transformer_config(
    data_type: str,
    data_size: int,
    architecture_size: str,
) -> TransformerConfig:
    """Set up the configuration for the transformer.

    Parameters
    ----------
    data_type : str
        Type of data. Must be one of 'image', 'series'.
    data_size : int
        Size of data. Must be one of 8, 16, 32, 64 for image data
        and 64, 128, 256, 512 for series data.
    architecture_size : str
        Size of architecture. Must be one of 'small', 'base', 'large', 'huge'.

    Returns
    -------
    TransformerConfig
        Default configuration for the transformer.
    """
    match data_type:
        case "image":
            patch_size = max(2, data_size // 8)
        case "series":
            patch_size = data_size // 32
    partial_config_dict = {
        "small": {
            "embed_dim": 32,
            "mlp_dim": 128,
            "num_layers": 4,
            "num_heads": 2,
        },
        "base": {
            "embed_dim": 64,
            "mlp_dim": 256,
            "num_layers": 8,
            "num_heads": 4,
        },
        "large": {
            "embed_dim": 128,
            "mlp_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
        },
        "huge": {
            "embed_dim": 256,
            "mlp_dim": 1024,
            "num_layers": 16,
            "num_heads": 16,
        },
    }
    return TransformerConfig(
        data_type=data_type,
        data_size=data_size,
        patch_size=patch_size,
        **partial_config_dict[architecture_size],
    )


class EmbeddingLayer(nn.Module):
    """Abstract class for embedding layer."""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
    ) -> None:
        """Initialize layer.

        Parameters
        ----------
        patch_size : int
            Patch size for embedding.
        embed_dim : int
            Embedding dimension.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches: int

        self.patch_embedding_layer: nn.Module

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, c, h, w)` for image data
            or `(b, c, t)` for series data, where `b` is the batch size
            and `c` is the number of channels.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(b, n, d)`, where `b` is the batch size,
            `n` is the number of patches (including the class token)
            and `d` is the embedding dimension.
        """
        # (b, c, h, w) -> (b, d, h/p, w/p) or (b, c, t) -> (b, d, 2t/p),
        # where p is patch_size
        embed: torch.Tensor = self.patch_embedding_layer(input)

        # (b, d, h/p, w/p) or (b, d, 2t/p) -> (b, num_patches, d)
        embed = embed.flatten(2).transpose(1, 2)

        # (b, num_pathes, d) -> (b, num_pathes + 1, d),
        # where cls_token is broadcasted to (b, 1, d)
        embed = torch.cat(
            [
                self.cls_token.expand(embed.shape[0], 1, self.embed_dim),
                embed,
            ],
            dim=1,
        )

        # (b, n, d) -> (b, n, d)
        return embed + self.position_embedding


class ImageEmbeddingLayer(EmbeddingLayer):
    """Class for image embedding layer."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
    ) -> None:
        """Initialize layer.

        Parameters
        ----------
        image_size : int
            Image size.
        patch_size : int
            Patch size for embedding.
        embed_dim : int
            Embedding dimension.
        """
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        super().__init__(patch_size, embed_dim)

        self.patch_embedding_layer = nn.Conv2d(
            in_channels=1,  # Assuming 1 channels for image
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


class SeriesEmbeddingLayer(EmbeddingLayer):
    """Class for series embedding layer."""

    def __init__(
        self,
        series_size: int,
        patch_size: int,
        embed_dim: int,
    ) -> None:
        """Initialize layer.

        Parameters
        ----------
        series_size : int
            Series size.
        patch_size : int
            Patch size for embedding.
        embed_dim : int
            Embedding dimension.
        """
        self.series_size = series_size
        self.num_patches = 2 * series_size // patch_size
        super().__init__(patch_size, embed_dim)

        self.patch_embedding_layer = nn.Conv1d(
            in_channels=1,  # Assuming 1 channels for series
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size // 2,
            padding=self.patch_size // 4,
        )


class MlpBlock(nn.Module):
    """Class for multiayer perceptron block."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Initialize block.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        hidden_dim : int
            Hidden dimension.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        self.dense_layer_1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.dense_layer_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout_layer_2 = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, n, d)`, where `b` is the batch size,
            `n` is the number of patches (including the class token)
            and `d` is the embedding dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(b, n, d)`, which is equal to the input shape.
        """
        output = self.dense_layer_1(input)
        output = nn.functional.gelu(output)
        output = self.dropout_layer_1(output)
        output = self.dense_layer_2(output)
        return self.dropout_layer_2(output)


class EncoderBlock(nn.Module):
    """Class for encoder block."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """Initialize block.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        mlp_dim : int
            Dimension of hidden states in the multilayer perceptron.
        num_heads : int
            Number of attention heads.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mhsa_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MlpBlock(embed_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, n, d)`, where `b` is the batch size,
            `n` is the number of patches (including the class token)
            and `d` is the embedding dimension.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output tensor of shape `(b, n, d)`, which is equal to the input shape, and
            attention weights of shape `(b, n, n)`, where `b` is the batch size,
            `n` is the number of patches (including the class token) and `d` is the embedding dimension.
        """
        output = self.layer_norm_1(input)
        output, attention_weights = self.mhsa_layer(
            query=output,
            key=output,
            value=output,
            need_weights=True,
        )
        output = output + input

        output = self.mlp(self.layer_norm_2(output)) + output
        return output, attention_weights


class TransformerModel(nn.Module):
    """Abstract class for transformer model."""

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        """Initialize model.

        config : TransformerConfig
            Configuration for the transformer.
        """
        super().__init__()

        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.mlp_dim = config.mlp_dim
        self.num_classes = config.num_classes
        self.dropout = config.dropout

        self.embedding_layer: EmbeddingLayer
        match config.data_type:
            case "image":
                self.embedding_layer = ImageEmbeddingLayer(
                    image_size=config.data_size,
                    patch_size=self.patch_size,
                    embed_dim=self.embed_dim,
                )
            case "series":
                self.embedding_layer = SeriesEmbeddingLayer(
                    series_size=config.data_size,
                    patch_size=self.patch_size,
                    embed_dim=self.embed_dim,
                )

        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ],
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.predictor = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, c, h, w)` for image data
            or `(b, c, t)` for series data, where `b` is the batch size
            and `c` is the number of channels.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(b, k)`,
            where `b` is the batch size and `k` is the number of classes.
        """
        # (b, c, h, w) or (b, c, t) -> (b, n, d)
        embed = self.embedding_layer(input)

        # (b, n, d) -> (b, n, d)
        for encoder_block in self.encoder_blocks:
            embed, _ = encoder_block(embed)

        # (b, n, d) -> (b, d)
        output = self.layer_norm(embed[:, 0, :])

        # (b, d) -> (b, k)
        output = self.predictor(output)
        return nn.functional.sigmoid(output)

    def extract_weights(self, input: torch.Tensor) -> torch.Tensor:
        """Extract attention weights.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(b, c, h, w)` for image data
            or `(b, c, t)` for series data, where `b` is the batch size
            and `c` is the number of channels.

        Returns
        -------
        torch.Tensor
            Attention weights of all multi-head attention layers
            as a tensor of shape `(b, l, n, n)`, where `b` is the batch size,
            `l` is the number of layers, `n` is the number of patches (including the class token).
        """
        # (b, c, h, w) or (b, c, t) -> (b, n, d)
        embed = self.embedding_layer(input)

        # (b, n, d) -> (b, n, d)
        attentions_ = []
        for encoder_block in self.encoder_blocks:
            embed, attention = encoder_block(embed)
            attentions_.append(attention)
        return torch.stack(attentions_, dim=1)
