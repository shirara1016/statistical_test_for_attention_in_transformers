"""Module for training the transformer model."""

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from experiment.transformer import TransformerModel


@dataclass
class TrainingConfig:
    """Dataclass for training configuration.

    Parameters
    ----------
    learning_rate : float
        Learning rate for training. Defaults to 0.0001.
    validation_rate : float
        Rate for validation. Defaults to 0.2.
    batch_size : int
        Batch size for training. Defaults to 256.
    epochs : int
        Number of epochs for training. Defaults to 100.
    """

    learning_rate: float = 0.0001
    validation_rate: float = 0.2
    batch_size: int = 256
    epochs: int = 100


def train_model(
    model: TransformerModel,
    x: torch.Tensor,
    y: torch.Tensor,
    config: TrainingConfig | None = None,
) -> TransformerModel:
    """Train the transformer model.

    Parameters
    ----------
    model : Transformer
        The transformer model.
    x : torch.Tensor
        Input tensor of shape `(b, c, h, w)` for image data or `(b, c, t)` for series data.
    y : torch.Tensor
        Target tensor of shape (b, k), where `k` is the number of classes.
    config : TrainingConfig, optional
        Training configuration. If set to None, default configuration is used.
        Defaults to None.
    """
    if config is None:
        config = TrainingConfig()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(x.to(device), y.to(device))
    val_size = int(len(dataset) * config.validation_rate)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            output = model(x_batch)
            loss = loss_fn(output, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                output = model(x_batch)
                val_loss += loss_fn(output, y_batch).item()
        print(  # noqa: T201
            f"Epoch [{epoch+1}/{config.epochs}], Validation Loss: {val_loss / len(val_loader):.4f}",
        )
    return model
