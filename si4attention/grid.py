"""Module for grid search."""

import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)


from si4attention.abstract import Transformer
from si4attention.hypothesis import Hypothesis


class Grid:
    """An abstract class for grid."""

    def __init__(self) -> None:
        """Initialize the class."""

    def set_test_statistic(self, stat: float) -> None:
        """Set the observed test statistic.

        Parameters
        ----------
        stat : float
            The observed test statistic.
        """
        self.stat = stat

    def evaluate_grid(
        self,
        model: Transformer,
        hypothesis: Hypothesis,
        a: torch.Tensor,
        b: torch.Tensor,
        z: float,
        shape: tuple[int, ...],
    ) -> tuple[bool, float]:
        """Evaluate a given grid point.

        Parameters
        ----------
        model : Transformer
            The transformer model.
        hypothesis : Hypothesis
            The hypothesis to test.
        a : torch.Tensor
            The vector specifying the search space.
        b : torch.Tensor
            The vector specifying the search direction.
        z : float
            The grid point to evaluate.
        shape : tuple[int, ...]
            The shape of the input data.

        Returns
        -------
        tuple[bool, float]
            Boolean value indicating if the grid point belongs to the
            truncated interval and the width of the next grid point.
        """
        raise NotImplementedError


class AdaptiveGrid(Grid):
    """Class for the adaptive grid."""

    def __init__(self, min_width: float = 1e-4, max_width: float = 1e-1) -> None:
        """Initialize the adaptive grid.

        Parameters
        ----------
        min_width : float
            The minimum width of the grid. Defaults to 1e-4.
        max_width : float
            The maximum width of the grid. Defaults to 1e-1.
        """
        self.min_width = min_width
        self.max_width = max_width

    def evaluate_grid(
        self,
        model: Transformer,
        hypothesis: Hypothesis,
        a: torch.Tensor,
        b: torch.Tensor,
        z: float,
        shape: tuple[int, ...],
    ) -> tuple[bool, float]:
        """Evaluate a given grid point.

        Parameters
        ----------
        model : Transformer
            The transformer model.
        hypothesis : Hypothesis
            The hypothesis to test.
        a : torch.Tensor
            The vector specifying the search space.
        b : torch.Tensor
            The vector specifying the search direction.
        z : float
            The grid point to evaluate.
        shape : tuple[int, ...]
            The shape of the input data.

        Returns
        -------
        tuple[bool, float]
            Boolean value indicating if the grid point belongs to the
            truncated interval and the width of the next grid point.
        """
        thr = 1e-1
        input = (a + b * z).reshape(shape)
        if abs(z - self.stat) < thr:
            f_value = hypothesis.convert_to_value(model.attention_map(input)).flatten()
            flag = bool((f_value <= 0.0).all().int().detach().item())
            step = (f_value / 1.0).abs().min() if flag else (f_value / 1.0).max()
        else:
            attention_map, attention_map_jvp = torch.func.jvp(
                func=model.attention_map,
                primals=(input,),
                tangents=(b.reshape(shape),),
            )
            f_value = hypothesis.convert_to_value(attention_map).flatten()
            f_derivative = hypothesis.convert_to_derivative(attention_map_jvp).flatten()
            flag = bool((f_value <= 0.0).all().int().detach().item())

            candidate = -f_value / (10.0 * f_derivative)
            candidate = torch.where(candidate < 0.0, 10.0, candidate)
            if flag:
                step = candidate[f_value <= 0.0].min()
            else:
                step = candidate[f_value >= 0.0].max()

        return flag, torch.clamp(step, self.min_width, self.max_width).detach().item()


class FixedGrid(Grid):
    """Class for the fixed grid."""

    def __init__(self, width: float = 1e-3) -> None:
        """Initialize the fixed grid.

        Parameters
        ----------
        width : float
            The width of the grid. Defaults to 1e-3.
        """
        self.width = width

    def evaluate_grid(
        self,
        model: Transformer,
        hypothesis: Hypothesis,
        a: torch.Tensor,
        b: torch.Tensor,
        z: float,
        shape: tuple[int, ...],
    ) -> tuple[bool, float]:
        """Evaluate a given grid point.

        Parameters
        ----------
        model : Transformer
            The transformer model.
        hypothesis : Hypothesis
            The hypothesis to test.
        a : torch.Tensor
            The vector specifying the search space.
        b : torch.Tensor
            The vector specifying the search direction.
        z : float
            The grid point to evaluate.
        shape : tuple[int, ...]
            The shape of the input data.

        Returns
        -------
        tuple[bool, float]
            Boolean value indicating if the grid point belongs to the
            truncated interval and the width of the next grid point.
        """
        input = (a + b * z).reshape(shape)
        f_value = hypothesis.convert_to_value(model.attention_map(input)).flatten()
        flag = bool((f_value <= 0.0).all().int().detach().item())
        return flag, self.width


class CombinationGrid(Grid):
    """Class for the combination grid."""

    def __init__(self, min_width: float = 1e-4, max_width: float = 1e-2) -> None:
        """Initialize the combination grid.

        Parameters
        ----------
        min_width : float
            The minimum width of the grid. Defaults to 1e-4.
        max_width : float
            The maximum width of the grid. Defaults to 1e-2.
        """
        self.min_width = min_width
        self.max_width = max_width

    def evaluate_grid(
        self,
        model: Transformer,
        hypothesis: Hypothesis,
        a: torch.Tensor,
        b: torch.Tensor,
        z: float,
        shape: tuple[int, ...],
    ) -> tuple[bool, float]:
        """Evaluate a given grid point.

        Parameters
        ----------
        model : Transformer
            The transformer model.
        hypothesis : Hypothesis
            The hypothesis to test.
        a : torch.Tensor
            The vector specifying the search space.
        b : torch.Tensor
            The vector specifying the search direction.
        z : float
            The grid point to evaluate.
        shape : tuple[int, ...]
            The shape of the input data.

        Returns
        -------
        tuple[bool, float]
            Boolean value indicating if the grid point belongs to the
            truncated interval and the width of the next grid point.
        """
        thr = 1e-1
        input = (a + b * z).reshape(shape)
        f_value = hypothesis.convert_to_value(model.attention_map(input)).flatten()
        flag = bool((f_value <= 0.0).all().int().detach().item())
        step = self.min_width if abs(z - self.stat) < thr else self.max_width
        return flag, step
