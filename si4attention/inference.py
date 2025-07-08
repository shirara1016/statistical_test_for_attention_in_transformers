"""Module for inference on attentions in transformers."""

import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)

import numpy as np
from scipy.stats import norm  # type: ignore[import]
from sicore import RealSubset, truncated_cdf  # type: ignore[import]

from si4attention.abstract import SupportsExtractWeights, Transformer
from si4attention.grid import Grid
from si4attention.hypothesis import Hypothesis


class GridBasedSelectiveInference:
    """Class for grid based selective inference for attentions in transformers."""

    def __init__(
        self,
        data_type: str,
        data_size: int,
        grid: Grid,
        hypothesis: Hypothesis,
        model: SupportsExtractWeights,
        *,
        progress: bool = False,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        data_type : str
            Type of data. Must be 'image' or 'series'.
        data_size : int
            Size of data.
        grid : Grid
            The grid to search.
        hypothesis : Hypothesis
            The hypothesis to test.
        model : Transformer
            The transformer model.
        progress : bool, optional
            Whether to show progress bar. Defaults to `False`.
        """
        self.data_type = data_type
        self.data_size = data_size
        self.grid = grid
        self.hypothesis = hypothesis
        self.model = Transformer(model, data_type)
        self.progress = progress

        if self.progress:
            from tqdm import tqdm  # type: ignore[import]

            self.bar = tqdm(
                total=100,
                desc="Progress of grid search",
                unit="%",
                bar_format="{desc}: {percentage:3.2f}{unit}|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

        self.shape: tuple[int, ...]
        match self.data_type:
            case "image":
                self.shape = (-1, 1, self.data_size, self.data_size)
            case "series":
                self.shape = (-1, 1, self.data_size)

        self.truncated: RealSubset
        self.searched: RealSubset

    def construct_hypothesis(
        self,
        cov: torch.Tensor,
        data: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> None:
        """Construct the hypothesis based on input data.

        Parameters
        ----------
        cov : torch.Tensor
            The covariance matrix or scalar specifying the variance of the input data.
        data : torch.Tensor
            The input data.
        reference : torch.Tensor, optional
            The reference data if any. Defaults to None.
        """
        self.attention_map = self.model.attention_map(data)
        self.stat, self.a, self.b = self.hypothesis.construct_hypothesis(
            cov,
            self.attention_map,
            data,
            reference,
        )
        if self.hypothesis.name == "reference":
            self.a, self.b = self.a[: len(self.a) // 2], self.b[: len(self.b) // 2]
        self.grid.set_test_statistic(self.stat)

        self.cov = cov
        self.data = data
        self.reference = reference

    @property
    def selective_p_value(self) -> float:
        """Selective p-value."""
        self._cut_grid()
        self._detect_change_points()
        self._finer_around_change_points()
        return 1.0 - truncated_cdf(
            norm(),
            self.stat,
            self.truncated,
            absolute=True,
        )

    @property
    def naive_p_value(self) -> float:
        """Naive p-value."""
        return 2.0 * norm.cdf(-abs(self.stat))

    @property
    def bonferroni_p_value(self) -> float:
        """Bonferroni p-value."""
        log_num_comparison = np.abs(np.prod(self.shape)) * np.log(2.0)
        log_naive_p_value = np.log(2.0) + norm.logcdf(-abs(self.stat))
        log_bonferroni_p_value = np.clip(
            log_naive_p_value + log_num_comparison,
            -np.inf,
            0.0,
        )
        return np.exp(log_bonferroni_p_value)

    @property
    def permutation_p_value(self) -> float:
        """Permutation p-value."""
        counter, num_iter = 0, 1000
        data_size = np.abs(np.prod(self.shape)).item()
        for _ in range(num_iter):
            if self.reference is not None:
                perm_flatten = torch.cat([self.data, self.reference]).flatten()[
                    torch.randperm(2 * data_size)
                ]
                perm_data = perm_flatten[:data_size].reshape(self.shape)
                perm_reference = perm_flatten[data_size:].reshape(self.shape)
            else:
                perm_data = self.data.flatten()[torch.randperm(data_size)].reshape(
                    self.shape,
                )
                perm_reference = None
            attention_map = self.model.attention_map(perm_data)
            stat = self.hypothesis.compute_test_statistic(
                self.cov,
                attention_map,
                perm_data,
                perm_reference,
                is_observed=False,
            )
            if abs(stat) >= abs(self.stat):
                counter += 1
        return counter / num_iter

    def _cut_grid(self) -> None:
        """Cut the grid in the search space."""
        left = -abs(self.stat) - 10.0
        right = abs(self.stat) + 10.0
        z_list, flag_list = [], []
        is_searched = False

        z = left
        while z < right:
            if not is_searched and z > self.stat:
                z, is_searched = self.stat, True
            flag, step = self.grid.evaluate_grid(
                self.model,
                self.hypothesis,
                self.a,
                self.b,
                z,
                self.shape,
            )
            z_list.append(z)
            flag_list.append(flag)
            z += step

            if self.progress:
                update = 100 * step / (right - left)
                self.bar.update(update if z < right else 100 - self.bar.n)

        self.zs = torch.tensor(z_list, dtype=torch.double)
        self.flags = torch.tensor(flag_list)

    def _detect_change_points(self) -> None:
        """Detect change points of in or out of the truncated region."""
        tol = 1e-12
        is_cp = torch.logical_xor(self.flags[:-1], self.flags[1:])
        loc_cp = torch.where(is_cp)[0]
        self.unsearched_flags = self.flags[loc_cp]
        self.unsearched = torch.stack(
            [self.zs[loc_cp], self.zs[loc_cp + 1]],
            dim=1,
        )

        n = self.zs.shape[0]
        loc_cp = torch.concat([torch.tensor([0]), loc_cp, torch.tensor([n - 1])])
        all_intervals = torch.stack(
            [self.zs[loc_cp[:-1] + 1], self.zs[loc_cp[1:]]],
            dim=1,
        )
        flags = self.flags[loc_cp[:-1] + 1]
        is_not_empty = all_intervals[:, 1] - all_intervals[:, 0] >= tol

        self.searched = RealSubset(all_intervals[is_not_empty].detach().numpy())
        self.truncated = RealSubset(
            all_intervals[torch.logical_and(flags, is_not_empty)].detach().numpy(),
        )

    def _finer_around_change_points(self) -> None:
        """Finer search around change points."""
        tol = 1e-10
        for index in range(self.unsearched.shape[0]):
            left, right = self.unsearched[index]
            left_flag = self.unsearched_flags[index]
            while right - left > tol:
                mid = (left + right) / 2.0
                input = (self.a + self.b * mid).reshape(self.shape)
                f_value = self.hypothesis.convert_to_value(
                    self.model.attention_map(input),
                )
                flag = bool((f_value <= 0.0).all().detach().item())

                if flag == left_flag:
                    interval_ = [[left.detach().item(), mid.detach().item()]]
                    left = mid
                else:
                    interval_ = [[mid.detach().item(), right.detach().item()]]
                    right = mid

                interval = RealSubset(interval_)
                self.searched |= interval
                if flag and left_flag:
                    self.truncated |= interval
