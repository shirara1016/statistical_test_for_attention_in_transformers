"""Module for grid search."""

from typing import Literal

import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)

import numpy as np
from scipy.stats import norm  # type: ignore[import]
from sicore import RealSubset, truncated_cdf  # type: ignore[import]

from si4attention.abstract import SupportsExtractWeights, Transformer
from si4attention.hypothesis import Hypothesis


class BatchAdaptiveGridInference:
    """Class for batch adaptive grid based selective inference for attentions in transformers."""

    def __init__(
        self,
        data_type: str,
        data_size: int,
        hypothesis: Hypothesis,
        model: SupportsExtractWeights,
        min_width: float = 1e-4,
        max_width: float = 1e-1,
        batch: int = 64,
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
        hypothesis : Hypothesis
            The hypothesis to test.
        model : Transformer
            The transformer model.
        min_width : float, optional
            The minimum width of the grid for adaptive grid search.
            Defaults to 1e-4.
        max_width : float, optional
            The maximum width of the grid for adaptive grid search.
            Defaults to 1e-1.
        batch: int, optional
            The batch size for the adaptive grid search.
        progress : bool, optional
            Whether to show progress bar. Defaults to `False`.
        """
        self.data_type = data_type
        self.data_size = data_size
        self.hypothesis = hypothesis
        self.model = Transformer(model, data_type)
        self.min_width = min_width
        self.max_width = max_width
        self.batch = batch
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

        self.cov, self.data, self.reference = cov, data, reference

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
        lefts, rights = self.unsearched[:, 0], self.unsearched[:, 1]
        left_flags = self.unsearched_flags
        truncated_list, searched_list = [], []
        while lefts.numel() > 0:
            lefts, lefts_ = lefts[: self.batch], lefts[self.batch :]
            rights, rights_ = rights[: self.batch], rights[self.batch :]
            left_flags, left_flags_ = left_flags[: self.batch], left_flags[self.batch :]

            mids = (lefts + rights) / 2.0
            inputs = (self.a + self.b * mids.unsqueeze(1)).reshape(self.shape)
            f_values = self.hypothesis.convert_to_value(
                self.model.attention_map(inputs),
            )
            flags = (f_values <= 0.0).all(1).detach()

            indices = flags == left_flags
            lefts = torch.where(indices, mids, lefts)
            rights = torch.where(indices, rights, mids)
            intervals_ = torch.stack([lefts, rights], dim=1)
            intervals_[indices, 1] = mids[indices]
            intervals_[~indices, 0] = mids[~indices]

            truncated_list.append(intervals_[flags & left_flags])
            searched_list.append(intervals_)

            lefts = torch.cat((lefts, lefts_))
            rights = torch.cat((rights, rights_))
            left_flags = torch.cat((left_flags, left_flags_))
            mask = rights - lefts >= tol
            lefts, rights, left_flags = lefts[mask], rights[mask], left_flags[mask]

        searched = RealSubset(torch.cat(searched_list).detach().numpy())
        truncated = RealSubset(torch.cat(truncated_list).detach().numpy())
        self.searched |= searched
        self.truncated |= truncated

    def _divide(
        self,
        starts: torch.Tensor,
        ends: torch.Tensor,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_indices = starts < ends
        starts, ends = starts[valid_indices], ends[valid_indices]

        current_batch = starts.numel()
        if current_batch in (batch, 0):
            return starts, ends

        indices = torch.argsort(ends - starts, descending=True)
        div_indices = indices[: batch - current_batch]
        all_indices = torch.arange(starts.numel())
        not_div_indices = all_indices[~torch.isin(all_indices, div_indices)]

        mids = (starts + ends) / 2.0
        starts = torch.cat(
            (starts[div_indices], starts[not_div_indices], mids[div_indices]),
        )
        ends = torch.cat((mids[div_indices], ends[not_div_indices], ends[div_indices]))
        starts, ends = self._divide(starts, ends, batch)

        sort_indices = torch.argsort(starts)
        return starts[sort_indices], ends[sort_indices]

    def _cut_grid(self) -> None:
        """Cut the grid in the search space."""
        self.zs, self.flags = torch.tensor([]).double(), torch.tensor([]).bool()

        starts, ends = self._divide(
            torch.tensor([self.stat - 0.1, self.stat]).double(),
            torch.tensor([self.stat, self.stat + 0.1]).double(),
            self.batch,
        )
        while starts.numel() > 0:
            flags_, steps = self.batch_evaluate_grids(
                self.model,
                self.hypothesis,
                self.a,
                self.b,
                starts,
                self.shape,
                mode="near",
            )
            self.zs = torch.cat((self.zs, starts))
            self.flags = torch.cat((self.flags, flags_))
            # print(steps.sum())
            if self.progress:
                clipped = torch.where(starts + steps > ends, ends - starts, steps)
                update = 100 * clipped.sum().item() / (2.0 * abs(self.stat) + 20.0)
                ratio = 100 * 0.1 / (abs(self.stat) + 10.0)
                self.bar.update(
                    update if (starts + steps < ends).any() else ratio - self.bar.n,
                )
            starts, ends = self._divide(starts + steps, ends, self.batch)

        starts, ends = self._divide(
            torch.tensor([-abs(self.stat) - 10.0, self.stat + 0.1]).double(),
            torch.tensor([self.stat - 0.1, abs(self.stat) + 10.0]).double(),
            self.batch,
        )
        while starts.numel() > 0:
            flags_, steps = self.batch_evaluate_grids(
                self.model,
                self.hypothesis,
                self.a,
                self.b,
                starts,
                self.shape,
                mode="far",
            )
            self.zs = torch.cat((self.zs, starts))
            self.flags = torch.cat((self.flags, flags_))
            # print(steps.sum())
            if self.progress:
                clipped = torch.where(starts + steps > ends, ends - starts, steps)
                update = 100 * clipped.sum().item() / (2.0 * abs(self.stat) + 20.0)
                self.bar.update(
                    update if (starts + steps < ends).any() else 100 - self.bar.n,
                )
            starts, ends = self._divide(starts + steps, ends, self.batch)
        if self.progress:
            self.bar.close()

        sort_indices = torch.argsort(self.zs)
        self.zs, self.flags = self.zs[sort_indices], self.flags[sort_indices]

    def batch_evaluate_grids(
        self,
        model: Transformer,
        hypothesis: Hypothesis,
        a: torch.Tensor,
        b: torch.Tensor,
        zs: torch.Tensor,
        shape: tuple[int, ...],
        mode: Literal["near", "far"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        zs : torch.Tensor
            The batch of grid points to evaluate.
        shape : tuple[int, ...]
            The shape of the input data.
        mode : Literal["near", "far"]
            The mode of the grid search.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Boolean values indicating if each grid point belongs to the
            truncated interval and the widths of the next grid points.
        """
        input = (a + b * zs.unsqueeze(1)).reshape(shape)
        if mode == "near":
            f_values = hypothesis.convert_to_value(model.attention_map(input))
            flags = (f_values <= 0.0).all(1).detach()
            steps = torch.where(
                flags,
                (f_values / 1.0).abs().min(1).values,
                (f_values / 1.0).max(1).values,
            )
        else:
            batch_attention_map, batch_attention_map_jvp = torch.func.jvp(
                func=model.attention_map,
                primals=(input,),
                tangents=(b.expand(input.shape[0], -1).reshape(shape),),
            )
            f_values = hypothesis.convert_to_value(batch_attention_map)
            f_derivatives = hypothesis.convert_to_derivative(batch_attention_map_jvp)
            flags = (f_values <= 0.0).all(1).detach()

            candidates = -f_values / (10.0 * f_derivatives)
            candidates = torch.where(candidates < 0.0, 10.0, candidates)
            steps = torch.zeros_like(zs)
            steps[flags] = (
                torch.where(f_values[flags] <= 0.0, candidates[flags], self.max_width)
                .min(1)
                .values
            )
            steps[~flags] = (
                torch.where(f_values[~flags] >= 0.0, candidates[~flags], self.min_width)
                .max(1)
                .values
            )
        return flags, torch.clamp(steps, self.min_width, self.max_width)
