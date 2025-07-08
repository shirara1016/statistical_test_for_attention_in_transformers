"""Module for local experiments."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import polars as pl
from sicore import generate_non_gaussian_rv  # type: ignore[import]

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

from experiment.transformer import (
    TransformerModel,
    setup_transformer_config,
)
from experiment.util import Config, create_cov_matrix
from si4attention.grid import AdaptiveGrid, CombinationGrid, FixedGrid, Grid
from si4attention.hypothesis import Background, Hypothesis, Neighbor, Reference
from si4attention.inference import GridBasedSelectiveInference


class Experiment:
    """Class for experiments."""

    def __init__(
        self,
        config: Config,
        root_seed: int,
        num_iter: int = 1000,
        num_worker: int = 32,
    ) -> None:
        """Initialize an experiment.

        Parameters
        ----------
        config : Config
            Configuration for the experiment.
        root_seed : int
            Root seed for the experiment.
        num_iter : int
            Number of iterations for the experiment. Defaults to 1000.
        num_worker : int
            Number of workers for parallel processing. Defaults to 32.
        """
        self.config = config
        self.root_seed = root_seed
        self.rng = np.random.default_rng([root_seed, config.unique_integer])
        self.num_iter = num_iter
        self.num_worker = num_worker

    def iter_experiment(
        self,
        data_: list[list[list[float]]] | list[list[float]],
    ) -> tuple[float, float]:
        """Run an iteration of the experiment.

        Parameters
        ----------
        data_ : list[list[list[float]]] | list[list[float]]
            Each data for the experiment.

        Returns
        -------
        tuple[float, float]
            P-value and elapsed time for the experiment.
        """
        grid: Grid
        hypothesis: Hypothesis
        reference: torch.Tensor | None

        shape: tuple[int, ...]
        match self.config.data_type:
            case "image":
                shape = (1, 1, self.config.data_size, self.config.data_size)
            case "series":
                shape = (1, 1, self.config.data_size)

        transformer_config = setup_transformer_config(
            self.config.data_type,
            self.config.data_size,
            self.config.architecture_size,
        )
        model = TransformerModel(transformer_config)
        model.load_state_dict(
            torch.load(
                f"model/{self.config.data_type}_{self.config.data_size}_{self.config.architecture_size}.pth",
                weights_only=True,
            ),
        )
        model.eval()
        model.double()

        match self.config.noise_type:
            case "corr":
                cov_ = create_cov_matrix(self.config.data_type, self.config.data_size)
                cov = torch.tensor(cov_).double()
            case "estimated":
                cov = torch.tensor(np.var(data_[-1])).double()
            case _:
                cov = torch.tensor(1.0).double()

        match self.config.method:
            case "combination":
                grid = CombinationGrid()
            case "fixed":
                grid = FixedGrid()
            case _:
                grid = AdaptiveGrid()

        match self.config.hypothesis:
            case "background":
                data = torch.tensor(data_).reshape(shape).double()
                reference = None
                hypothesis = Background()
            case "neighbor":
                data = torch.tensor(data_).reshape(shape).double()
                reference = None
                hypothesis = Neighbor(
                    data_type=self.config.data_type,
                    data_size=self.config.data_size,
                )
            case "reference":
                data, reference = (
                    torch.tensor(data_).reshape(2, *shape[1:]).double().chunk(2)
                )
                hypothesis = Reference()

        si = GridBasedSelectiveInference(
            data_type=self.config.data_type,
            data_size=self.config.data_size,
            grid=grid,
            hypothesis=hypothesis,
            model=model,
        )
        si.construct_hypothesis(cov, data, reference)

        start = time()
        match self.config.method:
            case "adaptive" | "combination" | "fixed":
                p_value = si.selective_p_value
            case "bonferroni":
                p_value = si.bonferroni_p_value
            case "naive":
                p_value = si.naive_p_value
            case "permutation":
                p_value = si.permutation_p_value
        return p_value, time() - start

    def run(self) -> None:
        """Run the experiment."""
        dataset = self._make_dataset().tolist()
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = executor.map(self.iter_experiment, dataset)
        p_values, times = zip(*results, strict=True)

        frame = pl.DataFrame(
            {
                **asdict(self.config),
                "root_seed": self.root_seed,
                "p_value": p_values,
                "time": times,
            },
        )
        frame.write_csv(
            f"result/{self.config}_{self.root_seed}.csv",
            float_scientific=True,
        )

    def _make_dataset(self) -> np.ndarray:
        """Make a dataset for the experiment.

        Returns
        -------
        np.ndarray
            Dataset for the experiment. Shape is (num_iter, num, *shape).
        """
        match self.config.data_type:
            case "image":
                shape = [self.config.data_size, self.config.data_size]
            case "series":
                shape = [self.config.data_size]
        num = 2 if self.config.hypothesis == "reference" else 1

        match self.config.noise_type:
            case "iid" | "estimated":
                x = self.rng.normal(size=(self.num_iter, num, *shape))
            case "corr":
                x = self.rng.multivariate_normal(
                    mean=np.zeros(np.prod(shape).item()),
                    cov=create_cov_matrix(self.config.data_type, self.config.data_size),
                    size=self.num_iter * num,
                ).reshape(self.num_iter, num, *shape)
            case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
                rv = generate_non_gaussian_rv(
                    self.config.noise_type,
                    self.config.deviation_from_gaussian,
                )
                x = rv.rvs(size=(self.num_iter, num, *shape), random_state=self.rng)

        if self.config.signal == 0.0:
            return x

        a = self.config.data_size // (3 if self.config.data_type == "image" else 10)
        for i in range(self.num_iter):
            match self.config.data_type:
                case "image":
                    ano_x, ano_y = self.rng.integers(0, self.config.data_size - a, 2)
                    x[i, 0, ano_x : ano_x + a, ano_y : ano_y + a] += self.config.signal
                case "series":
                    ano_x = self.rng.integers(0, self.config.data_size - a)
                    x[i, 0, ano_x : ano_x + a] += self.config.signal
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="adaptive")
    parser.add_argument("--data_type", type=str, default="image")
    parser.add_argument("--noise_type", type=str, default="iid")
    parser.add_argument("--data_size", type=int, default=-1)
    parser.add_argument("--architecture_size", type=str, default="base")
    parser.add_argument("--signal", type=float, default=0.0)
    parser.add_argument("--hypothesis", type=str, default="background")
    parser.add_argument("--deviation_from_gaussian", type=float, default=0.0)
    parser.add_argument("--root_seed", type=int, default=0)

    args = vars(parser.parse_args())
    root_seed = args.pop("root_seed")
    config = Config(**args)
    print(config, root_seed)  # noqa: T201
    experiment = Experiment(config, root_seed)
    experiment.run()
