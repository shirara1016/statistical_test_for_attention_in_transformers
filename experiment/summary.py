"""Module for plotting the results of the experiments."""

import contextlib
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import kstest, norm  # type: ignore[import]
from sicore import pvalues_hist, pvalues_qqplot, uniformity_test  # type: ignore[import]

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

from experiment.util import Config

plt.rcParams["font.size"] = 25  # 26

MARKERSIZE = 4
CAPSIZE = 7
CAPTHICK = 2
LW = 1.8

plt.rcParams["figure.figsize"] = (6.4, 5.4)


def check_uniformity(config: Config) -> None:
    """Check the uniformity of the p-values."""
    p_values = pl.read_csv(f"result/{config}_*.csv").get_column("p_value").to_numpy()
    print(kstest(p_values, "uniform"))  # noqa: T201
    print(uniformity_test(p_values[:1000], alpha=0.05))  # noqa: T201
    pvalues_hist(p_values, bins=50, fname=f"{config}_hist.pdf")
    pvalues_qqplot(p_values, fname=f"{config}_qqplot.pdf")


def _load_frame(num_seeds: int = 10, *, is_saved: bool = False) -> pl.DataFrame:
    """Load the result frame and preprocess it."""
    if is_saved:
        return pl.read_csv(f"result_processed/upto{num_seeds}.csv")
    frame = (
        pl.scan_csv("result/*.csv")
        # filter the rows with specified number of seeds
        .filter(pl.col("root_seed").is_in(range(num_seeds)))
        # group by each configuration and aggregate the reject rates and computation time
        .group_by(pl.exclude("p_value", "time", "root_seed"), maintain_order=True)
        .agg(
            *[
                (pl.col("p_value") < alpha).mean().alias(str(alpha))
                for alpha in (0.01, 0.05, 0.1)
            ],
            pl.col("time").mean(),
            (pl.col("time").std() / pl.col("time").len().sqrt()).alias("time_sd"),
            pl.col("p_value").len().alias("n"),  # to delete
        )
        # make new column alpha representing the significance level used for reject rate
        .unpivot(
            index=[*asdict(Config()).keys(), "time", "time_sd", "n"],
            variable_name="alpha",
            value_name="reject_rate",
        )
        .with_columns(pl.col("alpha").cast(pl.Float64))
        # compute the error bars for the reject rate
        .with_columns(
            (pl.col("reject_rate") * (1 - pl.col("reject_rate")) / pl.col("n"))
            .sqrt()
            .alias("reject_rate_sd"),
        )
        # lazy evaluation
        .collect()
    )
    frame.write_csv(f"result_processed/upto{num_seeds}.csv")
    return frame


def _extract_sub_frame(
    frame: pl.DataFrame,
    config: Config,
    target: str,
    hue: str,
    feature: str,
    hues: list[str] | list[float] | list[int],
    features: list[str] | list[float] | list[int],
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Extract a sub-frame from the preprocessed frame."""
    # setup the base configuration
    config_dict = asdict(config)
    config_dict.update({"alpha": alpha})
    return (
        # set the non-related configurations to the base configuration
        frame.filter(
            pl.col(key) == value
            for key, value in config_dict.items()
            if key not in (feature, hue)
        )
        # filter the relevant configurations
        .filter(pl.col(hue).is_in(hues), pl.col(feature).is_in(features))
        # cast the feature column to enum to ensure the specified order
        .with_columns(
            pl.col(feature).cast(pl.String).cast(pl.Enum([str(f) for f in features])),
        )
        .sort(feature)
        # select the relevant columns
        .select(pl.col(hue, feature, target, f"{target}_sd"))  # to delete "n"
    )


def _reshape_frame(
    extracted_frame: pl.DataFrame,
    target: str,
    hues: list[str] | list[float] | list[int],
    hue: str,
) -> tuple[dict[str | float | int, np.ndarray], dict[str | float | int, np.ndarray]]:
    """Reshape the frame for the plotting."""
    target_at_hue = {
        each_hue: extracted_frame.filter(pl.col(hue) == each_hue)
        .get_column(target)
        .to_numpy()
        for each_hue in hues
    }
    target_sd_at_hue = {
        each_hue: extracted_frame.filter(pl.col(hue) == each_hue)
        .get_column(f"{target}_sd")
        .to_numpy()
        for each_hue in hues
    }
    return target_at_hue, target_sd_at_hue


def _set_info(
    data_type: str,
    feature: str,
) -> tuple[list[int] | list[float] | list[str], list[str], str]:
    features: list[int] | list[float] | list[str]
    match data_type, feature:
        case "image", "data_size":
            features = [8, 16, 32, 64]
            xticks = ["64", "256", "1024", "4096"]
            xlabel = "Image Size"
        case "series", "data_size":
            features = [128, 256, 512, 1024]
            xticks = ["128", "256", "512", "1024"]
            xlabel = "Series Length"
        case _, "architecture_size":
            features = xticks = ["small", "base", "large", "huge"]
            xlabel = "Architecture"
        case _, "signal":
            features = [1.0, 2.0, 3.0, 4.0]
            xticks = ["1.0", "2.0", "3.0", "4.0"]
            xlabel = "Signal Strength"
    return features, xticks, xlabel


def fpr_plot(
    frame: pl.DataFrame,
    feature: str,
    data_type: str,
    noise_type: str,
    hypothesis: str,
) -> None:
    """Plot the false positive rate."""
    target = "reject_rate"
    hue = "method"
    hues = labels = ["adaptive", "bonferroni", "permutation", "naive"]

    features, xticks, xlabel = _set_info(data_type, feature)
    config = Config(data_type=data_type, noise_type=noise_type, hypothesis=hypothesis)

    extracted_frame = _extract_sub_frame(
        frame,
        config,
        target,
        hue,
        feature,
        hues,
        features,
    )
    target_at_hue, target_sd_at_hue = _reshape_frame(extracted_frame, target, hues, hue)

    plt.close()
    plt.figure()
    for each_hue, label in zip(hues, labels, strict=True):
        plt.errorbar(
            x=range(len(xticks)),
            y=target_at_hue[each_hue],
            yerr=norm.ppf(0.975) * target_sd_at_hue[each_hue],
            label=label,
            fmt="o-",
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            lw=LW,
        )
    plt.plot(range(len(xticks)), [0.05] * len(xticks), "--", color="red", lw=0.5)

    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel("Type I Error Rate")

    plt.savefig(
        f"figure/fpr/fpr_{data_type}_{hypothesis}_{noise_type}_{feature}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def tpr_plot(
    frame: pl.DataFrame,
    data_type: str,
    noise_type: str,
    hypothesis: str,
) -> None:
    """Plot the true positive rate."""
    target = "reject_rate"
    hue = "method"
    hues = labels = ["adaptive", "bonferroni"]
    feature = "signal"

    features, xticks, xlabel = _set_info(data_type, feature)
    config = Config(data_type=data_type, noise_type=noise_type, hypothesis=hypothesis)

    extracted_frame = _extract_sub_frame(
        frame,
        config,
        target,
        hue,
        feature,
        hues,
        features,
    )
    target_at_hue, target_sd_at_hue = _reshape_frame(extracted_frame, target, hues, hue)

    plt.close()
    plt.figure()
    for each_hue, label in zip(hues, labels, strict=True):
        plt.errorbar(
            x=range(len(xticks)),
            y=target_at_hue[each_hue],
            yerr=norm.ppf(0.975) * target_sd_at_hue[each_hue],
            label=label,
            fmt="o-",
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            lw=LW,
        )

    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel("Statistical Power")

    plt.savefig(
        f"figure/tpr/tpr_{data_type}_{hypothesis}_{noise_type}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def time_plot(
    frame: pl.DataFrame,
    feature: str,
    data_type: str,
    noise_type: str,
    hypothesis: str,
) -> None:
    """Plot the computation time."""
    target = "time"
    hue = "method"
    hues = labels = ["adaptive", "combination", "fixed"]

    features, xticks, xlabel = _set_info(data_type, feature)
    config = Config(data_type=data_type, noise_type=noise_type, hypothesis=hypothesis)

    extracted_frame = _extract_sub_frame(
        frame,
        config,
        target,
        hue,
        feature,
        hues,
        features,
    )
    target_at_hue, target_sd_at_hue = _reshape_frame(extracted_frame, target, hues, hue)

    plt.close()
    plt.figure()
    for each_hue, label in zip(hues, labels, strict=True):
        plt.errorbar(
            x=range(len(xticks)),
            y=target_at_hue[each_hue],
            yerr=norm.ppf(0.975) * target_sd_at_hue[each_hue],
            label=label,
            fmt="o-",
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            lw=LW,
        )

    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylabel("Computation Time (s)")
    plt.yscale("log")
    plt.ylim(1, 16500)

    plt.savefig(
        f"figure/time/time_{data_type}_{hypothesis}_{noise_type}_{feature}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def fpr_non_gaussian_plot(
    frame: pl.DataFrame,
    data_type: str,
    hypothesis: str,
    alpha: float,
) -> None:
    """Plot the false positive rate for non-Gaussian noise."""
    target = "reject_rate"
    hue = "noise_type"
    hues = labels = ["skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"]
    feature = "deviation_from_gaussian"

    features = [0.01, 0.02, 0.03, 0.04]
    xticks = ["0.01", "0.02", "0.03", "0.04"]
    xlabel = "Wasserstein Distance"
    config = Config(data_type=data_type, hypothesis=hypothesis)

    extracted_frame = _extract_sub_frame(
        frame,
        config,
        target,
        hue,
        feature,
        hues,
        features,
        alpha,
    )
    target_at_hue, target_sd_at_hue = _reshape_frame(extracted_frame, target, hues, hue)

    plt.close()
    plt.figure()
    for each_hue, label in zip(hues, labels, strict=True):
        plt.errorbar(
            x=range(len(xticks)),
            y=target_at_hue[each_hue],
            yerr=norm.ppf(0.975) * target_sd_at_hue[each_hue],
            label=label,
            fmt="o-",
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            lw=LW,
        )
    plt.plot(range(len(xticks)), [alpha] * len(xticks), "--", color="red", lw=0.5)

    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylim(0.0, 4 * alpha)
    plt.yticks([0.0, alpha, 2 * alpha, 3 * alpha, 4 * alpha])
    plt.ylabel("Type I Error Rate")

    plt.savefig(
        f"figure/robust/non_gaussian_{data_type}_{hypothesis}_{alpha}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def fpr_estimated_plot(
    frame: pl.DataFrame,
    feature: str,
    data_type: str,
    hypothesis: str,
) -> None:
    """Plot the false positive rate for estimated noise."""
    target = "reject_rate"
    hue = "alpha"
    hues = [0.05, 0.01, 0.1]
    labels = [f"alpha={alpha:.2f}" for alpha in hues]
    noise_type = "estimated"

    features, xticks, xlabel = _set_info(data_type, feature)
    config = Config(data_type=data_type, noise_type=noise_type, hypothesis=hypothesis)

    extracted_frame = _extract_sub_frame(
        frame,
        config,
        target,
        hue,
        feature,
        hues,
        features,
    )
    target_at_hue, target_sd_at_hue = _reshape_frame(extracted_frame, target, hues, hue)

    plt.close()
    plt.figure()
    for each_hue, label in zip(hues, labels, strict=True):
        plt.errorbar(
            x=range(len(xticks)),
            y=target_at_hue[each_hue],
            yerr=norm.ppf(0.975) * target_sd_at_hue[each_hue],
            label=label,
            fmt="o-",
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            lw=LW,
        )
    for alpha in (0.01, 0.05, 0.1):
        plt.plot(range(len(xticks)), [alpha] * len(xticks), "--", color="red", lw=0.5)

    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylim(0, 0.2)
    plt.yticks([0.01, 0.05, 0.1, 0.15, 0.2])
    plt.ylabel("Type I Error Rate")

    plt.savefig(
        f"figure/robust/estimated_{data_type}_{hypothesis}_{feature}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


if __name__ == "__main__":
    # check_uniformity(
    #     Config(data_type="series", hypothesis="background", noise_type="corr"),
    # )

    frame_upto_10 = _load_frame(num_seeds=10)
    frame_upto_1 = _load_frame(num_seeds=1)

    Path("figure").mkdir(exist_ok=True)
    for data_type, noise_type, hypothesis in product(
        ("image", "series"),
        ("iid", "corr"),
        ("background", "neighbor", "reference"),
    ):
        for feature in "data_size", "architecture_size":
            with contextlib.suppress(Exception):
                fpr_plot(
                    frame_upto_10,
                    feature=feature,
                    data_type=data_type,
                    noise_type=noise_type,
                    hypothesis=hypothesis,
                )
            with contextlib.suppress(Exception):
                time_plot(
                    frame_upto_1,
                    feature=feature,
                    data_type=data_type,
                    noise_type=noise_type,
                    hypothesis=hypothesis,
                )
        with contextlib.suppress(Exception):
            tpr_plot(
                frame_upto_10,
                data_type=data_type,
                noise_type=noise_type,
                hypothesis=hypothesis,
            )

    for feature, data_type, hypothesis in product(
        ("data_size", "architecture_size"),
        ("image", "series"),
        ("background", "neighbor", "reference"),
    ):
        with contextlib.suppress(Exception):
            fpr_estimated_plot(
                frame_upto_10,
                feature=feature,
                data_type=data_type,
                hypothesis=hypothesis,
            )
    for data_type, hypothesis, alpha in product(
        ("image", "series"),
        ("background", "neighbor", "reference"),
        (0.05, 0.01),
    ):
        with contextlib.suppress(Exception):
            fpr_non_gaussian_plot(
                frame_upto_10,
                data_type=data_type,
                hypothesis=hypothesis,
                alpha=alpha,
            )
