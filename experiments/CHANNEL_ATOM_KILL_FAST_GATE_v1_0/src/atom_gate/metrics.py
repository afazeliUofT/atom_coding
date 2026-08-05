from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from .models import DeterministicMap, Representation
from .representations import map_kappa, map_uniform_fiber_ambiguity


def entropy_bits(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def conditional_output_entropy_uniform(channel: np.ndarray) -> float:
    channel = np.asarray(channel, dtype=float)
    return float(np.mean([entropy_bits(row) for row in channel]))


def output_distribution_uniform(channel: np.ndarray) -> np.ndarray:
    return np.asarray(channel, dtype=float).mean(axis=0)


def mutual_information(channel: np.ndarray, input_distribution: np.ndarray) -> float:
    channel = np.asarray(channel, dtype=float)
    px = np.asarray(input_distribution, dtype=float)
    py = px @ channel
    value = 0.0
    for x in range(channel.shape[0]):
        for y in range(channel.shape[1]):
            pxy = px[x] * channel[x, y]
            if pxy > 0.0 and py[y] > 0.0:
                value += pxy * math.log2(channel[x, y] / py[y])
    return float(value)


def channel_capacity_blahut_arimoto(
    channel: np.ndarray,
    tolerance: float = 1e-12,
    max_iterations: int = 20000,
) -> tuple[float, np.ndarray, int]:
    """Blahut-Arimoto for a finite DMC, returning capacity in bits."""
    w = np.asarray(channel, dtype=float)
    m, _ = w.shape
    px = np.full(m, 1.0 / m)
    eps = 1e-300
    last = -1.0
    for iteration in range(1, max_iterations + 1):
        py = px @ w
        divergences = np.sum(
            np.where(w > 0.0, w * (np.log(np.maximum(w, eps)) - np.log(np.maximum(py, eps))), 0.0),
            axis=1,
        )
        r = np.exp(divergences - np.max(divergences))
        new_px = r / r.sum()
        capacity_nat = float(np.log(np.sum(np.exp(divergences))) - np.log(m))
        capacity = capacity_nat / np.log(2.0)
        if abs(capacity - last) < tolerance and np.max(np.abs(new_px - px)) < math.sqrt(tolerance):
            px = new_px
            return mutual_information(w, px), px, iteration
        px = new_px
        last = capacity
    return mutual_information(w, px), px, max_iterations


def representation_kappa(rep: Representation, output_size: int | None = None) -> float:
    qy = rep.output_size if output_size is None else output_size
    return float(
        sum(float(weight) * map_kappa(atom, qy) for atom, weight in zip(rep.maps, rep.weights, strict=True))
    )


def representation_uniform_fiber_ambiguity(rep: Representation, output_size: int | None = None) -> float:
    qy = rep.output_size if output_size is None else output_size
    return float(
        sum(
            float(weight) * map_uniform_fiber_ambiguity(atom, qy)
            for atom, weight in zip(rep.maps, rep.weights, strict=True)
        )
    )


def transition_degeneracy_entropy(rep: Representation, channel: np.ndarray) -> float:
    return max(0.0, entropy_bits(rep.weights) - conditional_output_entropy_uniform(channel))


def injective_mass(rep: Representation, code_symbols: Sequence[int] | None = None) -> float:
    return float(
        sum(
            float(weight)
            for atom, weight in zip(rep.maps, rep.weights, strict=True)
            if atom.is_injective(code_symbols)
        )
    )


def transition_multiplicity_stats(rep: Representation, channel: np.ndarray) -> dict[str, float]:
    m, qy = channel.shape
    multiplicities: list[int] = []
    weighted = 0.0
    for x in range(m):
        for y in range(qy):
            count = sum(1 for atom, weight in zip(rep.maps, rep.weights, strict=True) if weight > 0 and atom(x) == y)
            multiplicities.append(count)
            weighted += (1.0 / m) * channel[x, y] * count
    return {
        "multiplicity_mean_transition_weighted": float(weighted),
        "multiplicity_max": float(max(multiplicities, default=0)),
        "multiplicity_mean_unweighted": float(np.mean(multiplicities) if multiplicities else 0.0),
    }


def is_bi_unambiguous(rep: Representation, code_symbols: Sequence[int] | None = None) -> bool:
    symbols = tuple(range(rep.input_size)) if code_symbols is None else tuple(code_symbols)
    if any(not atom.is_injective(symbols) for atom in rep.maps):
        return False
    for x in symbols:
        outputs = [atom(x) for atom in rep.maps]
        if len(outputs) != len(set(outputs)):
            return False
    return True


def representation_summary(rep: Representation, channel: np.ndarray) -> dict[str, float | int | str | bool]:
    rep.verify(channel)
    kappa = representation_kappa(rep, channel.shape[1])
    capacity, px, iterations = channel_capacity_blahut_arimoto(channel)
    mult = transition_multiplicity_stats(rep, channel)
    summary: dict[str, float | int | str | bool] = {
        "representation": rep.name,
        "support_size": rep.support_size,
        "latent_entropy_bits": entropy_bits(rep.weights),
        "channel_h_y_given_x_uniform_bits": conditional_output_entropy_uniform(channel),
        "transition_degeneracy_bits": transition_degeneracy_entropy(rep, channel),
        "uniform_fiber_ambiguity_bits": representation_uniform_fiber_ambiguity(rep, channel.shape[1]),
        "kappa": kappa,
        "random_code_fiber_ceiling_bits_per_symbol": math.log2(rep.input_size) - math.log2(kappa),
        "injective_mass": injective_mass(rep),
        "bi_unambiguous": is_bi_unambiguous(rep),
        "channel_capacity_bits_per_symbol": capacity,
        "capacity_ba_iterations": iterations,
    }
    summary.update(mult)
    return summary


def expected_random_code_fiber(
    input_size: int,
    blocklength: int,
    code_size: int,
    kappa: float,
) -> float:
    ambient = input_size**blocklength
    return float(1.0 + (code_size - 1.0) / (ambient - 1.0) * (kappa**blocklength - 1.0))


def empirical_slope(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def bootstrap_slope_ci(
    xs: Sequence[float],
    grouped_values: Sequence[Sequence[float]],
    rng: np.random.Generator,
    replicates: int = 400,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    x = np.asarray(xs, dtype=float)
    means = np.array([np.mean(v) for v in grouped_values], dtype=float)
    point, _ = empirical_slope(x, np.log2(np.maximum(means, 1e-12)))
    if len(xs) < 2 or any(len(v) == 0 for v in grouped_values):
        return point, float("nan"), float("nan")
    slopes = []
    for _ in range(replicates):
        boot_means = []
        for values in grouped_values:
            arr = np.asarray(values, dtype=float)
            sample = rng.choice(arr, size=arr.size, replace=True)
            boot_means.append(float(np.mean(sample)))
        slope, _ = empirical_slope(x, np.log2(np.maximum(boot_means, 1e-12)))
        slopes.append(slope)
    low = float(np.quantile(slopes, alpha / 2.0))
    high = float(np.quantile(slopes, 1.0 - alpha / 2.0))
    return point, low, high
