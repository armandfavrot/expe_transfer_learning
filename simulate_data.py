"""Simule les deux jeux de donnees decrits dans protocole.md."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


X1_LEVELS = ("a1", "b1")
X2_LEVELS = ("a2", "b2", "c2")
X3_DISTRIBUTIONS = {"a1": (0.0, 1.0), "b1": (0.5, 1.2)}
X4_DISTRIBUTIONS = {"a1": (0.0, 1.0), "b1": (-0.5, 1.2)}


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def balanced_design(rng: np.random.Generator) -> pd.DataFrame:
    # 334 + 333 + 333 = 1 000 observations dans chaque domaine.
    rows = [
        (x1, x2)
        for x1 in X1_LEVELS
        for x2, count in zip(X2_LEVELS, (334, 333, 333), strict=True)
        for _ in range(count)
    ]
    rng.shuffle(rows)
    data = pd.DataFrame(rows, columns=["X1", "X2"])
    x3_mean = data["X1"].map({x1: params[0] for x1, params in X3_DISTRIBUTIONS.items()})
    x3_std = data["X1"].map({x1: params[1] for x1, params in X3_DISTRIBUTIONS.items()})
    x4_mean = data["X1"].map({x1: params[0] for x1, params in X4_DISTRIBUTIONS.items()})
    x4_std = data["X1"].map({x1: params[1] for x1, params in X4_DISTRIBUTIONS.items()})
    data["X3"] = rng.normal(loc=x3_mean, scale=x3_std)
    data["X4"] = rng.normal(loc=x4_mean, scale=x4_std)
    return data


def response_model_1(data: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    mu1 = data["X1"].map({"a1": 0.0, "b1": 1.0}).to_numpy()
    mu2 = data["X2"].map({"a2": 0.0, "b2": -0.75, "c2": 0.75}).to_numpy()
    gamma1 = data["X1"].map({"a1": 0.0, "b1": -0.5}).to_numpy()
    return 2.0 + mu1 + mu2 + (1.5 + gamma1) * data["X3"].to_numpy() + rng.normal(size=len(data))


def response_model_2(data: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    mu1_map = {"a1": 0.0, "b1": 0.75}
    mu12_map = {
        ("a1", "a2"): 0.0, ("a1", "b2"): -1.0, ("a1", "c2"): 1.0,
        ("b1", "a2"): 0.0, ("b1", "b2"): -0.5, ("b1", "c2"): 1.5,
    }
    gamma12_map = {
        ("a1", "a2"): 0.0, ("a1", "b2"): 0.4, ("a1", "c2"): -0.4,
        ("b1", "a2"): -0.3, ("b1", "b2"): 0.7, ("b1", "c2"): -0.8,
    }
    gamma2_map = {"a2": 0.0, "b2": 0.4, "c2": -0.3}
    pairs = list(zip(data["X1"], data["X2"], strict=True))
    x3, x4 = data["X3"].to_numpy(), data["X4"].to_numpy()
    return (
        2.0
        + data["X1"].map(mu1_map).to_numpy()
        + np.array([mu12_map[p] for p in pairs])
        + (1.2 + np.array([gamma12_map[p] for p in pairs])) * x3
        + (0.8 + data["X2"].map(gamma2_map).to_numpy()) * x3 * x4
        + rng.normal(size=len(data))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = set_seed(args.seed)
    generated = {}
    for model, response_fn in (("model_1", response_model_1), ("model_2", response_model_2)):
        data = balanced_design(rng)
        data.insert(0, "id", [f"{model}_{i:04d}" for i in range(len(data))])
        data["Y"] = response_fn(data, rng)
        path = args.output_dir / f"{model}.csv"
        data.to_csv(path, index=False)
        generated[model] = str(path)

    metadata = {
        "seed": args.seed,
        "n_per_model": 2000,
        "x3_distributions": X3_DISTRIBUTIONS,
        "x4_distributions": X4_DISTRIBUTIONS,
        "files": generated,
    }
    (args.output_dir / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Donnees creees dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
