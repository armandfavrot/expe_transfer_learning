"""Execute l'experience de transfert definie dans protocole.md."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


X1_MAP = {"a1": 0, "b1": 1}
X2_MAP = {"a2": 0, "b2": 1, "c2": 2}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


class RegressionMLP(nn.Module):
    def __init__(self, include_x1: bool) -> None:
        super().__init__()
        self.include_x1 = include_x1
        self.x1_embedding = nn.Embedding(2, 1) if include_x1 else None
        self.x2_embedding = nn.Embedding(3, 2)
        self.network = nn.Sequential(
            nn.Linear(5 if include_x1 else 4, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.x1_embedding is not None:
            nn.init.zeros_(self.x1_embedding.weight)
        nn.init.normal_(self.x2_embedding.weight, mean=0.0, std=0.01)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        features = [self.x2_embedding(x2), continuous]
        if self.x1_embedding is not None:
            features.insert(0, self.x1_embedding(x1))
        features = torch.cat(features, dim=1)
        return self.network(features).squeeze(1)


def stratified_indices(indices: np.ndarray, labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tire n indices avec des effectifs par classe differant d'au plus un."""
    classes = np.unique(labels[indices])
    base, remainder = divmod(n, len(classes))
    order = rng.permutation(classes)
    wanted = {cls: base + int(cls in order[:remainder]) for cls in classes}
    chosen = []
    for cls in classes:
        pool = indices[labels[indices] == cls]
        chosen.extend(rng.choice(pool, size=wanted[cls], replace=False).tolist())
    return rng.permutation(np.asarray(chosen, dtype=int))


def nested_target_samples(pool: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict[int, np.ndarray]:
    # Un ordre stratifie par blocs garantit l'emboitement des echantillons et leur equilibre.
    per_class = {c: rng.permutation(pool[labels[pool] == c]).tolist() for c in np.unique(labels[pool])}
    order = []
    class_order = rng.permutation(list(per_class))
    while len(order) < 500:
        for cls in class_order:
            if per_class[cls] and len(order) < 500:
                order.append(per_class[cls].pop())
    selected = np.asarray(order, dtype=int)
    return {n: selected[:n].copy() for n in (10, 50, 100, 200, 500)}


def split_train_validation(sample: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_val = len(sample) // 5
    val = stratified_indices(sample, labels, n_val, rng)
    train = np.asarray([i for i in sample if i not in set(val)], dtype=int)
    return train, val


def tensors(data: pd.DataFrame, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> TensorDataset:
    part = data.iloc[indices]
    x1 = torch.tensor(part["X1"].map(X1_MAP).to_numpy(), dtype=torch.long)
    x2 = torch.tensor(part["X2"].map(X2_MAP).to_numpy(), dtype=torch.long)
    continuous = torch.tensor((part[["X3", "X4"]].to_numpy() - mean) / std, dtype=torch.float32)
    y = torch.tensor(part["Y"].to_numpy(), dtype=torch.float32)
    return TensorDataset(x1, x2, continuous, y)


def dataset_mse(model: nn.Module, dataset: TensorDataset, device: torch.device) -> float:
    """Calcule la MSE sur un jeu complet, avec dropout desactive."""
    model.eval()
    with torch.no_grad():
        x1, x2, cont, y = (v.to(device) for v in dataset.tensors)
        return nn.functional.mse_loss(model(x1, x2, cont), y).item()


def train_model(model: nn.Module, train_data: TensorDataset, val_data: TensorDataset, device: torch.device,
                seed: int, lr: float, max_epochs: int, patience: int,
                min_delta: float) -> tuple[nn.Module, int, float, list[dict]]:
    seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=min(32, len(train_data)), shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x1, x2, cont, y in loader:
            x1, x2, cont, y = x1.to(device), x2.to(device), cont.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x1, x2, cont), y)
            loss.backward()
            optimizer.step()
        train_loss = dataset_mse(model, train_data, device)
        val_loss = dataset_mse(model, val_data, device)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss - min_delta:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in history:
        row["is_best_epoch"] = row["epoch"] == best_epoch
    return model, best_epoch, best_loss, history


def rmse(model: nn.Module, dataset: TensorDataset, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        x1, x2, cont, y = (v.to(device) for v in dataset.tensors)
        return torch.sqrt(torch.mean((model(x1, x2, cont) - y) ** 2)).item()


def generator_prediction(data: pd.DataFrame, dataset: str) -> np.ndarray:
    """Retourne E[Y | X], c'est-a-dire la prediction du modele generateur."""
    x1 = data["X1"]
    x2 = data["X2"]
    x3 = data["X3"].to_numpy()
    x4 = data["X4"].to_numpy()
    pairs = list(zip(x1, x2, strict=True))

    if dataset == "model_1":
        mu1 = x1.map({"a1": 0.0, "b1": 1.0}).to_numpy()
        mu2 = x2.map({"a2": 0.0, "b2": -0.75, "c2": 0.75}).to_numpy()
        gamma1 = x1.map({"a1": 0.0, "b1": -0.5}).to_numpy()
        return 2.0 + mu1 + mu2 + (1.5 + gamma1) * x3
    if dataset == "model_2":
        mu1 = x1.map({"a1": 0.0, "b1": 0.75}).to_numpy()
        mu12 = {
            ("a1", "a2"): 0.0, ("a1", "b2"): -1.0, ("a1", "c2"): 1.0,
            ("b1", "a2"): 0.0, ("b1", "b2"): -0.5, ("b1", "c2"): 1.5,
        }
        gamma12 = {
            ("a1", "a2"): 0.0, ("a1", "b2"): 0.4, ("a1", "c2"): -0.4,
            ("b1", "a2"): -0.3, ("b1", "b2"): 0.7, ("b1", "c2"): -0.8,
        }
        gamma2 = x2.map({"a2": 0.0, "b2": 0.4, "c2": -0.3}).to_numpy()
        return (2.0 + mu1 + np.array([mu12[p] for p in pairs])
                + (1.2 + np.array([gamma12[p] for p in pairs])) * x3
                + (0.8 + gamma2) * x3 * x4)
    raise ValueError(f"Modele generateur inconnu : {dataset}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda demande, mais CUDA n'est pas disponible.")
    return torch.device(requested)


def run_dataset(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict, list[dict]]:
    data = pd.read_csv(path)
    labels = data["X2"].map(X2_MAP).to_numpy()
    source = np.flatnonzero(data["X1"].to_numpy() == "a1")
    target = np.flatnonzero(data["X1"].to_numpy() == "b1")
    results = []
    histories = []
    splits = {"repetitions": {}}

    for repetition in range(1, args.repetitions + 1):
        split_seed = args.split_seed + repetition
        sampling_seed = args.sampling_seed + repetition
        base_seed = args.training_seed + 10_000 * repetition
        split_rng = np.random.default_rng(split_seed)
        target_test = stratified_indices(target, labels, 500, split_rng)
        target_pool = np.asarray([i for i in target if i not in set(target_test)], dtype=int)
        source_val = stratified_indices(source, labels, 200, split_rng)
        source_train = np.asarray([i for i in source if i not in set(source_val)], dtype=int)
        mean = data.iloc[source_train][["X3", "X4"]].mean().to_numpy()
        std = data.iloc[source_train][["X3", "X4"]].std(ddof=0).to_numpy()
        test_ds = tensors(data, target_test, mean, std)
        test_data = data.iloc[target_test]
        oracle_predictions = generator_prediction(test_data, path.stem)
        oracle_rmse = float(np.sqrt(np.mean((test_data["Y"].to_numpy() - oracle_predictions) ** 2)))

        rng = np.random.default_rng(sampling_seed)
        nested = nested_target_samples(target_pool, labels, rng)
        seed_everything(base_seed)
        pretrained = RegressionMLP(include_x1=False)
        pretrained, pre_epoch, _, pre_history = train_model(
            pretrained, tensors(data, source_train, mean, std), tensors(data, source_val, mean, std),
            device, base_seed, 1e-3, args.max_epochs, args.patience, args.min_delta,
        )
        rep_splits = {
            "split_seed": split_seed,
            "sampling_seed": sampling_seed,
            "pretraining_seed": base_seed,
            "source_train": source_train.tolist(),
            "source_validation": source_val.tolist(),
            "target_pool": target_pool.tolist(),
            "target_test": target_test.tolist(),
            "standardization": {"mean": mean.tolist(), "std": std.tolist()},
            "oracle_rmse": oracle_rmse,
            "pretraining_best_epoch": pre_epoch,
            "samples": {},
        }
        for row in pre_history:
            histories.append({"dataset": path.stem, "repetition": repetition,
                              "training_type": "source_pretraining", "strategy": "source_pretraining",
                              "n_target": None, "uses_x1": False, **row})
        for size, sample in nested.items():
            train_idx, val_idx = split_train_validation(sample, labels, rng)
            rep_splits["samples"][str(size)] = {
                "sample": sample.tolist(), "train": train_idx.tolist(), "validation": val_idx.tolist()
            }
            val_ds = tensors(data, val_idx, mean, std)
            strategies = (
                ("fine_tuning", copy.deepcopy(pretrained), 1e-4, False),
                ("from_scratch", None, 1e-3, False),
                ("combined", None, 1e-3, True),
            )
            for offset, (strategy, model, lr, include_x1) in enumerate(strategies):
                model_seed = base_seed + size * 10 + offset
                if model is None:
                    seed_everything(model_seed)
                    model = RegressionMLP(include_x1=include_x1)
                fit_indices = np.concatenate((source, train_idx)) if strategy == "combined" else train_idx
                model, best_epoch, val_loss, history = train_model(
                    model, tensors(data, fit_indices, mean, std), val_ds, device, model_seed,
                    lr, args.max_epochs, args.patience, args.min_delta,
                )
                results.append({"dataset": path.stem, "repetition": repetition, "n_target": size,
                                "strategy": strategy, "uses_x1": include_x1,
                                "rmse": rmse(model, test_ds, device), "oracle_rmse": oracle_rmse,
                                "best_epoch": best_epoch, "best_validation_mse": val_loss, "seed": model_seed})
                for row in history:
                    histories.append({"dataset": path.stem, "repetition": repetition,
                                      "training_type": "target_adaptation", "strategy": strategy,
                                      "n_target": size, "uses_x1": include_x1, **row})
        splits["repetitions"][str(repetition)] = rep_splits
        print(f"{path.stem}: repetition {repetition}/{args.repetitions} terminee", flush=True)
    return results, splits, histories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--split-seed", type=int, default=31001)
    parser.add_argument("--sampling-seed", type=int, default=41001)
    parser.add_argument("--training-seed", type=int, default=51001)
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_epochs < 1 or args.patience < 1:
        parser.error("repetitions, max-epochs et patience doivent etre positifs")
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calcul sur {device}")
    all_results, all_splits, all_histories = [], {}, []
    for name in ("model_1.csv", "model_2.csv"):
        path = args.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} absent : lancer simulate_data.py d'abord.")
        results, splits, histories = run_dataset(path, args, device)
        all_results.extend(results)
        all_histories.extend(histories)
        all_splits[path.stem] = splits
    pd.DataFrame(all_results).to_csv(args.output_dir / "rmse_results.csv", index=False)
    pd.DataFrame(all_histories).to_csv(args.output_dir / "learning_curves.csv", index=False)
    (args.output_dir / "splits.json").write_text(json.dumps(all_splits, indent=2) + "\n", encoding="utf-8")
    run_info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "device": str(device), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "arguments": vars(args)}
    run_info["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in run_info["arguments"].items()}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"Resultats ecrits dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
