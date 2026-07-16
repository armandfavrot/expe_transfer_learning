"""Produit les figures des donnees simulees et des RMSE."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABELS = {"fine_tuning": "Fine-tuning", "from_scratch": "From scratch", "combined": "Combine"}


def plot_simulated(data_dir: Path, output_dir: Path) -> None:
    frames = []
    for name, label in (("model_1.csv", "Modele 1"), ("model_2.csv", "Modele 2")):
        frame = pd.read_csv(data_dir / name)
        frame["Modele generateur"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    grid = sns.displot(data=data, x="Y", hue="X1", col="Modele generateur", kind="kde",
                       fill=True, common_norm=False, height=4, aspect=1.15)
    grid.set_axis_labels("Y", "Densite")
    grid.figure.suptitle("Distribution de Y selon le domaine", y=1.04)
    grid.savefig(output_dir / "distribution_y_par_domaine.png", dpi=300, bbox_inches="tight")
    plt.close(grid.figure)


def plot_rmse(results_path: Path, output_dir: Path) -> None:
    results = pd.read_csv(results_path)
    results["Strategie"] = results["strategy"].map(LABELS)
    results["Modele generateur"] = results["dataset"].map({"model_1": "Modele 1", "model_2": "Modele 2"})
    results["n cible"] = results["n_target"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    palette = sns.color_palette("colorblind", 3)
    order = ["Fine-tuning", "From scratch", "Combine"]
    for ax, model in zip(axes, ("Modele 1", "Modele 2"), strict=True):
        subset = results[results["Modele generateur"] == model]
        sns.boxplot(data=subset, x="n cible", y="rmse", hue="Strategie", hue_order=order,
                    palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=subset, x="n cible", y="rmse", hue="Strategie", hue_order=order,
                      dodge=True, palette=palette, alpha=0.75, size=4, linewidth=0.3, ax=ax)
        ax.set(title=model, xlabel="Taille de l'echantillon cible", ylabel="RMSE")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="Strategie")
    fig.suptitle("Performance sur le jeu de test cible fixe")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results/rmse_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--data-only", action="store_true", help="Ne trace pas les RMSE.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plot_simulated(args.data_dir, args.output_dir)
    if not args.data_only:
        if not args.results.exists():
            raise FileNotFoundError(f"{args.results} absent : lancer run_experiment.py d'abord.")
        plot_rmse(args.results, args.output_dir)
    print(f"Figures creees dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
