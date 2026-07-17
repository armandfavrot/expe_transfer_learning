"""Produit les figures des donnees simulees, des RMSE et des courbes d'apprentissage."""

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
    grid.savefig(output_dir / "distribution_y_par_domaine.pdf", bbox_inches="tight")
    plt.close(grid.figure)

    continuous = data.melt(
        id_vars=["X1", "Modele generateur"], value_vars=["X3", "X4"],
        var_name="Variable", value_name="Valeur",
    )
    grid = sns.displot(
        data=continuous, x="Valeur", hue="X1", hue_order=["a1", "b1"],
        row="Variable", col="Modele generateur",
        kind="kde", fill=True, common_norm=False, height=3.2, aspect=1.3,
        facet_kws={"sharex": False, "sharey": False},
    )
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.set_axis_labels("Valeur", "Densite")
    grid.figure.suptitle("Distributions de X3 et X4 selon le domaine", y=1.02)
    grid.figure.set_facecolor("white")
    grid.savefig(output_dir / "distributions_x3_x4_par_domaine.pdf", bbox_inches="tight", facecolor="white")
    grid.savefig(output_dir / "distributions_x3_x4_par_domaine.svg", bbox_inches="tight", facecolor="white")
    plt.close(grid.figure)


def plot_rmse(results_path: Path, output_dir: Path) -> None:
    results = pd.read_csv(results_path)
    results["Strategie"] = results["strategy"].map(LABELS)
    results["Modele generateur"] = results["dataset"].map({"model_1": "Modele 1", "model_2": "Modele 2"})
    results["n cible"] = results["n_target"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    palette = sns.color_palette("colorblind", 3)
    order = ["From scratch", "Fine-tuning", "Combine"]
    for ax, model in zip(axes, ("Modele 1", "Modele 2"), strict=True):
        subset = results[results["Modele generateur"] == model]
        oracle_by_repetition = subset.groupby("repetition")["oracle_rmse"].first()
        oracle_rmse = oracle_by_repetition.mean()
        sns.boxplot(data=subset, x="n cible", y="rmse", hue="Strategie", hue_order=order,
                    palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=subset, x="n cible", y="rmse", hue="Strategie", hue_order=order,
                      dodge=True, palette=palette, alpha=0.75, size=4, linewidth=0.3, ax=ax)
        ax.set(title=model, xlabel="Taille de l'echantillon cible", ylabel="RMSE")
        handles, labels = ax.get_legend_handles_labels()
        oracle_line = ax.axhline(oracle_rmse, color="black", linestyle="--", linewidth=1.5,
                                 label=f"Modele generateur (RMSE oracle moyenne = {oracle_rmse:.3f})")
        ax.legend(handles[:3] + [oracle_line], labels[:3] + [oracle_line.get_label()], title="Strategie")
    fig.suptitle("Performance sur les jeux de test cibles renouvelés")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_boxplots.pdf", bbox_inches="tight")
    # Apercu vectoriel affichable directement dans le README GitHub.
    fig.savefig(output_dir / "rmse_boxplots.svg", bbox_inches="tight")
    plt.close(fig)


def _use_log_scale(ax: plt.Axes, values: pd.Series) -> None:
    positive = values[values > 0]
    if not positive.empty and positive.max() / positive.min() >= 100:
        ax.set_yscale("log")


def plot_source_learning_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    """Trace les pertes train/validation des dix preentrainements source."""
    colors = {"train_mse": "#0072B2", "validation_mse": "#D55E00"}
    model_labels = {"model_1": "Modele 1", "model_2": "Modele 2"}
    source = curves[curves["training_type"] == "source_pretraining"]
    for dataset, subset in source.groupby("dataset", sort=True):
        repetitions = sorted(subset["repetition"].unique())
        ncols = 5
        nrows = (len(repetitions) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), squeeze=False)
        for ax, repetition in zip(axes.flat, repetitions, strict=False):
            current = subset[subset["repetition"] == repetition]
            ax.plot(current["epoch"], current["train_mse"], color=colors["train_mse"], label="Train")
            ax.plot(current["epoch"], current["validation_mse"], color=colors["validation_mse"], label="Validation")
            best = current.loc[current["is_best_epoch"].astype(bool), "epoch"]
            if not best.empty:
                ax.axvline(best.iloc[0], color="black", linestyle="--", linewidth=1, label="Meilleure epoque")
            _use_log_scale(ax, pd.concat((current["train_mse"], current["validation_mse"])))
            ax.set(title=f"Repetition {repetition}", xlabel="Epoque", ylabel="MSE")
        for ax in axes.flat[len(repetitions):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"Courbes d'apprentissage du preentrainement source — {model_labels.get(dataset, dataset)}", y=1.02)
        fig.tight_layout()
        stem = output_dir / f"learning_curves_source_{dataset}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def plot_target_learning_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    """Trace une grille strategie x taille avec les repetitions et leur mediane."""
    colors = {"train_mse": "#0072B2", "validation_mse": "#D55E00"}
    strategies = ["fine_tuning", "from_scratch", "combined"]
    sizes = sorted(int(v) for v in curves["n_target"].dropna().unique())
    model_labels = {"model_1": "Modele 1", "model_2": "Modele 2"}
    target = curves[curves["training_type"] == "target_adaptation"]
    for dataset, subset in target.groupby("dataset", sort=True):
        fig, axes = plt.subplots(len(strategies), len(sizes), figsize=(4 * len(sizes), 3.5 * len(strategies)),
                                 squeeze=False, sharex=False, sharey=False)
        for row, strategy in enumerate(strategies):
            for col, size in enumerate(sizes):
                ax = axes[row, col]
                current = subset[(subset["strategy"] == strategy) & (subset["n_target"] == size)]
                for _, repetition in current.groupby("repetition"):
                    ax.plot(repetition["epoch"], repetition["train_mse"], color=colors["train_mse"], alpha=0.18, linewidth=0.8)
                    ax.plot(repetition["epoch"], repetition["validation_mse"], color=colors["validation_mse"], alpha=0.18, linewidth=0.8)
                medians = current.groupby("epoch", as_index=False)[["train_mse", "validation_mse"]].median()
                ax.plot(medians["epoch"], medians["train_mse"], color=colors["train_mse"], linewidth=2, label="Train (mediane)")
                ax.plot(medians["epoch"], medians["validation_mse"], color=colors["validation_mse"], linewidth=2, label="Validation (mediane)")
                _use_log_scale(ax, pd.concat((current["train_mse"], current["validation_mse"])))
                if row == 0:
                    ax.set_title(f"n = {size}")
                if col == 0:
                    ax.set_ylabel(f"{LABELS[strategy]}\nMSE")
                else:
                    ax.set_ylabel("MSE")
                if row == len(strategies) - 1:
                    ax.set_xlabel("Epoque")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Courbes d'apprentissage des strategies cibles — {model_labels.get(dataset, dataset)}", y=1.01)
        fig.tight_layout()
        stem = output_dir / f"learning_curves_target_{dataset}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def plot_learning_curves(curves_path: Path, output_dir: Path) -> None:
    curves = pd.read_csv(curves_path)
    required = {"dataset", "repetition", "training_type", "strategy", "n_target", "epoch",
                "train_mse", "validation_mse", "is_best_epoch"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {curves_path}: {sorted(missing)}")
    plot_source_learning_curves(curves, output_dir)
    plot_target_learning_curves(curves, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results/rmse_results.csv"))
    parser.add_argument("--learning-curves", type=Path, default=Path("results/learning_curves.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--data-only", action="store_true", help="Ne trace pas les résultats de l’expérience.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plot_simulated(args.data_dir, args.output_dir)
    if not args.data_only:
        if not args.results.exists():
            raise FileNotFoundError(f"{args.results} absent : lancer run_experiment.py d'abord.")
        plot_rmse(args.results, args.output_dir)
        if not args.learning_curves.exists():
            raise FileNotFoundError(f"{args.learning_curves} absent : lancer run_experiment.py d'abord.")
        plot_learning_curves(args.learning_curves, args.output_dir)
    print(f"Figures creees dans {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
