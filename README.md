# Transfert d'apprentissage par fine-tuning en régression

Ce dépôt contient une expérience visant à évaluer le fine-tuning comme approche de transfert d'apprentissage dans un problème de régression à partir de données simulées.

Trois stratégies sont comparées :

- le **fine-tuning** d'un MLP préentraîné sur le domaine source ;
- l'apprentissage sur le domaine cible uniquement (**from scratch**) ;
- l'apprentissage conjoint sur les données source et cible (**combined**).

L'expérience est menée sur deux jeux de données produits par deux modèles générateurs de complexités différentes.

## Résultats

![Comparaison des RMSE selon la stratégie, la taille de l'échantillon cible et le modèle générateur](figures/rmse_boxplots.svg)

[Ouvrir la figure des résultats en PDF](figures/rmse_boxplots.pdf)

Le panneau de gauche correspond au premier jeu de données, issu du modèle générateur 1, et le panneau de droite au second jeu de données, issu du modèle générateur 2. Chaque boxplot résume dix répétitions. Les points représentent les RMSE individuelles obtenues sur le jeu de test cible renouvelé à chaque répétition. La ligne horizontale pointillée indique la moyenne des dix RMSE du modèle générateur exact sur ces jeux de test.

Les résultats mettent principalement en évidence l'intérêt du transfert pour les plus petits échantillons cibles. L'écart entre les stratégies diminue lorsque davantage de données cibles sont disponibles. Le second mécanisme générateur, plus complexe, conserve une erreur légèrement plus élevée.

## Protocole expérimental

Chaque jeu simulé contient 2 000 observations : 1 000 dans le domaine source et 1 000 dans le domaine cible. Quatre variables explicatives sont simulées : `X1` indique le domaine, `X2` est une variable catégorielle à trois modalités, et `X3` et `X4` sont deux covariables continues. Un décalage modéré des distributions de `X3` et `X4` est introduit entre les domaines afin de créer un *covariate shift*.

L'expérience comprend dix répétitions. À chaque répétition :

- les 1 000 observations cibles sont à nouveau réparties entre un jeu de test de 500 observations et un réservoir d'apprentissage de 500 observations ;
- les 1 000 observations source sont à nouveau réparties entre 800 observations d'entraînement et 200 de validation ;
- la standardisation de `X3` et `X4` est recalculée uniquement à partir du nouveau train source ;
- des échantillons cibles emboîtés de tailles 10, 50, 100, 200 et 500 sont construits.

Les trois stratégies partagent, au sein d'une répétition et pour une taille cible donnée, les mêmes observations cibles d'entraînement et de validation ainsi que le même jeu de test cible.

Les réseaux sont implémentés avec PyTorch :

- le modèle source, le fine-tuning et le modèle *from scratch* utilisent `X2`, `X3` et `X4` ;
- le modèle combiné utilise également la variable de domaine `X1` ;
- les variables catégorielles utilisées par chaque réseau sont représentées par des embeddings.

Tous les entraînements utilisent un arrêt anticipé fondé sur un jeu de validation. Le protocole complet, les modèles générateurs et les hyperparamètres sont détaillés dans [protocole.md](protocole.md).

## Installation

Le projet utilise Python 3.12.3. Il est conseillé de travailler dans un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduire l'expérience

Depuis la racine du dépôt :

```bash
# 1. Simuler les deux jeux de données
python simulate_data.py

# 2. Entraîner et évaluer les modèles
python run_experiment.py --device auto

# 3. Produire les figures
python visualize_results.py
```

L'option `--device auto` utilise CUDA lorsqu'il est disponible et se replie sinon sur le CPU. Pour imposer un calcul GPU, utiliser `--device cuda`.

Un test rapide de la chaîne de calcul peut être lancé avec :

```bash
python run_experiment.py \
  --device auto \
  --repetitions 1 \
  --max-epochs 2 \
  --patience 1 \
  --output-dir results_smoke

python visualize_results.py \
  --results results_smoke/rmse_results.csv \
  --learning-curves results_smoke/learning_curves.csv \
  --output-dir figures_smoke
```

Le test rapide vérifie le fonctionnement de la chaîne, mais ses résultats ne doivent pas être interprétés scientifiquement.

## Fichiers produits

La simulation produit :

- `data/model_1.csv` et `data/model_2.csv` : données simulées ;
- `data/simulation_metadata.json` : paramètres et graine de simulation.

L'expérience complète produit :

- `results/rmse_results.csv` : RMSE des réseaux, RMSE oracle et informations d'arrêt anticipé ;
- `results/learning_curves.csv` : MSE d'entraînement et de validation à chaque époque ;
- `results/splits.json` : indices, graines et statistiques de standardisation propres à chaque répétition ;
- `results/run_metadata.json` : versions, matériel et arguments d'exécution.

La visualisation produit notamment :

- `figures/rmse_boxplots.pdf` et `.svg` : comparaison des RMSE ;
- `figures/distribution_y_par_domaine.pdf` : distributions de la réponse ;
- `figures/distributions_x3_x4_par_domaine.pdf` et `.svg` : distributions des covariables selon le domaine ;
- `figures/learning_curves_source_model_*.pdf` et `.svg` : courbes des préentraînements source ;
- `figures/learning_curves_target_model_*.pdf` et `.svg` : courbes des trois stratégies cibles.

Les données et résultats intermédiaires peuvent être entièrement régénérés à partir des scripts et des graines enregistrées.

## Organisation du dépôt

```text
simulate_data.py       Simulation des deux jeux de données
run_experiment.py      Entraînement, évaluation et sauvegarde des résultats
visualize_results.py   Création des figures
protocole.md           Description détaillée du protocole expérimental
README_experience.md   Guide d'exécution condensé
requirements.txt       Dépendances Python
```
