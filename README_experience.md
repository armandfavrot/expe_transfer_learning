# Execution de l'experience

Les commandes sont a lancer depuis la racine du projet.

## 1. Simuler les donnees

```bash
python simulate_data.py
```

Cette commande cree `data/model_1.csv`, `data/model_2.csv` et les metadonnees de
simulation. Pour visualiser uniquement ces donnees :

```bash
python visualize_results.py --data-only
```

## 2. Lancer les calculs

```bash
python run_experiment.py --device auto
```

Le lancement complet applique les 10 repetitions, 500 epoques au maximum et
l'arret anticipe du protocole. `--device auto` choisit automatiquement CUDA lorsqu'il est disponible, puis le CPU en solution de repli. `--device cuda` provoque une erreur explicite si CUDA n'est pas disponible.

Un test rapide de la chaine de calcul peut etre effectue avant le lancement long :

```bash
python run_experiment.py --device auto --repetitions 1 --max-epochs 2 --patience 1 --output-dir results_smoke
```

Pour visualiser ce test sans remplacer les figures principales :

```bash
python visualize_results.py --results results_smoke/rmse_results.csv --learning-curves results_smoke/learning_curves.csv --output-dir figures_smoke
```

Le calcul complet produit :

- `results/rmse_results.csv` : RMSE des reseaux, RMSE oracle du modele generateur et informations d'arret anticipe ;
- `results/learning_curves.csv` : MSE d'entrainement et de validation a chaque epoque pour tous les entrainements ;
- `results/splits.json` : tous les indices des decoupages et les statistiques de standardisation ;
- `results/run_metadata.json` : versions, GPU, graines et arguments d'execution.

## 3. Visualiser les resultats

```bash
python visualize_results.py
```

Les figures sont enregistrees dans `figures/`, notamment les courbes des
preentrainements source et les grilles de courbes des trois strategies cibles.
Pour changer les dossiers, utiliser les options `--data-dir`, `--output-dir`,
`--results` et `--learning-curves`; chaque script expose
l'ensemble de ses options avec `--help`.
