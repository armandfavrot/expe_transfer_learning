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

## 2. Lancer les calculs sur GPU

```bash
python run_experiment.py --device cuda
```

Le lancement complet applique les 10 repetitions, 500 epoques au maximum et
l'arret anticipe du protocole. `--device cuda` provoque une erreur explicite si
CUDA n'est pas disponible ; `--device auto` choisit automatiquement CUDA puis le
CPU en solution de repli.

Un test rapide de la chaine de calcul peut etre effectue avant le lancement long :

```bash
python run_experiment.py --device cuda --repetitions 1 --max-epochs 2 --patience 1 --output-dir results_smoke
```

Le calcul complet produit :

- `results/rmse_results.csv` : RMSE des reseaux, RMSE oracle du modele generateur et informations d'arret anticipe ;
- `results/splits.json` : tous les indices des decoupages et les statistiques de standardisation ;
- `results/run_metadata.json` : versions, GPU, graines et arguments d'execution.

## 3. Visualiser les resultats

```bash
python visualize_results.py
```

Les figures sont enregistrees dans `figures/`. Pour changer les dossiers, utiliser
les options `--data-dir`, `--output-dir` et `--results`; chaque script expose
l'ensemble de ses options avec `--help`.
