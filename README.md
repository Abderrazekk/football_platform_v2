# Football Intelligence Platform (v3.0)

Projet commun Machine Learning & Django — Analyse, segmentation et prédiction des performances de Premier League (2020-21 à 2026-27).

## Installation (Windows 11)
1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Copier `.env.example` vers `.env` et renseigner `SECRET_KEY`.
5. Placer les dossiers de saisons dans `data/raw/`.
6. `python manage.py migrate`
7. `python manage.py runserver`

## Ordre d'exécution des Notebooks ML (`ml/`)
Exécuter chaque notebook via `Kernel -> Restart & Run All` :
1. `ml/match_model/01_data_loading.ipynb` -> `10_save_final_model.ipynb`
2. `ml/clustering_model/01_team_player_clustering.ipynb`
3. `ml/player_model/01_player_preprocessing.ipynb` -> `04_player_evaluation_save.ipynb`
4. `ml/season_model/01_season_features.ipynb` -> `02_season_train_eval_save.ipynb`