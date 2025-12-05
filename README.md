⚡ Prévision de la Consommation Électrique en France (Time Series Project)

Ce projet analyse et modélise la consommation électrique française à partir des données Eco2mix (RTE).
L’objectif est d’appliquer plusieurs techniques de séries temporelles pour comprendre les tendances, la saisonnalité et prédire la consommation future.

📊 Objectifs du projet

Nettoyer et préparer les données (parsing datetime, gestion des fuseaux horaires, normalisation des colonnes).

Resampler la consommation en données journalières.

Visualiser l’évolution temporelle.

Diviser les données en train/test.

Appliquer plusieurs modèles de prévision :

Modèle naïf

Modèle naïf saisonnier

Moyenne mobile

SARIMA (ARIMA saisonnier)

Évaluer les modèles (MAE, RMSE).

Sélectionner le meilleur modèle pour la prédiction.

├── data/
│   └── eco2mix.csv
├── notebooks/
│   └── time_series_conso.ipynb
├── src/
│   ├── fonctions
├── README.md
└── requirements.txt

Métriques d'évaluation

Les modèles sont comparés via :

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

📌 Données

Les données proviennent de RTE Eco2mix :
https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-national-cons-def

📬 Contact

Projet réalisé dans le cadre d’un travail de recherche en séries temporelles.
Pour toute question : Donaldsedd@gmail.com
