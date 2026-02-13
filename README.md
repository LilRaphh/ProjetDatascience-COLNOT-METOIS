# 📈 Système de Trading Algorithmique GBP/USD

**Projet Data Science M1 — Universite de Lorraine**
*Auteurs : COLNOT & MÉTOIS*
*Date : Février 2026*

Ce projet implémente un pipeline complet de trading algorithmique, de l'ingestion des données brutes (M1) à la prise de décision par Intelligence Artificielle (Machine Learning & Reinforcement Learning).

## 🚀 Fonctionnalités Clés

-   **Pipeline Automatisé** : Import → Agrégation M15 → Nettoyage → Features → ML/RL → Évaluation.
-   **Dashboard Interactif** : Interface web moderne avec "Live Trading Desk" pour suivre les signaux en temps réel.
-   **Machine Learning** : Modèles (RandomForest, GBM) avec optimisation via Grid Search.
-   **Reinforcement Learning** : Agent Q-Learning entraîné sur l'environnement de marché.
-   **Interprétabilité** : Explication en langage naturel des décisions de l'IA.

## 🛠️ Stack Technique

-   **Backend** : Python 3.10+, FastAPI, Pandas, Scikit-learn, Numpy.
-   **Frontend** : HTML5, CSS3 (Grid/Flexbox), Vanilla JS, Chart.js.
-   **Données** : Historique GBP/USD (M1).

## 📦 Installation & Lancement

### 1. Prérequis
-   Python 3.10 ou supérieur
-   Navigateur web moderne

### 2. Installation
```bash
# Cloner le projet (si applicable) ou extraire l'archive
cd ProjetDatascience-COLNOT-METOIS

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Lancer l'API
```bash
cd API
uvicorn app.main:app --reload
```
*L'API sera accessible sur `http://localhost:8000`.*

### 4. Lancer l'Interface
Ouvrez simplement le fichier `web-interface/index.html` dans votre navigateur.
*Pas besoin de serveur web pour le frontend, il communique directement avec l'API locale.*

## 📂 Structure du Projet

```
.
├── API/                 # Backend FastAPI
│   ├── app/
│   │   ├── routers/     # Endpoints (M15, Features, ML, RL, Eval)
│   │   ├── services/    # Logique métier (TradingEnv, MLService...)
│   │   └── main.py      # Point d'entrée
│   └── data/            # Stockage des fichiers CSV (M1, datasets)
├── web-interface/       # Frontend
│   ├── css/             # Styles (Thème Dark/Blue)
│   ├── js/              # Logique UI et appels API
│   └── index.html       # Dashboard principal
├── documentation.md     # Documentation détaillée du projet
└── requirements.txt     # Dépendances Python
```

## 📘 Documentation
Pour plus de détails sur le fonctionnement interne, les algos utilisés et la méthodologie, consultez le fichier [documentation.md](./documentation.md).
