# 📡 API de Trading (FastAPI)

Le cerveau du système. Cette API expose tous les services nécessaires au pipeline de trading : gestion des données, calcul d'indicateurs, entraînement des modèles et prédictions.

## 🚀 Démarrage Rapide

```bash
cd API
uvicorn app.main:app --reload
```
Documentation interactive (Swagger UI) disponible sur : **http://localhost:8000/docs**

## 🔧 Architecture

L'application est structurée autour de l'architecture **Router-Service-Repository** :

-   **Routers** (`app/routers/`) : Gèrent les requêtes HTTP et la validation des données (Pydantic).
-   **Services** (`app/services/`) : Contiennent la logique métier pure (Nettoyage Pandas, Entraînement Scikit-learn, Q-Learning).
-   **Repositories** (`app/repositories/`) : Gèrent la persistance des données (ici, `In-Memory Dataset Store` pour la performance).

## 🔌 Modules Principaux

### 1. Data Processing (`/m15`, `/features`)
-   Chargement des CSV bruts.
-   Agrégation temporelle (Resampling M1 -> M15).
-   Calcul des indicateurs techniques (RSI, EMA, ATR, MACD...).

### 2. Machine Learning (`/trading_ml`)
-   **Train** : Entraînement avec validation croisée temporelle (Grid Search).
-   **Predict** : Génération de signaux et d'explications ("Explainable AI").
-   **Models** : RandomForest, GradientBoosting, LogisticRegression.

### 3. Reinforcement Learning (`/rl`)
-   Environnement de trading personnalisé type Gym (`TradingEnv`).
-   Agent Q-Learning tabulaire optimisé.
-   Streaming des métriques d'entraînement via SSE (Server-Sent Events).

### 4. Évaluation (`/evaluate`, `/baseline`)
-   Comparaison multi-stratégies (Sharpe Ratio, Max Drawdown).
-   Stress Tests sur périodes volatiles.

## ⚠️ Notes Importantes
-   **Split Temporel** : L'API impose une séparation stricte des données pour éviter le *Data Leakage*.
    -   Train : 2022
    -   Val : 2023
    -   Test : 2024
-   **Stockage** : Les datasets transformés sont stockés en RAM pour la rapidité. Si l'API redémarre, il faut relancer le pipeline d'import.
