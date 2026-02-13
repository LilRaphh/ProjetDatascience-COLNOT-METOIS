# 📘 Documentation du Système de Trading Algorithmique GBP/USD

## 🌟 Introduction
Ce projet est un **système de décision algorithmique** conçu pour trader automatiquement la paire de devises **GBP/USD** (Livre Sterling contre Dollar US).

Imaginez ce système comme une usine numérique qui transforme des données brutes en décisions d'investissement intelligentes. Il utilise deux formes d'Intelligence Artificielle pour maximiser les profits tout en maîtrisant les risques.

---

## 🚀 Le Pipeline de Données (La Chaîne de Fabrication)

Avant de pouvoir prendre des décisions, le système doit traiter et comprendre les données du marché. C'est ce qu'on appelle le **Pipeline**.

### 1. Importation (La Matière Première)
Nous récupérons l'historique des prix minute par minute (**M1**). C'est la donnée la plus brute : à quel prix s'échangeait la devise à 10h01, 10h02, etc.

### 2. Agrégation (Le Raffinage)
Traiter chaque minute est trop bruyant (trop de mouvements aléatoires). Nous regroupons les données par paquets de **15 minutes (M15)**.
*   *Terme Technique* : **OHLC** (Open, High, Low, Close). Pour chaque 15 min, on garde le prix d'ouverture, le plus haut, le plus bas et la clôture.

### 3. Nettoyage (Le Contrôle Qualité)
Les données financières contiennent parfois des erreurs (prix manquants, trous de cotation). Le système détecte et corrige ces anomalies pour ne pas biaiser l'apprentissage.

### 4. Feature Engineering (L'Enrichissement)
C'est l'étape cruciale où l'on transforme le prix brut en **indicateurs** compréhensibles par l'IA.
*   **Tendance** : Est-ce que ça monte ou descend sur le long terme ? (via *EMA - Moyennes Mobiles Exponentielles*)
*   **Momentum** : Est-ce que le mouvement accélère ? (via *RSI - Relative Strength Index*)
*   **Volatilité** : Est-ce que le marché est calme ou nerveux ? (via *ATR - Average True Range*)

---

## 🧠 Les Cerveaux du Système

Le système utilise deux approches d'IA complémentaires.

### 🤖 1. Machine Learning (L'Analyste)
Ce module agit comme un analyste financier qui regarde des milliers de graphiques passés.
*   **Son rôle** : Prédire si la prochaine bougie de 15 min sera verte (hausse) ou rouge (baisse).
*   **Son outil** : Un **Random Forest** (Forêt Aléatoire). Imaginez 200 experts qui votent chacun sur la direction du marché. La décision finale est prise à la majorité.
*   **Optimisation** : Nous utilisons une méthode appelée **Grid Search** pour trouver les meilleurs réglages de ces 200 experts (profondeur d'analyse, sensibilité, etc.).

### 🎮 2. Reinforcement Learning (Le Trader)
Ce module agit comme un trader junior qui apprend par l'expérience.
*   **Son rôle** : Décider s'il faut acheter, vendre ou ne rien faire, en tenant compte de son capital et des risques.
*   **Son outil** : Le **Q-Learning**. C'est un système de récompense/punition.
    *   S'il gagne de l'argent → Il reçoit une "récompense" (+1).
    *   S'il perd ou prend trop de risques → Il reçoit une "punition" (-1).
    *   *Concept Clé* : **Exploration vs Exploitation**. Au début, il tente des choses au hasard (exploration), puis petit à petit, il n'utilise que les stratégies qui ont fonctionné (exploitation).

---

## 🛡️ Gestion du Risque et Évaluation

Comment savoir si le système est performant ? Nous utilisons des mesures précises (métriques).

### 📅 Le Split Temporel (La Règle d'Or)
Pour ne pas tricher, nous découpons le temps strictement :
*   **2022 (Train)** : L'IA étudie cette année-là.
*   **2023 (Validation)** : On vérifie si elle a bien appris sur une année qu'elle n'a jamais vue.
*   **2424 (Test)** : L'examen final. On lance l'IA dans le grand bain.

### 📊 Les indicateurs de performance
*   **Sharpe Ratio** : Le juge de paix. Il mesure le rendement par unité de risque.
    *   *Analogie* : Rouler à 100km/h sur autoroute (bon Sharpe) vs rouler à 100km/h en ville (mauvais Sharpe). Plus il est haut, mieux c'est (viser > 1.0).
*   **Max Drawdown** : La pire chute. C'est la perte maximale que le portefeuille a subie depuis son sommet historique.
    *   *Analogie* : La "douleur" maximale ressentie par l'investisseur. On veut ce chiffre le plus bas possible.
*   **Equity Curve** : La courbe de votre compte en banque au fil du temps. On veut qu'elle monte régulièrement, sans trop de secousses.

---

## 💻 Guide de l'Interface

1.  **🚀 Pipeline Automatique** : Le panneau de contrôle principal.
    *   Sélectionnez les années, cochez "Optimiser" (pour un meilleur cerveau ML), et cliquez sur "Lancer".
    *   Le système fera tout le travail (Import -> ML -> RL -> Évaluation).
2.  **📈 Graphiques** :
    *   Page **RL** : Suivez l'entraînement du trader en direct. Barres bleues = Gains, Ligne verte = Capital total.
    *   Page **Évaluation** : Comparez votre IA contre des stratégies basiques (comme "Acheter et garder").

---
*Projet réalisé par COLNOT & MÉTOIS — Data Science & Trading Algorithmique — Février 2026*
