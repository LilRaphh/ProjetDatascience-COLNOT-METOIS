"""
Service de génération de datasets
Génère des datasets reproductibles pour chaque phase du projet
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class DatasetGenerator:
    """
    Générateur de datasets pour les différentes phases du projet
    
    Chaque phase génère un dataset avec des caractéristiques spécifiques :
    - clean : données sales (missing, outliers, doublons)
    - eda : données pour analyse exploratoire
    - mv : données pour analyse multivariée (PCA, clustering)
    - ml : données pour machine learning (classification binaire)
    - ml2 : même que ml (réutilisable)
    """
    
    # Stockage en mémoire des datasets générés
    _datasets: Dict[str, pd.DataFrame] = {}
    
    @classmethod
    def generate(cls, phase: str, seed: int, n: int) -> Tuple[str, pd.DataFrame]:
        """
        Génère un dataset pour une phase donnée
        
        Args:
            phase: Phase du projet (clean, eda, mv, ml, ml2)
            seed: Graine aléatoire pour reproductibilité
            n: Nombre de lignes
            
        Returns:
            Tuple (dataset_id, dataframe)
        """
        # Créer un ID unique et reproductible
        dataset_id = f"{phase}_{seed}_{n}"
        
        # Vérifier si le dataset existe déjà
        if dataset_id in cls._datasets:
            return dataset_id, cls._datasets[dataset_id]
        
        # Générer le dataset selon la phase
        if phase == "clean":
            df = cls._generate_clean(seed, n)
        elif phase == "eda":
            df = cls._generate_eda(seed, n)
        elif phase == "mv":
            df = cls._generate_mv(seed, n)
        elif phase in ["ml", "ml2"]:
            df = cls._generate_ml(seed, n)
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        # Stocker le dataset
        cls._datasets[dataset_id] = df
        
        return dataset_id, df
    
    @classmethod
    def get_dataset(cls, dataset_id: str) -> pd.DataFrame:
        """
        Récupère un dataset déjà généré
        
        Args:
            dataset_id: Identifiant du dataset
            
        Returns:
            DataFrame
        """
        if dataset_id not in cls._datasets:
            raise ValueError(f"Dataset {dataset_id} introuvable. Générez-le d'abord.")
        return cls._datasets[dataset_id]
    
    @classmethod
    def _generate_clean(cls, seed: int, n: int) -> pd.DataFrame:
        """
        Génère un dataset pour TP1 (Clean)
        
        Variables :
        - x1, x2, x3 : numériques
        - segment : catégorielle (A, B, C)
        - target : binaire (0, 1)
        
        Défauts injectés :
        - Missing values : 10-20%
        - Doublons : 1-5%
        - Outliers : 1-3%
        - Types cassés : quelques valeurs non-numériques dans x2
        """
        np.random.seed(seed)
        
        # Générer données de base
        df = pd.DataFrame({
            'x1': np.random.normal(100, 20, n),
            'x2': np.random.normal(50, 10, n),
            'x3': np.random.exponential(30, n),
            'segment': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
            'target': np.random.binomial(1, 0.3, n)
        })
        
        # Injecter des missing values (10-20% selon colonnes)
        for col in ['x1', 'x2', 'x3']:
            missing_rate = np.random.uniform(0.10, 0.20)
            mask = np.random.random(n) < missing_rate
            df.loc[mask, col] = np.nan
        
        # Injecter des outliers (1-3% extrêmes)
        outlier_rate = 0.02
        n_outliers = int(n * outlier_rate)
        outlier_indices = np.random.choice(n, n_outliers, replace=False)
        df.loc[outlier_indices, 'x1'] = np.random.uniform(300, 400, n_outliers)
        
        # Injecter des types cassés dans x2 (valeurs non-numériques)
        n_broken = max(1, int(n * 0.01))
        broken_indices = np.random.choice(n, n_broken, replace=False)
        df.loc[broken_indices, 'x2'] = 'oops'
        
        # Injecter des doublons (1-5%)
        n_duplicates = int(n * 0.03)
        duplicate_indices = np.random.choice(n, n_duplicates, replace=True)
        df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
        
        return df
    
    @classmethod
    def _generate_eda(cls, seed: int, n: int) -> pd.DataFrame:
        """
        Génère un dataset pour TP2 (EDA)
        
        Variables :
        - age : numérique
        - income : numérique (avec outliers)
        - spend : numérique
        - visits : numérique
        - segment : catégorielle (A, B, C)
        - channel : catégorielle (web, store, app)
        - churn : binaire (0, 1) - optionnel
        
        Défauts injectés :
        - NA légers (5-10%)
        - Outliers sur income (1-2%)
        """
        np.random.seed(seed)
        
        df = pd.DataFrame({
            'age': np.random.normal(40, 15, n).clip(18, 80).astype(int),
            'income': np.random.lognormal(10.5, 0.8, n),
            'spend': np.random.gamma(2, 500, n),
            'visits': np.random.poisson(5, n),
            'segment': np.random.choice(['A', 'B', 'C'], n, p=[0.4, 0.35, 0.25]),
            'channel': np.random.choice(['web', 'store', 'app'], n, p=[0.5, 0.3, 0.2]),
            'churn': np.random.binomial(1, 0.25, n)
        })
        
        # Injecter NA légers (5-10%)
        for col in ['income', 'spend']:
            missing_rate = np.random.uniform(0.05, 0.10)
            mask = np.random.random(n) < missing_rate
            df.loc[mask, col] = np.nan
        
        # Injecter outliers sur income (1-2%)
        outlier_rate = 0.015
        n_outliers = int(n * outlier_rate)
        outlier_indices = np.random.choice(n, n_outliers, replace=False)
        df.loc[outlier_indices, 'income'] = np.random.uniform(200000, 500000, n_outliers)
        
        return df
    
    @classmethod
    def _generate_mv(cls, seed: int, n: int) -> pd.DataFrame:
        """
        Génère un dataset pour TP3 (Multivarié)
        
        Variables :
        - x1..x8 : numériques
        
        Caractéristiques :
        - 3 clusters simulés
        - Colinéarité (x5 ≈ x1 + bruit)
        - NA faibles (2-5%)
        - Pas de target
        """
        np.random.seed(seed)
        
        # Simuler 3 clusters
        n_per_cluster = n // 3
        
        # Cluster 1
        cluster1 = np.random.multivariate_normal(
            mean=[10, 5, 15, 20, 10, 25, 8, 12],
            cov=np.eye(8) * 2,
            size=n_per_cluster
        )
        
        # Cluster 2
        cluster2 = np.random.multivariate_normal(
            mean=[30, 25, 35, 40, 30, 45, 28, 32],
            cov=np.eye(8) * 2,
            size=n_per_cluster
        )
        
        # Cluster 3
        cluster3 = np.random.multivariate_normal(
            mean=[50, 45, 55, 60, 50, 65, 48, 52],
            cov=np.eye(8) * 2,
            size=n - 2*n_per_cluster  # Le reste
        )
        
        # Combiner les clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        
        df = pd.DataFrame(
            data,
            columns=[f'x{i}' for i in range(1, 9)]
        )
        
        # Ajouter colinéarité : x5 ≈ x1 + bruit
        df['x5'] = df['x1'] + np.random.normal(0, 1, n)
        
        # Injecter NA faibles (2-5%)
        for col in df.columns[:4]:
            missing_rate = np.random.uniform(0.02, 0.05)
            mask = np.random.random(n) < missing_rate
            df.loc[mask, col] = np.nan
        
        return df
    
    @classmethod
    def _generate_ml(cls, seed: int, n: int) -> pd.DataFrame:
        """
        Génère un dataset pour TP4 et TP5 (ML)
        
        Variables :
        - x1..x6 : numériques
        - segment : catégorielle (A, B, C)
        - target : binaire (0, 1)
        
        Caractéristiques :
        - Classification binaire
        - Déséquilibre contrôlé (70/30)
        - Bruit
        - Split reproductible via seed
        """
        np.random.seed(seed)
        
        # Générer features numériques
        X = np.random.randn(n, 6)
        
        # Générer segment catégoriel
        segment = np.random.choice(['A', 'B', 'C'], n, p=[0.4, 0.35, 0.25])
        
        # Générer target avec déséquilibre (70/30)
        # Target dépend des features avec du bruit
        z = (
            0.5 * X[:, 0] + 
            0.3 * X[:, 1] - 
            0.2 * X[:, 2] + 
            0.4 * X[:, 3] +
            np.random.randn(n) * 0.5  # Bruit
        )
        
        # Convertir en probabilité puis en classe binaire
        prob = 1 / (1 + np.exp(-z))
        # Ajuster pour avoir ~30% de 1
        threshold = np.percentile(prob, 70)
        target = (prob > threshold).astype(int)
        
        df = pd.DataFrame(
            X,
            columns=[f'x{i}' for i in range(1, 7)]
        )
        df['segment'] = segment
        df['target'] = target
        
        return df


# Stockage des pipelines de nettoyage (pour TP1)
_cleaners: Dict[str, Dict[str, Any]] = {}

def store_cleaner(cleaner_id: str, cleaner_data: Dict[str, Any]):
    """Stocke un pipeline de nettoyage"""
    _cleaners[cleaner_id] = cleaner_data

def get_cleaner(cleaner_id: str) -> Dict[str, Any]:
    """Récupère un pipeline de nettoyage"""
    if cleaner_id not in _cleaners:
        raise ValueError(f"Cleaner {cleaner_id} introuvable")
    return _cleaners[cleaner_id]


# Stockage des modèles ML (pour TP4 et TP5)
_models: Dict[str, Dict[str, Any]] = {}

def store_model(model_id: str, model_data: Dict[str, Any]):
    """Stocke un modèle ML"""
    _models[model_id] = model_data

def get_model(model_id: str) -> Dict[str, Any]:
    """Récupère un modèle ML"""
    if model_id not in _models:
        raise ValueError(f"Model {model_id} introuvable")
    return _models[model_id]
