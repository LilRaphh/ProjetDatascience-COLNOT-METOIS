"""
Service Machine Learning Baseline (TP4 - ML)
Entraînement, métriques, prédictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MlService:
    """
    Service pour Machine Learning baseline
    
    Fonctionnalités :
    - Entraînement de modèles (LogisticRegression, RandomForest)
    - Preprocessing automatique (scaling, encoding)
    - Calcul de métriques complètes
    - Prédictions avec probabilités
    """
    
    @staticmethod
    def train(df: pd.DataFrame, model_type: str, test_size: float = 0.2) -> Tuple[str, Dict[str, Any]]:
        """
        Entraîne un modèle de classification binaire
        
        Args:
            df: DataFrame avec features + target
            model_type: Type de modèle (logreg, rf)
            test_size: Proportion du test set
            
        Returns:
            Tuple (model_id, model_data)
        """
        # Vérifier que target existe
        if 'target' not in df.columns:
            raise ValueError("Colonne 'target' manquante dans le dataset")
        
        # Séparer features et target
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Identifier colonnes numériques et catégorielles
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Preprocessing
        X_processed = X.copy()
        
        # Encoder catégorielles (one-hot)
        if len(categorical_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        # Imputer NA (mean pour numériques)
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement du modèle
        if model_type == "logreg":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Probabilités (si disponible)
        try:
            y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        except:
            y_train_proba = None
            y_test_proba = None
        
        # Calculer métriques
        metrics_train = MlService._compute_metrics(y_train, y_train_pred, y_train_proba)
        metrics_test = MlService._compute_metrics(y_test, y_test_pred, y_test_proba)
        
        # Générer model_id
        model_id = f"model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Stocker les métadonnées du modèle
        model_data = {
            "model_id": model_id,
            "model_type": model_type,
            "model_object": model,
            "scaler": scaler,
            "features": X_processed.columns.tolist(),
            "feature_names_in": X.columns.tolist(),
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "train": metrics_train,
                "test": metrics_test
            },
            "hyperparams": model.get_params(),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test)
        }
        
        return model_id, model_data
    
    @staticmethod
    def _compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """
        Calcule les métriques de classification
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)
            
        Returns:
            Dictionnaire de métriques
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        # AUC si probabilités disponibles
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0
        }
        
        return metrics
    
    @staticmethod
    def predict(model_data: Dict[str, Any], X: pd.DataFrame) -> Dict[str, Any]:
        """
        Fait des prédictions avec un modèle entraîné
        
        Args:
            model_data: Données du modèle (de train())
            X: Features (sans target)
            
        Returns:
            Dictionnaire avec prédictions et probabilités
        """
        # Récupérer le modèle et le scaler
        model = model_data["model_object"]
        scaler = model_data["scaler"]
        categorical_cols = model_data["categorical_cols"]
        features = model_data["features"]
        
        # Preprocessing
        X_processed = X.copy()
        
        # Encoder catégorielles
        if len(categorical_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        # S'assurer d'avoir toutes les features (ajouter colonnes manquantes avec 0)
        for col in features:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        # Réordonner les colonnes
        X_processed = X_processed[features]
        
        # Imputer NA
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                X_processed[col] = X_processed[col].fillna(0)
        
        # Standardisation
        X_scaled = scaler.transform(X_processed)
        
        # Prédictions
        predictions = model.predict(X_scaled)
        
        # Probabilités
        try:
            probabilities = model.predict_proba(X_scaled)
            proba_class_1 = probabilities[:, 1].tolist()
        except:
            proba_class_1 = None
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": proba_class_1,
            "n_predictions": len(predictions)
        }
    
    @staticmethod
    def get_model_info(model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retourne les informations sur un modèle
        
        Args:
            model_data: Données du modèle
            
        Returns:
            Dictionnaire d'informations
        """
        return {
            "model_id": model_data["model_id"],
            "model_type": model_data["model_type"],
            "features": model_data["features"],
            "n_features": len(model_data["features"]),
            "created_at": model_data["created_at"],
            "hyperparams": model_data["hyperparams"],
            "n_samples_train": model_data["n_samples_train"],
            "n_samples_test": model_data["n_samples_test"]
        }
