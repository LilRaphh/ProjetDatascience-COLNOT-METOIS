"""
Script de diagnostic pour vérifier les datasets avant ML training
"""

# Copiez ce code dans la console Python de votre API ou dans un notebook

# 1. Vérifier qu'un dataset existe
from app.repositories.dataset_store import dataset_store

# Remplacez par votre dataset_id
dataset_id = "VOTRE_DATASET_ID_ICI"  # ex: "m1_2022_abc_m15_clean_features"

if dataset_store.exists(dataset_id):
    entry = dataset_store.get(dataset_id)
    df = entry.df
    
    print(f"✅ Dataset trouvé: {dataset_id}")
    print(f"📊 Shape: {df.shape}")
    print(f"\n📋 Colonnes disponibles:")
    print(df.columns.tolist())
    
    # Vérifier la colonne target
    if 'target' in df.columns:
        print(f"\n✅ Colonne 'target' présente")
        print(f"   - Valeurs uniques: {df['target'].unique()}")
        print(f"   - NaN: {df['target'].isna().sum()}")
    else:
        print(f"\n❌ Colonne 'target' MANQUANTE")
        print(f"   → Vous devez calculer les features avec add_target=True")
    
    # Vérifier les features
    from app.services.feature_service import FeatureService
    expected_features = FeatureService.ALL_FEATURES
    missing_features = [f for f in expected_features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Features manquantes ({len(missing_features)}):")
        print(missing_features[:10])  # Afficher les 10 premières
    else:
        print(f"\n✅ Toutes les features sont présentes ({len(expected_features)})")
    
    # Vérifier les NaN
    if 'target' in df.columns:
        feat_cols = [c for c in expected_features if c in df.columns]
        valid = df[feat_cols + ['target']].dropna()
        print(f"\n📈 Lignes valides après dropna: {len(valid)} / {len(df)}")
        if len(valid) == 0:
            print(f"   ❌ PROBLÈME: Aucune ligne valide!")
        elif len(valid) < 100:
            print(f"   ⚠️  ATTENTION: Très peu de lignes valides")
else:
    print(f"❌ Dataset introuvable: {dataset_id}")
    print(f"\n📋 Datasets disponibles:")
    # Lister les datasets
    try:
        store_dict = dataset_store.__dict__.get("_datasets", {})
        for did in list(store_dict.keys())[:10]:
            print(f"   - {did}")
    except:
        print("   (impossible de lister)")
