"""
Application FastAPI - Projet Data Science
Système de décision de trading GBP/USD
Point d'entrée principal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Routers existants ──────────────────────────────────────────────────────
from app.routers.clean import router as clean_router
from app.routers.dataset import router as dataset_router
from app.routers import m15

# ── Nouveaux routers (Feature Engineering, EDA, Baseline, ML, RL, Eval) ───
from app.routers.features import router as features_router
from app.routers.eda import router as eda_router
from app.routers.baseline import router as baseline_router
from app.routers.trading_ml import router as trading_ml_router
from app.routers.rl import router as rl_router
from app.routers.evaluate import router as evaluate_router


# ============================================================
# =================== CREATION APPLICATION ===================
# ============================================================

app = FastAPI(
    title="Système de décision de trading GBP/USD",
    description="""
## Projet Fil Rouge – GBP/USD M1 → M15 → ML → RL

**Auteurs** : Raphaël COLNOT & Clément MÉTOIS  
**Date** : Février 2026

---

### Pipeline complet

| Étape | Endpoint | Description |
|-------|----------|-------------|
| 1 | `POST /dataset/load_m1` | Import CSV M1 brut (2022, 2023 ou 2024) |
| 2 | `POST /m15/aggregate` | Agrégation M1 → M15 (OHLC) |
| 3 | `POST /m15/clean` | Nettoyage M15 (gaps, prix négatifs, OHLC incohérents) |
| 4 | `POST /features/compute` | Feature Engineering V2 (20 features + target) |
| 5 | `GET /eda/full_report/{id}` | Analyse exploratoire complète (ADF, ACF, volatilité...) |
| 6 | `GET /baseline/compare/{id}` | Stratégies de référence (random, B&H, règles) |
| 7 | `POST /trading_ml/train` | ML avec split temporel 2022/2023/2024 |
| 8 | `POST /rl/train` | Q-Learning (walk-forward 2022→2023→2024) |
| 9 | `GET /evaluate/compare/{id}` | Évaluation finale toutes stratégies |

---

**Split temporel strict** : 2022 train / 2023 val / 2024 test (jamais vu)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# ======================= MIDDLEWARE =========================
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ======================= ROUTERS ============================
# ============================================================

# Phase 1-3 : Import / Agrégation / Nettoyage
app.include_router(dataset_router)
app.include_router(m15.router)
app.include_router(clean_router)

# Phase 4 : Feature Engineering V2
app.include_router(features_router)

# Phase 5 : EDA
app.include_router(eda_router)

# Phase 6 : Baselines
app.include_router(baseline_router)

# Phase 7 : ML Trading (split temporel)
app.include_router(trading_ml_router)

# Phase 8 : RL
app.include_router(rl_router)

# Phase 9 : Évaluation finale
app.include_router(evaluate_router)




# ============================================================
# ======================== ENDPOINTS =========================
# ============================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Système de décision de trading GBP/USD",
        "version": "2.0.0",
        "documentation": "/docs",
        "pipeline": [
            "POST /dataset/load_m1       → Charger CSV M1 (2022/2023/2024)",
            "POST /m15/aggregate          → Agréger M1 → M15",
            "POST /m15/clean              → Nettoyer M15",
            "POST /features/compute       → Feature Engineering V2",
            "GET  /eda/full_report/{id}   → Analyse exploratoire",
            "GET  /baseline/compare/{id} → Stratégies de référence",
            "POST /trading_ml/train       → ML (split 2022/2023/2024)",
            "POST /rl/train               → Q-Learning (walk-forward)",
            "GET  /evaluate/compare/{id} → Évaluation finale",
        ],
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "trading-gbpusd-api", "version": "2.0.0"}


# ============================================================
# ====================== LANCEMENT LOCAL =====================
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
