"""
Application FastAPI - Projet Data Science
Point d'entrée principal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 👇 Import du router dataset
from app.routers.dataset import router as dataset_router


# ============================================================
# =================== CREATION APPLICATION ===================
# ============================================================

app = FastAPI(
    title="Data Science - Projet Final",
    description="""
    Système de décision de trading GBP/USD   

    **Auteur** : Raphaël COLNOT & Clément MÉTOIS  
    **Durée** : 2 jours  
    **Date de début** : 11 février 2026
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# 👇 Ajout du router dataset
app.include_router(dataset_router)


# ============================================================
# ======================== ENDPOINTS =========================
# ============================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Bienvenue sur l'API FastAPI Data Science !",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "fastapi-datascientist-api"}


# ============================================================
# ====================== LANCEMENT LOCAL =====================
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
