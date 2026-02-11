"""
Application FastAPI - Projet Data Science en 5 phases
Point d'entrée principal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Créer l'application FastAPI
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

# Configuration CORS (pour permettre les requêtes depuis un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def root():
    """
    Endpoint racine - Informations sur l'API
    """
    return {
        "message": "Bienvenue sur l'API FastAPI Data Science !",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "fastapi-datascientist-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
