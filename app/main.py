from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel
import numpy as np
import os
from app.classifier import classify_batch
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

# Neue Imports für die Datenbankanbindung
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models import Prediction

# Rate Limiter initialisieren
limiter = Limiter(key_func=get_remote_address)

# Datenbank Setup
db_url = os.getenv("DATABASE_URL").replace("${DB_PASSWORD}", os.getenv("DB_PASSWORD"))
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency: Erstellt eine DB-Session pro Request und schließt sie danach sauber
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ClassifyRequest(BaseModel):
    pixels: list[list[int]]

class ClassifyResponse(BaseModel):
    prediction: str
    confidence: float
    scores: dict[str, float]

app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# API-Key Überprüfung
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("SECRET_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": os.getenv("MODEL_VERSION", "v1.0")}

@app.get("/results")
def results(db: Session = Depends(get_db)):
    # Lade die letzten 10 Vorhersagen chronologisch absteigend
    recent = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(10).all()
    return {
        "results": [
            {
                "id": p.id, 
                "prediction": p.prediction, 
                "confidence": p.confidence, 
                "model_version": p.model_version,
                "created_at": p.created_at
            } for p in recent
        ]
    }

@app.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def classify(request: Request, req: ClassifyRequest, db: Session = Depends(get_db)):
    # 1. Vorhersage generieren
    arr = np.array(req.pixels, dtype=np.uint8)[np.newaxis]
    result = classify_batch(arr)[0]
    
    # 2. In die Datenbank schreiben
    db_prediction = Prediction(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=os.getenv("MODEL_VERSION", "v1.0")
    )
    db.add(db_prediction)
    db.commit()
    
    # 3. Antwort zurückgeben
    return result
