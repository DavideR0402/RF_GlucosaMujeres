# main.py
from fastapi import FastAPI, HTTPException, Security
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict
import os
import logging
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Carga el pipeline SOLO una vez al arrancar
# Es mejor práctica usar una variable de entorno, con un valor por defecto.
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_gradient_boosting_2.joblib")
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pipe = joblib.load(MODEL_PATH)

# Descubre columnas esperadas (num/cat) desde el ColumnTransformer
pre = pipe.named_steps["preprocess"]
num_cols, cat_cols = [], []
for name, transformer, cols in pre.transformers_:
    if name == "num":
        num_cols = list(cols)
    elif name == "cat":
        cat_cols = list(cols)
FEATURE_COLS = num_cols + cat_cols

def compute_imc(peso: Optional[float], talla: Optional[float]) -> Optional[float]:
    if peso is None or talla is None:
        return None
    try:
        t_m = talla/100.0 if talla > 3 else talla  # si parece cm, pasar a metros
        if t_m and t_m > 0 and peso > 0:
            return round(peso/(t_m**2), 2)
    except (TypeError, ZeroDivisionError) as e:
        logger.warning(f"No se pudo calcular IMC con peso={peso}, talla={talla}. Error: {e}")
    return None

def align_row(payload: Dict) -> pd.DataFrame:
    """Alinea dict a FEATURE_COLS + calcula IMC si la feature 'imc' existe."""
    row = {c: np.nan for c in FEATURE_COLS}
    # Copiar valores conocidos
    for k, v in payload.items():
        if k in row:
            row[k] = v
    # Calcular IMC si corresponde
    if "imc" in FEATURE_COLS and ("peso" in payload or "talla" in payload):
        imc_val = compute_imc(payload.get("peso"), payload.get("talla"))
        if imc_val is not None:
            row["imc"] = imc_val
    df = pd.DataFrame([row])
    # Coerción numérica
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

class PredictItem(BaseModel):
    # Define aquí SOLO lo que quieras recibir; otros campos opcionales entran igual
    edad: Optional[float] = Field(None, ge=0)
    tas: Optional[float] = None
    tad: Optional[float] = None
    perimetro_abdominal: Optional[float] = None
    peso: Optional[float] = Field(None, gt=0)
    talla: Optional[float] = Field(None, gt=0)
    realiza_ejercicio: Optional[str] = None
    frecuencia_frutas: Optional[str] = None
    medicamentos_hta: Optional[str] = None
    ips_codigo: Optional[float] = None

    # Valores adicionales no listados explícitamente también se aceptarán (**kwargs)
    class Config:
        extra = "allow"

    @field_validator("talla")
    def talla_valida(cls, v):
        if v is not None and v <= 0:
            raise ValueError("talla debe ser > 0")
        return v

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI(title="API Glucosa RF", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "features_esperadas": len(FEATURE_COLS)}

@app.post("/predict", dependencies=[Security(get_api_key)])
def predict(item: PredictItem, ):
    try:
        df_one = align_row(item.dict())
        y_pred = float(pipe.predict(df_one)[0])
        return {
            "pred_glucosa_mg_dl": round(y_pred, 2),
            "imc_usado": float(df_one["imc"].iloc[0]) if "imc" in df_one.columns else None,
            "campos_faltantes_imputados": [c for c in FEATURE_COLS if pd.isna(df_one[c].iloc[0])]
        }
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
