# main.py
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any
import os
import logging
import joblib
import numpy as np
import pandas as pd
import math

# ---- Config / Logging --------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_gradient_boosting_2.joblib")
API_KEY = os.getenv("API_KEY")

# ---- Globales (se rellenan en startup) --------------------------------------
pipe = None
num_cols: List[str] = []
cat_cols: List[str] = []
FEATURE_COLS: List[str] = []

# ---- Utils ------------------------------------------------------------------
def to_float_or_none(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except Exception:
        return None

def compute_imc(peso: Any, talla: Any) -> Optional[float]:
    """ Calcula IMC tolerante a strings/listas; talla en m o cm. """
    p = to_float_or_none(peso)
    t = to_float_or_none(talla)
    if p is None or t is None:
        return None
    try:
        t_m = t / 100.0 if t > 3 else t  # si parece cm, pasar a metros
        if t_m and t_m > 0 and p > 0:
            return round(p / (t_m ** 2), 2)
    except (TypeError, ZeroDivisionError) as e:
        logger.warning(f"No se pudo calcular IMC con peso={peso}, talla={talla}. Error: {e}")
    return None

def sanitize_value(v: Any):
    """
    Convierte valores del payload a escalares seguros para pandas/sklearn.
    - list -> primer elemento (recursivo)
    - dict -> None (o serializa si quieres)
    - strings 'null'/'none'/'nan'/'' -> None
    """
    if isinstance(v, list):
        return sanitize_value(v[0]) if v else None
    if isinstance(v, dict):
        return None  # si prefieres conservarlo: json.dumps(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"", "null", "none", "nan"}:
            return None
        return s
    return v

def safe_isna(v: Any) -> bool:
    """
    Alternativa segura a pd.isna que evita llamar isnan sobre no-numéricos.
    """
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float, np.floating, np.integer)):
        return isinstance(v, float) and math.isnan(v)
    if isinstance(v, str):
        return False
    return False

def align_row(payload: Dict) -> pd.DataFrame:
    """
    Alinea dict a FEATURE_COLS y calcula IMC si 'imc' existe en las features.
    Sanea valores provenientes de ManyChat (listas/dicts/strings).
    """
    global FEATURE_COLS, num_cols, cat_cols

    # Saneo previo del payload
    clean = {k: sanitize_value(v) for k, v in payload.items()}

    # Si no tenemos FEATURES (modelo sin metadatos), usa las claves del payload
    cols = FEATURE_COLS or list(clean.keys())
    row = {c: np.nan for c in cols}

    # Copiar valores saneados que sí estén en el esquema
    for k, v in clean.items():
        if k in row:
            row[k] = v

    # Calcular IMC si corresponde
    if "imc" in cols and ("peso" in clean or "talla" in clean):
        imc_val = compute_imc(clean.get("peso"), clean.get("talla"))
        if imc_val is not None:
            row["imc"] = imc_val

    df = pd.DataFrame([row])

    # Coerción numérica de columnas numéricas conocidas
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalización de categóricas a string sin romper NaN
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: x if safe_isna(x) else str(x))

    return df

# ---- Seguridad ---------------------------------------------------------------
async def get_api_key(incoming_key: str = Security(api_key_header)):
    if API_KEY and incoming_key == API_KEY:
        return incoming_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

# ---- App ---------------------------------------------------------------------
app = FastAPI(title="API Glucosa RF", version="1.0")

@app.on_event("startup")
def load_artifacts():
    """
    Carga el modelo y resuelve columnas/transformers de forma tolerante.
    """
    from sklearn.compose import ColumnTransformer

    global pipe, num_cols, cat_cols, FEATURE_COLS

    logger.info(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        pipe = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}", exc_info=True)
        raise

    # Log de pasos disponibles
    try:
        ns = list(getattr(pipe, "named_steps", {}).keys())
        logger.info(f"named_steps: {ns}")
    except Exception:
        pass

    # 1) Intentar 'preprocess' o 'preprocessor'
    pre = None
    if hasattr(pipe, "named_steps"):
        pre = pipe.named_steps.get("preprocess") or pipe.named_steps.get("preprocessor")

    # 2) Buscar el primer ColumnTransformer dentro de steps
    if pre is None and hasattr(pipe, "steps"):
        for name, step in pipe.steps:
            if isinstance(step, ColumnTransformer):
                pre = step
                logger.info(f"Usando ColumnTransformer encontrado en step: {name}")
                break

    # 3) Extraer columnas si tenemos ColumnTransformer
    if pre is not None:
        try:
            ncols, ccols = [], []
            for name, transformer, cols in pre.transformers_:
                if name == "num":
                    ncols = list(cols)
                elif name == "cat":
                    ccols = list(cols)
            num_cols = ncols
            cat_cols = ccols
            FEATURE_COLS = ncols + ccols

            try:
                fnames = list(pre.get_feature_names_out())
                logger.info(f"pre.get_feature_names_out(): {len(fnames)} columnas transformadas")
            except Exception:
                pass

            logger.info(f"num_cols={len(num_cols)} cat_cols={len(cat_cols)} FEATURES={len(FEATURE_COLS)}")
        except Exception as e:
            logger.warning(f"No se pudieron derivar columnas desde ColumnTransformer: {e}")

    # 4) Fallback: usar feature_names_in_ del pipeline/modelo
    if not FEATURE_COLS:
        fallback = list(getattr(pipe, "feature_names_in_", []))
        if fallback:
            FEATURE_COLS = fallback
            logger.info(f"Usando feature_names_in_: {len(FEATURE_COLS)} columnas")
        else:
            logger.warning("No se pudieron determinar FEATURE_COLS; se usarán keys del payload en cada request.")

# ---- Schemas -----------------------------------------------------------------
class PredictItem(BaseModel):
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

    class Config:
        extra = "allow"  # en Pydantic v2 puedes usar: model_config = ConfigDict(extra="allow")

    @field_validator("talla")
    def talla_valida(cls, v):
        if v is not None and v <= 0:
            raise ValueError("talla debe ser > 0")
        return v

# ---- Endpoints ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "features_esperadas": len(FEATURE_COLS),
        "tiene_named_steps": bool(getattr(pipe, "named_steps", {})),
    }

@app.post("/predict", dependencies=[Security(get_api_key)])
def predict(item: PredictItem):
    global pipe
    try:
        df_one = align_row(item.model_dump())

        # (opcional) log de tipos para depurar
        # logger.info(df_one.dtypes.to_dict())

        y_pred = float(pipe.predict(df_one)[0])
        return {
            "pred_glucosa_mg_dl": round(y_pred, 2),
            "imc_usado": float(df_one["imc"].iloc[0]) if "imc" in df_one.columns and not safe_isna(df_one["imc"].iloc[0]) else None,
            "campos_faltantes_imputados": [
                c for c in (FEATURE_COLS or df_one.columns.tolist())
                if (c in df_one.columns and safe_isna(df_one[c].iloc[0]))
            ],
        }
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
