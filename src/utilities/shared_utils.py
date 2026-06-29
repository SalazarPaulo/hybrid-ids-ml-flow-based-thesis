# utilities/shared_utils.py
# -*- coding: utf-8 -*-
"""
Funciones compartidas para calibrated.py, automaton.py y scripts de evaluación.

Incluye:
- Wrapper LGBMBoosterWrapper (Booster puro -> interfaz sklearn-like) con alineación.
- Helpers de nombres de features (_get_model_feature_names, union_model_features).
- Alineación de X por modelo (align_X_for_model) para evitar mismatch de nombres.
- Armonizadores de esquema por dataset (NSL-KDD, UNSW15, CICIDS2018).
- Loaders de datos de test.
- Split estratificado train/calib/test con configuración por dataset (vía ENV).
- Predicción unificada (Keras / sklearn / CatBoost / LGBM wrapper).
- Construcción de input para modelos stacking RF (meta-features *_pred).
- Generación de registered_results y guardado a CSV.
- Métricas binarias básicas.
- Quick sample check para debug rápido de CSVs.
"""

from __future__ import annotations

import os
import json
import math
from typing import Any, Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

# ==========================================================
# 0. ALINEACIÓN ( para evitar mismatch )
# ==========================================================

def align_X_for_model(X_df: pd.DataFrame, expected_names, model_name: str = "(model)"):
    """fun cion:
    Devuelve (X_aligned, mode_str).

    MODOS:
      - "exact-order"
      - "subset-reindex"        : expected ⊆ cols -> reindex por nombres
      - "positional-rename"     : ancho coincide pero nombres no -> renombra por posición (SIN ceros)
      - "width-mismatch-fill0"  : ancho distinto -> reindex + fill_value=0 (último recurso)
      - "no-names"              : el modelo no expone nombres -> se devuelve X_df
    """
    if expected_names is None:
        return X_df, "no-names"

    expected = [str(c) for c in list(expected_names)]
    cols = [str(c) for c in list(X_df.columns)]

    if cols == expected:
        return X_df, "exact-order"

    set_expected = set(expected)
    set_cols = set(cols)

    if set_expected.issubset(set_cols):
        X_al = X_df.reindex(columns=expected, fill_value=0.0)
        return X_al, "subset-reindex"

    if X_df.shape[1] == len(expected):
        X_tmp = X_df.copy()
        X_tmp.columns = expected
        return X_tmp, "positional-rename"

    X_al = X_df.reindex(columns=expected, fill_value=0.0)
    return X_al, "width-mismatch-fill0"


# ==========================================================
# 1. LGBM BOOSTER WRAPPER ( Booster -> interfaz )
# ==========================================================

class LGBMBoosterWrapper:
    """ funcion:
    Envoltorio para un Booster de LightGBM.

    - en la alineación si expected no está en cols pero el ancho coincide,
      se usa alineación posicional (no rellena con ceros).
    """

    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names_in_ = np.array([str(x) for x in list(feature_names)], dtype=object)
        self.classes_ = np.array([0, 1], dtype=int)

    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            expected = [str(c) for c in list(self.feature_names_in_)]
            cols = [str(c) for c in list(X.columns)]

            # 1) Reindex SOLO si expected ⊆ cols
            if set(expected).issubset(set(cols)):
                X_al = X.reindex(columns=expected, fill_value=0.0)
                return X_al.to_numpy(dtype=np.float32)

            # 2) Si ancho coincide pero nombres no -> posicional (sin ceros)
            if X.shape[1] == len(expected):
                X_tmp = X.copy()
                X_tmp.columns = expected
                return X_tmp.to_numpy(dtype=np.float32)

            # 3) Mismatch real -> fill0 (último recurso / deseperacion hrmano)
            X_al = X.reindex(columns=expected, fill_value=0.0)
            return X_al.to_numpy(dtype=np.float32)

        return np.asarray(X, dtype=np.float32)

    def predict_proba(self, X):
        X_np = self._to_numpy(X)
        p = np.asarray(self.booster.predict(X_np))

        if p.ndim == 1:
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.vstack([1.0 - p, p]).T

        if p.ndim == 2 and p.shape[1] == 1:
            pp = np.clip(p[:, 0], 1e-7, 1 - 1e-7)
            return np.vstack([1.0 - pp, pp]).T

        return p

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ==========================================================
# 2. FEATURE-NAME HELPERS
# ==========================================================

def _get_model_feature_names(model) -> Optional[list]:
    """ funcion:
    Intenta obtener los nombres de columnas que el modelo espera a su vez que
    soporta el sklearn con feature_names_in_ y el
    CatBoost con feature_names_ o fallback prettified feature importance
    """
    # sklearn / compatibles
    try:
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            return [str(c) for c in list(model.feature_names_in_)]
    except Exception:
        pass

    # CatBoost
    try:
        import catboost  # type: ignore
        if isinstance(model, (catboost.CatBoostClassifier, catboost.CatBoostRegressor)):
            if getattr(model, "feature_names_", None):
                return [str(x) for x in model.feature_names_]
            try:
                fi = model.get_feature_importance(prettified=True)
                if hasattr(fi, "columns"):
                    for col in ("Feature", "FeatureName", "Feature Id", "FeatureId"):
                        if col in fi.columns:
                            return [str(x) for x in fi[col].tolist()]
            except Exception:
                pass
    except Exception:
        pass

    return None


def union_model_features(models: Dict[str, Any]) -> list:
    """
    Devuelve la unión de todas las feature_names conocidas de un dict de modelos.
    """
    feats = set()
    for m in (models or {}).values():
        cols = _get_model_feature_names(m)
        if cols:
            feats.update(str(c) for c in cols)
    return list(feats)


# ==========================================================
# 3. DATA KEY GENERATOR (para registered_results)
# ==========================================================

def make_data_key(row: pd.Series) -> str:
    """
    funcion:
    Formatos:
      - por defecto: JSON normalizado (determinista).
      - compatibilidad: DATA_KEY_FORMAT='repr' -> repr(tuple(...)).
    """
    fmt = os.getenv("DATA_KEY_FORMAT", "json").lower()
    vals = []
    for v in row.tolist():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            vals.append(0.0)
        elif isinstance(v, (float, np.floating)):
            vals.append(round(float(v), 6))
        elif isinstance(v, (np.integer, int)):
            vals.append(int(v))
        else:
            try:
                fv = float(v)
                vals.append(round(fv, 6))
            except Exception:
                vals.append(str(v))

    if fmt == "repr":
        return repr(tuple(vals))
    return json.dumps(vals, separators=(",", ":"), ensure_ascii=False)


# ==========================================================
# 4. ARMONIZADORES POR DATASET
# ==========================================================

# --- NSL-KDD: cargar ref features desde global_paths y renombrar feature_0.. ---
def _load_nsl_ref_cols_from_global() -> Optional[List[str]]:
    try:
        from utilities.global_paths import NSL_REF_FEATURES_JSON  # type: ignore
    except Exception:
        return None
    try:
        if NSL_REF_FEATURES_JSON and os.path.exists(NSL_REF_FEATURES_JSON):
            with open(NSL_REF_FEATURES_JSON, "r", encoding="utf-8") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols and all(isinstance(x, str) for x in cols):
                return cols
    except Exception:
        pass
    return None


def harmonize_nslkdd_schema(
    X_df: pd.DataFrame,
    ref_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    funcion:
    - Si existe una lista canónica y X viene como feature_0..feature_n (mismo ancho) entonces se renombra.
    - se hace coerción numérica, NaN/inf -> 0, float32.
    - si hay ref_features/ref_cols entonces se reindexa a ese orden exacto.
    """
    X = X_df.copy()

    # 1) Renombra solo si columnas tipo feature_0..feature_n y hay referencia canónica
    ref_cols = ref_features or _load_nsl_ref_cols_from_global()
    if ref_cols and X.shape[1] == len(ref_cols):
        cols = [str(c) for c in X.columns]
        if all(c.startswith("feature_") for c in cols):
            X.columns = list(ref_cols)

    # 2) Limpieza numérica
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3) Reordenar a ref_cols si existe (idealmente sin introducir ceros si el schema está bien)
    if ref_cols is not None:
        X = X.reindex(columns=list(ref_cols), fill_value=0.0)

    return X.astype(np.float32, copy=False)


def harmonize_unsw15_schema(
    X_df: pd.DataFrame,
    models_dict: Optional[Dict[str, Any]] = None,
    ref_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    funcion:
    - OHE solo si vienen columnas crudas (proto/service/state) y no hay dummies.
    - numeric, fillna=0, float32.
    - reindex a ref_features si se provee y si no se intenta deducir del primer modelo con nombres.
    """
    X = X_df.copy()

    has_raw = any(c in X.columns for c in ("proto", "service", "state"))
    has_dummies = any(str(c).startswith(("proto_", "service_", "state_")) for c in X.columns)

    if has_raw and not has_dummies:
        for c in ("proto", "service", "state"):
            if c in X.columns and X[c].dtype == object:
                X[c] = X[c].replace("-", "None")
        for col, pref in (("proto", "proto"), ("service", "service"), ("state", "state")):
            if col in X.columns:
                dummies = pd.get_dummies(X[col].astype(str), prefix=pref, dtype=np.float32)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    X = (X.apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .astype(np.float32, copy=False))

    if ref_features and len(ref_features) > 0:
        ref = list(ref_features)
    else:
        ref = None
        if models_dict:
            for m in models_dict.values():
                cols = _get_model_feature_names(m)
                if cols:
                    ref = [str(c) for c in cols]
                    break

    if ref:
        X = X.reindex(columns=ref, fill_value=0.0)

    return X


def _infer_cicids_ref_features_from_models(
    models_dict: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    if not models_dict:
        return None
    for _, m in models_dict.items():
        cols = _get_model_feature_names(m)
        if cols and len(cols) > 0:
            return [str(c) for c in cols]
    return None


def harmonize_cicids2018_schema(
    X_df: pd.DataFrame,
    models_dict: Optional[Dict[str, Any]] = None,
    ref_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    funcion:
    - conservamos nombres del preprocess.
    - definimos orden por: ref_features > inferido de modelos > orden actual.
    """
    X = X_df.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if ref_features and len(ref_features) > 0:
        ref = list(ref_features)
    else:
        ref = _infer_cicids_ref_features_from_models(models_dict) or list(X.columns)

    X = X.reindex(columns=ref, fill_value=0.0)
    return X.astype(np.float32, copy=False)


# ==========================================================
# 5. LOADERS POR DATASET
# ==========================================================

def load_test_data(path: str, dataset_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    funcion:
    Carga CSV y devuelve (X, y):
      - cicids2018: etiqueta 'Threat' (0/1), drop Label/Threat/Attack Type.
      - unsw15: prioriza 'label', luego 'attack'.
      - nslkdd: requiere 'attack'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    d = dataset_type.lower()

    if d == "cicids2018":
        if "Threat" not in df.columns:
            raise ValueError("CICIDS2018 requiere columna 'Threat' en CSV preprocesado.")
        y = df["Threat"].map({0: 0, 1: 1})
        X = df.drop(columns=["Label", "Threat", "Attack Type"], errors="ignore")
        return X.reset_index(drop=True), y.reset_index(drop=True)

    if d == "unsw15":
        if "label" in df.columns:
            y = df["label"]
            X = df.drop(columns=["label"])
        elif "attack" in df.columns:
            y = df["attack"]
            X = df.drop(columns=["attack"])
        else:
            raise ValueError("UNSW15 requiere columna 'label' o 'attack'.")
        return X.reset_index(drop=True), y.reset_index(drop=True)

    if d == "nslkdd":
        if "attack" not in df.columns:
            raise ValueError("NSL-KDD requiere columna 'attack'.")
        y = df["attack"]
        X = df.drop(columns=["attack"])
        return X.reset_index(drop=True), y.reset_index(drop=True)

    raise ValueError(f"dataset_type no reconocido: {dataset_type}")


# ==========================================================
# 6. LABEL COERCION + STRATIFIED SPLIT
# ==========================================================

def _coerce_binary_labels(y_raw) -> pd.Series:
    y = pd.Series(y_raw, copy=True)
    y_num = pd.to_numeric(y, errors="coerce")
    txt = y.astype(str).str.strip().str.lower()
    txt_map = {
        "benign": 0, "normal": 0, "none": 0, "benigno": 0, "ok": 0,
        "malicious": 1, "attack": 1, "anomaly": 1, "maligno": 1, "bad": 1,
    }
    mapped = txt.map(txt_map)
    y_num = y_num.where(~y_num.isna(), mapped)
    return y_num


def make_stratified_splits(
    X_all: pd.DataFrame,
    y_all,
    *,
    train_frac: float = None,
    calib_frac: float = None,
    test_frac: float = None,
    seed: int = None,
):
    """
    funcion:
    Devuelve: X_train, X_calib, X_test, y_train, y_calib, y_test
    """
    from sklearn.model_selection import train_test_split

    def _dataset_hint():
        for k in ("DATASET_HINT", "SPLIT_DATASET", "DATASET_TYPE"):
            v = os.getenv(k, "").strip().lower()
            if v in ("nslkdd", "cicids2018", "unsw15"):
                return v
        return None

    def _defaults_for(ds_hint: str):
        if ds_hint == "nslkdd":
            return dict(train=0.85, calib=0.05, test=0.10, seed=43)
        if ds_hint == "cicids2018":
            return dict(train=0.70, calib=0.10, test=0.20, seed=42)
        if ds_hint == "unsw15":
            return dict(train=0.70, calib=0.15, test=0.15, seed=57)
        return dict(train=0.70, calib=0.15, test=0.15, seed=43)

    y_series = pd.Series(y_all) if not isinstance(y_all, pd.Series) else y_all
    try:
        y_num = _coerce_binary_labels(y_series)
    except Exception:
        y_num = pd.to_numeric(y_series, errors="coerce")

    mask = y_num.notna() & np.isfinite(y_num.astype(float))
    X = X_all.loc[mask].reset_index(drop=True)
    y = y_num.loc[mask].astype(int).reset_index(drop=True)

    if len(y) == 0:
        emptyX = X.iloc[0:0]
        emptyy = y.iloc[0:0]
        return emptyX, emptyX, emptyX, emptyy, emptyy, emptyy

    tf = train_frac if train_frac is not None else (float(os.getenv("SPLIT_TRAIN_FRACTION")) if os.getenv("SPLIT_TRAIN_FRACTION") else None)
    cf = calib_frac if calib_frac is not None else (float(os.getenv("SPLIT_CALIB_FRACTION")) if os.getenv("SPLIT_CALIB_FRACTION") else None)
    ef = test_frac  if test_frac  is not None else (float(os.getenv("SPLIT_TEST_FRACTION"))  if os.getenv("SPLIT_TEST_FRACTION")  else None)
    sd = seed       if seed       is not None else (int(os.getenv("SPLIT_RANDOM_STATE"))     if os.getenv("SPLIT_RANDOM_STATE")     else None)

    ds_hint = _dataset_hint()
    defs = _defaults_for(ds_hint)

    tf = defs["train"] if tf is None else tf
    cf = defs["calib"] if cf is None else cf
    ef = defs["test"]  if ef is None else ef
    sd = defs["seed"]  if sd is None else sd

    disable_calib = (os.getenv("DISABLE_CALIB", "0") == "1") or (os.getenv("SPLIT_USE_CALIB", "1") == "0")
    if disable_calib:
        ef = float(ef) + float(cf)
        cf = 0.0

    tf = max(0.0, min(1.0, float(tf)))
    cf = max(0.0, min(1.0, float(cf)))
    ef = max(0.0, min(1.0, float(ef)))

    total = tf + cf + ef
    if total > 1.0 + 1e-9:
        tf, cf, ef = tf / total, cf / total, ef / total
        print(f"[shared_utils] Aviso: fracciones normalizadas a sumar=1 ({tf:.3f},{cf:.3f},{ef:.3f})")

    # split 1
    if tf <= 0.0:
        X_train = X.iloc[0:0]; y_train = y.iloc[0:0]
        X_tmp = X; y_tmp = y
    elif tf >= 1.0:
        X_train = X; y_train = y
        X_tmp = X.iloc[0:0]; y_tmp = y.iloc[0:0]
    else:
        test_size_lvl1 = 1.0 - tf
        use_strat = np.unique(y.values).size >= 2
        try:
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y,
                test_size=test_size_lvl1,
                stratify=(y if use_strat else None),
                random_state=sd,
                shuffle=True,
            )
        except Exception as e:
            print(f"[shared_utils] Stratify nivel-1 falló ({e}); reintento sin stratify.")
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y,
                test_size=test_size_lvl1,
                stratify=None,
                random_state=sd,
                shuffle=True,
            )

    # split 2
    if len(X_tmp) == 0 or (cf + ef) <= 0.0:
        X_calib = X_tmp.iloc[0:0]; y_calib = y_tmp.iloc[0:0]
        X_test  = X_tmp.iloc[0:0]; y_test  = y_tmp.iloc[0:0]
        return X_train, X_calib, X_test, y_train, y_calib, y_test

    rel_calib = (cf / (cf + ef)) if (cf + ef) > 0 else 0.0

    if rel_calib <= 0.0:
        X_calib = X_tmp.iloc[0:0]; y_calib = y_tmp.iloc[0:0]
        X_test  = X_tmp;          y_test  = y_tmp
        return X_train, X_calib, X_test, y_train, y_calib, y_test

    if rel_calib >= 1.0:
        X_calib = X_tmp;           y_calib = y_tmp
        X_test  = X_tmp.iloc[0:0]; y_test  = y_tmp.iloc[0:0]
        return X_train, X_calib, X_test, y_train, y_calib, y_test

    use_strat2 = np.unique(y_tmp.values).size >= 2
    try:
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=(1.0 - rel_calib),
            stratify=(y_tmp if use_strat2 else None),
            random_state=sd,
            shuffle=True,
        )
    except Exception as e:
        print(f"[shared_utils] Stratify nivel-2 falló ({e}); reintento sin stratify.")
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=(1.0 - rel_calib),
            stratify=None,
            random_state=sd,
            shuffle=True,
        )

    return X_train, X_calib, X_test, y_train, y_calib, y_test


# ==========================================================
# 7. PREDICCIÓN UNIFICADA 
# ==========================================================

def predict_proba_and_pred(
    model,
    X_input: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    funcion:
    Predicción unificada con la alineación por nombres.
    Para stacking usa el build_stacking_X_input().
    """
    # --- Detectar Keras ---
    try:
        import tensorflow as _tf  # type: ignore
        is_keras = isinstance(model, _tf.keras.Model)
    except Exception:
        is_keras = False

    # Asegurar DataFrame base
    X_df = X_input if isinstance(X_input, pd.DataFrame) else pd.DataFrame(X_input)

    if is_keras:
        X_keras = (X_df.apply(pd.to_numeric, errors="coerce")
                     .replace([np.inf, -np.inf], np.nan)
                     .fillna(0.0))
        try:
            input_dim = int(model.input_shape[-1])
        except Exception:
            input_dim = None
        arr = X_keras.to_numpy(dtype=np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if input_dim is not None:
            if arr.shape[1] > input_dim:
                arr = arr[:, :input_dim]
            elif arr.shape[1] < input_dim:
                pad = np.zeros((arr.shape[0], input_dim - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
        probas = np.asarray(model.predict(arr, verbose=0))
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = np.hstack([1.0 - probas, probas])
        preds = (probas[:, 1] >= 0.5).astype(int)
        return preds, probas

    # Limpieza numérica base (antes de alinear)
    X_df = (X_df.apply(pd.to_numeric, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0)
              .astype(np.float32, copy=False))

    names = _get_model_feature_names(model)
    if names:
        X_for_model, mode = align_X_for_model(X_df, names, model_name=model.__class__.__name__)
        if mode == "width-mismatch-fill0":
            print(f"[shared_utils][WARN] {model.__class__.__name__}: width-mismatch -> reindex+fill0 (revisar columnas).")
    else:
        X_for_model = X_df

    # KNN chunking
    cname = model.__class__.__name__.lower()
    is_knn = cname.startswith("kneighbors") or cname.startswith("knn")
    if is_knn and isinstance(X_for_model, pd.DataFrame) and X_for_model.shape[0] > 20000:
        bs = int(os.getenv("STACK_META_BATCH", "20000"))
        outs_pred = []
        outs_proba = [] if hasattr(model, "predict_proba") else None
        for i in range(0, X_for_model.shape[0], bs):
            sl = X_for_model.iloc[i : i + bs]
            outs_pred.append(model.predict(sl))
            if outs_proba is not None:
                outs_proba.append(model.predict_proba(sl))
        preds = np.concatenate(outs_pred, axis=0)
        probas = np.vstack(outs_proba) if outs_proba is not None else None
        return preds.astype(int), probas

    # Predicción normal (DF primero, fallback numpy)
    try:
        preds = model.predict(X_for_model)
    except Exception:
        X_np = X_for_model.to_numpy(dtype=np.float32, copy=False) if isinstance(X_for_model, pd.DataFrame) else np.asarray(X_for_model, dtype=np.float32)
        preds = model.predict(X_np)

    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_for_model)
        except Exception:
            try:
                X_np = X_for_model.to_numpy(dtype=np.float32, copy=False) if isinstance(X_for_model, pd.DataFrame) else np.asarray(X_for_model, dtype=np.float32)
                probas = model.predict_proba(X_np)
            except Exception:
                probas = None
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_for_model)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p1 = np.clip(p1, 1e-7, 1 - 1e-7)
            probas = np.vstack([1.0 - p1, p1]).T
        except Exception:
            probas = None

    preds = np.asarray(preds).reshape(-1)
    if preds.dtype not in (np.int64, np.int32):
        preds = (preds.astype(float) >= 0.5).astype(int)

    return preds.astype(int), (None if probas is None else np.asarray(probas))


# ==========================================================
# 7.1 STACKING HELPERS
# ==========================================================

def _is_stacking_model_name(model_name: str) -> bool:
    n = (model_name or "").lower()
    return ("stack" in n) or ("rf_stacking" in n)

def _find_model_key(models: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in models:
            return c
    return None

def _resolve_base_models_for_stacking(models: Dict[str, Any], dataset_type: str) -> Dict[str, str]:
    """
    funcion:
    Devuelve keys de los modelos base presentes en 'models'.
    Ajusta candidatos por dataset sin romper otros.
    """
    d = (dataset_type or "").lower()

    # candidatos comunes
    dt_c = ["decision_tree_model", "decision_tree"]
    lr_c = ["logistic_regression_model", "log_reg_model", "logistic_regression"]
    svc_c = ["svc_best_clf_model", "svc_model", "svc_best_model", "svc"]
    knn_c = ["knn_model", "knn"]

    # variantes por UNSW por si hay sufijos
    if d == "unsw15":
        dt_c = ["decision_tree_model_unsw", *dt_c]
        lr_c = ["logistic_regression_model_unsw", *lr_c]
        svc_c = ["svc_best_clf_model_unsw", "svc_model_unsw", *svc_c]
        knn_c = ["knn_model_unsw", *knn_c]

    out = {}
    out["dt"]  = _find_model_key(models, dt_c)
    out["lr"]  = _find_model_key(models, lr_c)
    out["svc"] = _find_model_key(models, svc_c)
    out["knn"] = _find_model_key(models, knn_c)

    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise FileNotFoundError(f"[shared_utils][stacking] faltan modelos base: {missing}. Keys disponibles={list(models.keys())[:20]}...")

    return {k: v for k, v in out.items() if v is not None}

def build_stacking_X_input(
    models: Dict[str, Any],
    X_df: pd.DataFrame,
    meta_model: Any,
    dataset_type: str = "nslkdd",
) -> pd.DataFrame:
    """
    funcion:
    Construye X_input para el meta-modelo de stacking:

    - Genera 4 señales base (dt/lr/svc/knn) usando score positivo si existe, si no pred.
    - Crea candidatos:
        * meta_pred_only (4 cols: *_pred)
        * meta_base_only (4 cols: nombres reales de base)
        * meta_both (8 cols: ambos)
        * X_plus_meta_pred / X_plus_meta_base / X_plus_meta_both
    - Si meta_model expone feature_names_in_, reindex exacto a esas columnas.
    - Si no expone nombres pero sí n_features_in_, elige el candidato que calce en ancho.
      (Si ninguno calza, ajusta por trunc/pad como último recurso para evitar feed de ceros “por accidente”.)
    """
    X = X_df if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df)
    X = (X.apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .astype(np.float32, copy=False))

    base = _resolve_base_models_for_stacking(models, dataset_type)

    # 1) predicciones base
    def _score_from_model(mkey: str) -> np.ndarray:
        preds, probas = predict_proba_and_pred(models[mkey], X)
        if probas is not None:
            arr = np.asarray(probas)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1].astype(np.float32)
        return np.asarray(preds).astype(np.float32)

    dt_s  = _score_from_model(base["dt"])
    lr_s  = _score_from_model(base["lr"])
    svc_s = _score_from_model(base["svc"])
    knn_s = _score_from_model(base["knn"])

    # 2) meta_df(s)
    meta_pred_only = pd.DataFrame({
        "decision_tree_pred": dt_s,
        "logistic_regression_pred": lr_s,
        "svc_pred": svc_s,
        "knn_pred": knn_s,
    }, index=X.index).astype(np.float32, copy=False)

    meta_base_only = pd.DataFrame({
        base["dt"]: dt_s,
        base["lr"]: lr_s,
        base["svc"]: svc_s,
        base["knn"]: knn_s,
    }, index=X.index).astype(np.float32, copy=False)

    meta_both = pd.concat([meta_base_only, meta_pred_only], axis=1).astype(np.float32, copy=False)

    X_plus_meta_pred = pd.concat([X, meta_pred_only], axis=1).astype(np.float32, copy=False)
    X_plus_meta_base = pd.concat([X, meta_base_only], axis=1).astype(np.float32, copy=False)
    X_plus_meta_both = pd.concat([X, meta_both], axis=1).astype(np.float32, copy=False)

    # 3) si hay nombres esperados, reindex a eso (mejor caso)
    expected = _get_model_feature_names(meta_model)
    if expected:
        exp_list = [str(c) for c in list(expected)]
        sup_cols = set(X_plus_meta_both.columns)

        # PRINT 1: estado general (se imprime siempre que haya expected)
        print(
            f"[shared_utils][stacking] dataset={dataset_type} meta={meta_model.__class__.__name__} "
            f"expected_n={len(exp_list)} X_n={X.shape[1]} X+pred_n={X_plus_meta_pred.shape[1]} "
            f"sup_n={X_plus_meta_both.shape[1]} expected_head={exp_list[:6]}"
        )
        
        # Caso 1: meta espera solo 4 columnas (stack puro), por lo que si los nombres no están disponibles,
        # se hace una alineación posiional (sin ceros) sobre meta_pred_only.
        if len(exp_list) == 4 and not set(exp_list).issubset(set(X_plus_meta_both.columns)):
            X_tmp = meta_pred_only.copy()
            X_tmp.columns = exp_list
            return X_tmp.astype(np.float32, copy=False)

        # Caso 2: meta espera (X originales + 4 preds) pero los nombres no coinciden.
        # Esto ocurre cuando el meta se entrenó con columnas feature_0..feature_n + *_pred, y en runtime
        # X ya viene con nombres canónicos/armonizados. Si al alinear contra X_plus_meta_both (X+8),
        # puede caer en width-mismatch-fill0. Aquí forzamos el candidato X_plus_meta_pred (X+4) y
        # alineamos posicinalen te (sin ceros).
        if (len(exp_list) == X_plus_meta_pred.shape[1]) and (not set(exp_list).issubset(set(X_plus_meta_pred.columns))):
            print("[shared_utils][stacking] CASE=positional-rename on X_plus_meta_pred (avoid fill0)")
            X_tmp = X_plus_meta_pred.copy()
            X_tmp.columns = exp_list
            return X_tmp.astype(np.float32, copy=False)

        # Caso normal: usar superconjunto (X + base + pred) para cubrir expected
        sup = X_plus_meta_both
        X_al, mode = align_X_for_model(sup, exp_list, model_name="stacking_meta")
        
        # PRINT 3: modo final que ganó 
        print(f"[shared_utils][stacking] CASE=align_X_for_model mode={mode} out_shape={getattr(X_al, 'shape', None)}")
        
        if mode == "width-mismatch-fill0":
            print("[shared_utils][WARN] stacking_meta: width-mismatch even after meta-build (revisar feature_names_in_).")
        return X_al.astype(np.float32, copy=False)

    # 4) si no hay nombres, intentar calzar por ancho
    n_exp = getattr(meta_model, "n_features_in_", None)
    cands = [
        ("meta_pred_only", meta_pred_only),
        ("meta_base_only", meta_base_only),
        ("meta_both", meta_both),
        ("X_plus_meta_pred", X_plus_meta_pred),
        ("X_plus_meta_base", X_plus_meta_base),
        ("X_plus_meta_both", X_plus_meta_both),
    ]

    if isinstance(n_exp, (int, np.integer)) and n_exp > 0:
        for _, dfc in cands:
            if dfc.shape[1] == int(n_exp):
                return dfc.astype(np.float32, copy=False)

        # último recurso: ajustar ancho para que no caiga en “todo ceros por reindex”
        best = meta_pred_only
        if int(n_exp) == X.shape[1] + 4:
            best = X_plus_meta_pred

        dfc = best.copy()
        if dfc.shape[1] > int(n_exp):
            dfc = dfc.iloc[:, : int(n_exp)]
        elif dfc.shape[1] < int(n_exp):
            # pad de columnas dummy
            need = int(n_exp) - dfc.shape[1]
            for j in range(need):
                dfc[f"pad_{j}"] = 0.0
        return dfc.astype(np.float32, copy=False)

    # 5) sin info: volver a la opción más “típica”
    return meta_pred_only.astype(np.float32, copy=False)


# ==========================================================
# 8. REGISTERED_RESULTS Y GUARDADO A CSV
# ==========================================================

def build_registered_results(
    models: Dict[str, Any],
    X_df: pd.DataFrame,
    dataset_type: str = "nslkdd",
    n_threads: int = 4,
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict]]:
    """
    funcion:
    Genera:
      - predictions: dict name -> preds array
      - registered_results: index -> {"__data_key__": str, "<model>": {"prediction": int, "probabilities": ...}}

    stacking-aware:
      - si el nombre del modelo sugiere stacking, construye X_input con build_stacking_X_input
        para evitar que el meta-modelo reciba X crudo y caiga en reindex+fill0.
    """
    X_test_df = (
        (X_df if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df))
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )

    n_samples = X_test_df.shape[0]
    registered_results: Dict[int, Dict[str, Any]] = {
        row_id: {"__data_key__": make_data_key(X_test_df.iloc[row_id])}
        for row_id in range(n_samples)
    }

    predictions: Dict[str, np.ndarray] = {}

    def _worker_predict(name, mdl):
        X_in = X_test_df
        if _is_stacking_model_name(name):
            try:
                X_in = build_stacking_X_input(models, X_test_df, mdl, dataset_type=dataset_type)
            except Exception as e:
                print(f"[shared_utils][WARN] stacking build falló para '{name}': {e}. Se usa X crudo.")
                X_in = X_test_df

        preds, probas = predict_proba_and_pred(mdl, X_in)
        return name, preds, probas

    with ThreadPoolExecutor(max_workers=int(n_threads)) as ex:
        futures = {ex.submit(_worker_predict, name, mdl): name for name, mdl in (models or {}).items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                mname, preds, probas = fut.result()
            except Exception as e:
                print(f"[shared_utils] Warning: modelo {name} falló en predicción: {e}")
                continue

            predictions[mname] = np.asarray(preds)

            for row_id in range(n_samples):
                pred_i = int(preds[row_id])
                if probas is not None:
                    try:
                        p_raw = probas[row_id]
                        if hasattr(p_raw, "tolist"):
                            p_i = p_raw.tolist()
                        elif isinstance(p_raw, (list, tuple)):
                            p_i = list(p_raw)
                        else:
                            p_i = float(p_raw)
                    except Exception:
                        p_i = None
                else:
                    p_i = None
                registered_results[row_id][mname] = {"prediction": pred_i, "probabilities": p_i}

    return predictions, registered_results


def save_registered_results_csv(results: Dict[int, Dict], path: str):
    """
    Guarda registered_results en CSV:
      row_id, data_key, model_name, prediction, probabilities
    """
    import csv
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "data_key", "model_name", "prediction", "probabilities"])
            for row_id, model_dict in results.items():
                data_key_str = model_dict.get("__data_key__", "")
                for mname, info in model_dict.items():
                    if mname == "__data_key__":
                        continue
                    pred = info.get("prediction", 0)
                    probs = info.get("probabilities", "")
                    if isinstance(probs, (list, tuple)):
                        probs_str = json.dumps(list(probs), ensure_ascii=False)
                    else:
                        try:
                            probs_str = json.dumps(probs, ensure_ascii=False)
                        except Exception:
                            probs_str = str(probs)
                    writer.writerow([row_id, data_key_str, mname, pred, probs_str])
    except Exception as e:
        print(f"[shared_utils] Error guardando CSV: {e}")


# ==========================================================
# 9. MÉTRICAS BINARIAS
# ==========================================================

def compute_binary_metrics(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    out: Dict[str, Any] = {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if y_score is not None:
        try:
            arr = np.asarray(y_score)
            if arr.ndim == 2 and arr.shape[1] == 2:
                pos_score = arr[:, 1]
            elif arr.ndim == 1:
                pos_score = arr
            else:
                pos_score = None
            if pos_score is not None:
                out["roc_auc"] = float(roc_auc_score(y_true, pos_score))
        except Exception:
            out["roc_auc"] = None

    return out


# ==========================================================
# 10. QUICK SAMPLE CHECK
# ==========================================================

def quick_sample_check(csv_path: str, nrows: int = 5000) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path, nrows=nrows)
