#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrated_predict.py  — TIENE5LOS   SEED OVERRIDES Y ES STACKING NSL/UNSW

Resumen de semillas por defecto:
  - NSL-KDD   -> seed=43
  - CICIDS2018-> seed=42
  - UNSW15    -> seed=57

Overrides por variables de entorno (prioridad sufijada > genérica):
  - EVAL_SEED[_NSLKDD|_CICIDS2018|_UNSW15]
  - SPLIT_TRAIN_FRACTION[_NSLKDD|_CICIDS2018|_UNSW15]
  - SPLIT_CALIB_FRACTION[_NSLKDD|_CICIDS2018|_UNSW15]
  - SPLIT_TEST_FRACTION[_NSLKDD|_CICIDS2018|_UNSW15]
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load, dump

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix  # noqa: F401  (usados a futuro)
)

# =============================================================================
# Paths globales
# =============================================================================
from utilities.global_paths import (
    PROCESSED_NSLKDD_CSV, MODELS_DIR, REGISTERED_RESULTS_PATH_AUTOMATA,
    CALIBRATED_MODELS_DIR_NSLKDD,
    PROCESSED_CICIDS_CSV, MODELS_DIR_CICIDS, REGISTERED_RESULTS_PATH_AUTOMATA_CICIDS,
    CALIBRATED_MODELS_DIR_CICIDS,
    PROCESSED_UNSW_CSV, MODELS_DIR_UNSW, REGISTERED_RESULTS_PATH_AUTOMATA_UNSW,
    CALIBRATED_MODELS_DIR_UNSW15,
    JSONS_DIR
)

# =============================================================================
# Config/constantes
# =============================================================================
KERAS_MODEL_NAME_NSL    = "simple_neural_network_model.keras"
KERAS_MODEL_NAME_UNSW   = "simple_neural_network_model_unsw.keras"
KERAS_MODEL_NAME_CICIDS = "simple_neural_network_model_cicids.keras"

STACKING_MODELS = {
    "nslkdd":     "rf_stacking_model_optimized",
    "cicids2018": "rf_stacking_model_optimized",
    "unsw15":     "rf_stacking_model_optimized_unsw",
}

# Silenciar logs de TF y otros
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Silenciar warnings ruidosos
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but .* was fitted without feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation"
)
warnings.filterwarnings(
    "ignore",
    message=r".*If you are loading a serialized model.*",
    category=UserWarning,
    module=r"xgboost\.core"
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LightGBM.*")

# shared_utils
try:
    from utilities import shared_utils as su
    HAS_SU = True
except Exception:
    su = None
    HAS_SU = False


# =============================================================================
# Utilidad de progreso (aqui el tqdm para ver mejor)
# =============================================================================
def _progress(total, desc, unit="task"):
    """
    es la barra de progreso es opcional, si no hay tqdm, retorna un "no-op".
    """
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit=unit)
    except Exception:
        class _Noop:
            def update(self, n=1): pass
            def close(self): pass
        return _Noop()


# =============================================================================
# [SEED] / [SPLIT-OVR]: helpers de entorno por dataset
# =============================================================================
def _env_float_ds(name_base: str, dataset_type: str, default: float) -> float:
    """fucion:
    Lee FLOAT de ENV con prioridad sufijada por dataset:
      NAME_BASE_{DATASET} > NAME_BASE
    dataset_type se normaliza a mayúsculas (NSLKDD, CICIDS2018, UNSW15).
    """
    ds = dataset_type.upper()
    for key in (f"{name_base}_{ds}", name_base):
        v = os.getenv(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default

def _env_int_ds(name_base: str, dataset_type: str, default: int) -> int:
    """fucion:
    Lee INT de ENV con prioridad sufijada por dataset:
      NAME_BASE_{DATASET} > NAME_BASE
    """
    ds = dataset_type.upper()
    for key in (f"{name_base}_{ds}", name_base):
        v = os.getenv(key)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return default


# =============================================================================
# LightGBM Booster Wrapper
# =============================================================================
class LGBMBoosterWrapper:
    """
    funcion:
    Permite usar un Booster como si tuviera predict/predict_proba de sklearn.
    Procesa el dataset y lo evalua, devuelve el predict.
    """
    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names_in_ = np.array(feature_names)

    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.feature_names_in_, fill_value=0)
            return X.to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def predict_proba(self, X):
        X_np = self._to_numpy(X)
        p = self.booster.predict(X_np)
        if p.ndim == 1:
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.vstack([1 - p, p]).T
        if p.shape[1] == 1:
            pp = np.clip(p[:, 0], 1e-7, 1 - 1e-7)
            return np.vstack([1 - pp, pp]).T
        return p

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def _patch_main_for_lgbm_wrapper():
    """
    fucio:
    Permite que el joblib.load recupere objetos LGBMBoosterWrapper
    guardados cuando la clase estaba en __main__ (es decir se carga el
    wrapper se abre el evoltorio y se analiza).
    """
    try:
        import __main__ as _m
        if not hasattr(_m, "LGBMBoosterWrapper"):
            setattr(_m, "LGBMBoosterWrapper", LGBMBoosterWrapper)
    except Exception:
        pass


def safe_joblib_load(path: str):
    """
    joblib.load con patch para LGBMBoosterWrapper.
    """
    _patch_main_for_lgbm_wrapper()
    return load(path)


# =============================================================================
# Helpers: nombres de features / alineación
# =============================================================================
def _get_model_feature_names(model):
    """fucion:
    Obtiene ua lista de nombres de columnas esperadas por el modelo, si está disponible.
    Usa el shared_utils._get_model_feature_names si existe si no va al fallback.
    """
    if HAS_SU and getattr(su, "_get_model_feature_names", None):
        try:
            return su._get_model_feature_names(model)
        except Exception:
            pass
    try:
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            return [str(c) for c in list(model.feature_names_in_)]
    except Exception:
        pass
    # CatBoost fallback
    try:
        import catboost
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


def _union_model_features(models: dict) -> list:
    """
    funcion:
    Une todas las columnas conocidas por los modelos (si las exponen).
    """
    feats = set()
    for m in (models or {}).values():
        cols = _get_model_feature_names(m)
        if cols:
            feats.update(str(c) for c in cols)
    return list(feats)


def _keras_align_input(model, X_df):
    """funcion:
    Alinea la entrada para modelos Keras (dtype float32 y el shape correcto).
    """
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    X_df = (X_df.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0))
    try:
        input_dim = int(model.input_shape[-1])
    except Exception:
        input_dim = None
    arr = X_df.to_numpy(dtype=np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if input_dim is not None:
        if arr.shape[1] > input_dim:
            arr = arr[:, :input_dim]
        elif arr.shape[1] < input_dim:
            pad = np.zeros((arr.shape[0], input_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
    return arr


# =============================================================================
# Armonización (fallbacks) NSL/UNSW
# =============================================================================
def _load_unsw15_ref_features():
    """
    funcion:
    Intenta cargar lalista de features de referencia para UNSW15 (si existe).
    """
    try:
        ref_path = os.path.join(JSONS_DIR, "res", "preprocessed", "unsw15_feature_list.json") # global path
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return None


def _load_nslkdd_ref_features():
    """
    funcion:
    Lista de referencia para NSL-KDD (si existe), Si no etoces se
    renombra a feature_0..feature_n.
    """
    try:
        ref_path = os.path.join(JSONS_DIR, "res", "preprocessed", "nslkdd_feature_list.json") # el glpbql
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
    except Exception:
        pass
    return None


def harmonize_unsw15_schema_fallback(X_df, models_dict=None):
    """
    fucion:
    este es el fallback de armonización UNSW15:
    - OHE si vienen columnas crudas ('proto','service','state') sin dummies
    - Numérico, sin NaN/inf
    - Reindex a lista de referencia si existe; si no, usa unión de features de modelos
    """
    X = X_df.copy()
    has_raw = any(c in X.columns for c in ("proto", "service", "state"))
    has_dummies = any(c.startswith(("proto_", "service_", "state_")) for c in X.columns)

    if has_raw and not has_dummies:
        for c in ("proto", "service", "state"):
            if c in X.columns and X[c].dtype == object:
                X[c] = X[c].replace("-", "None")
        for col, pref in (("proto", "proto"), ("service", "service"), ("state", "state")):
            if col in X.columns and X[col].dtype == object:
                dummies = pd.get_dummies(X[col].astype(str), prefix=pref, dtype=np.float32)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    X = (X.apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan)
           .fillna(0.0)
           .astype(np.float32, copy=False))

    ref_cols = _load_unsw15_ref_features()
    union = _union_model_features(models_dict or {}) if models_dict else []
    final_cols = list(ref_cols) + [c for c in union if c not in (ref_cols or [])] if ref_cols else (union or list(X.columns))
    return X.reindex(columns=final_cols, fill_value=0.0).copy()


def harmonize_nslkdd_schema_fallback(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    fucion:
    el fallback NSL-KDD:
    - Asegura numérico/float32
    - Reindex a lista de referencia si existe; si no, renombra columnas a feature_i.
    """
    X = (X_df.apply(pd.to_numeric, errors="coerce")
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0))
    ref_cols = _load_nslkdd_ref_features()
    if ref_cols:
        X = X.reindex(columns=ref_cols, fill_value=0.0)
    else:
        # Modo sabio: renombrar a feature_i
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    return X.astype(np.float32, copy=False)


# =============================================================================
# Carga de datos y splits
# =============================================================================
def load_dataset_Xy(data_path: str, dataset_type: str):
    """
    funcion:
    Carga X,y desde CSV con tratamiento por dataset:
    - CICIDS2018: usa 'Threat' y drops específicos y renombra a feature_i.
    - NSL/UNSW: en UNSW primero intenta 'label', luego 'attack'. Si ninguna existe -> ValueError.
    """
    if HAS_SU and getattr(su, "load_test_data", None):
        try:
            return su.load_test_data(data_path, dataset_type)
        except Exception:
            pass

    df = pd.read_csv(data_path)
    d = dataset_type.lower()

    if d == "cicids2018":
        if "Threat" not in df.columns:
            raise ValueError("CICIDS2018 requiere columna 'Threat'.")
        y = df["Threat"].map({0: 0, 1: 1})
        X = df.drop(columns=['Label', 'Threat', 'Attack Type'], errors='ignore')
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        X = (X.apply(pd.to_numeric, errors="coerce")
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0)
               .astype(np.float32, copy=False))
        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    else:
        # NSL/UNSW
        if d == "unsw15":
            if 'label' in df.columns:
                y = df['label']
                X = df.drop(columns=['label'])
            elif 'attack' in df.columns:
                y = df['attack']
                X = df.drop(columns=['attack'])
            else:
                raise ValueError("UNSW15 requiere columna 'label' o 'attack'. No se encontró ninguna.")
        else:
            # nslkdd
            if 'attack' in df.columns:
                y = df['attack']
                X = df.drop(columns=['attack'])
            else:
                raise ValueError("NSL-KDD requiere columna 'attack'.")

        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

        if d != "unsw15":
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        X = (X.apply(pd.to_numeric, errors="coerce")
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0)
               .astype(np.float32, copy=False))

    _u, _c = np.unique(y, return_counts=True)
    print(f"[CHK] y_all clases: {dict(zip(_u.tolist(), _c.tolist()))}  N={len(y)}")

    return X.reset_index(drop=True), y.reset_index(drop=True)


def dataset_default_split_config(dataset_type: str):
    """
    funcion:s
    Configuracio de splits por dataset mas el overrides ENV.
    Defaults:
      - NSL-KDD   : 0.85 / 0.05 / 0.10, seed=43
      - CICIDS2018: 0.70 / 0.10 / 0.20, seed=42
      - UNSW15    : 0.70 / 0.15 / 0.15, seed=57
    """
    d = dataset_type.lower()
    if d == "nslkdd":
        base = dict(train_frac=0.85, calib_frac=0.05, test_frac=0.10, seed=43)
    elif d == "cicids2018":
        base = dict(train_frac=0.70, calib_frac=0.10, test_frac=0.20, seed=42)
    elif d == "unsw15":
        base = dict(train_frac=0.70, calib_frac=0.15, test_frac=0.15, seed=57)
    else:
        base = dict(train_frac=0.70, calib_frac=0.15, test_frac=0.15, seed=43)

    base["train_frac"] = _env_float_ds("SPLIT_TRAIN_FRACTION", dataset_type, base["train_frac"])
    base["calib_frac"] = _env_float_ds("SPLIT_CALIB_FRACTION", dataset_type, base["calib_frac"])
    base["test_frac"]  = _env_float_ds("SPLIT_TEST_FRACTION",  dataset_type, base["test_frac"])
    base["seed"]       = _env_int_ds  ("EVAL_SEED",            dataset_type, base["seed"])
    return base


def make_stratified_splits_dataset(X_all: pd.DataFrame, y_all: pd.Series, dataset_type: str):
    """
    fuccion:
    Hace un split estratificado (o fallback) y armonización post-split por dataset.
    y respeta el overrides por ENV.
    """
    cfg = dataset_default_split_config(dataset_type)
    os.environ["SPLIT_TRAIN_FRACTION"] = str(cfg["train_frac"])
    os.environ["SPLIT_CALIB_FRACTION"] = str(cfg["calib_frac"])
    os.environ["SPLIT_TEST_FRACTION"]  = str(cfg["test_frac"])
    os.environ["SPLIT_RANDOM_STATE"]   = str(cfg["seed"])

    if HAS_SU and getattr(su, "make_stratified_splits", None):
        try:
            X_tr, X_cal, X_te, y_tr, y_cal, y_te = su.make_stratified_splits(X_all, y_all)
        except Exception:
            X_tr = X_cal = X_te = y_tr = y_cal = y_te = None
    else:
        X_tr = X_cal = X_te = y_tr = y_cal = y_te = None

    if X_tr is None:
        y = pd.to_numeric(pd.Series(y_all), errors="coerce").fillna(0).astype(int)
        X = (X_all.apply(pd.to_numeric, errors="coerce")
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0.0))
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y,
            test_size=(1.0 - cfg["train_frac"]),
            random_state=cfg["seed"],
            stratify=y if len(np.unique(y)) >= 2 else None,
            shuffle=True
        )
        caf, tef = cfg["calib_frac"], cfg["test_frac"]
        rel_calib = caf / (caf + tef) if (caf + tef) > 0 else 0.0
        if len(X_tmp) > 0 and (caf + tef) > 0:
            X_cal, X_te, y_cal, y_te = train_test_split(
                X_tmp, y_tmp,
                test_size=(1.0 - rel_calib),
                random_state=cfg["seed"],
                stratify=y_tmp if len(np.unique(y_tmp)) >= 2 else None,
                shuffle=True
            )
        else:
            X_cal = X_tmp.iloc[0:0]; y_cal = y_tmp.iloc[0:0]
            X_te  = X_tmp.iloc[0:0]; y_te  = y_tmp.iloc[0:0]

    d = dataset_type.lower()
    if d == "unsw15":
        if HAS_SU and getattr(su, "harmonize_unsw15_schema", None):
            X_tr = su.harmonize_unsw15_schema(X_tr); X_cal = su.harmonize_unsw15_schema(X_cal); X_te = su.harmonize_unsw15_schema(X_te)
        else:
            X_tr = harmonize_unsw15_schema_fallback(X_tr); X_cal = harmonize_unsw15_schema_fallback(X_cal); X_te = harmonize_unsw15_schema_fallback(X_te)
    elif d == "nslkdd":
        if HAS_SU and getattr(su, "harmonize_nslkdd_schema", None):
            X_tr = su.harmonize_nslkdd_schema(X_tr); X_cal = su.harmonize_nslkdd_schema(X_cal); X_te = su.harmonize_nslkdd_schema(X_te)
        else:
            X_tr = harmonize_nslkdd_schema_fallback(X_tr); X_cal = harmonize_nslkdd_schema_fallback(X_cal); X_te = harmonize_nslkdd_schema_fallback(X_te)
    else:
        def _clean_ci(X):
            return (X.apply(pd.to_numeric, errors="coerce")
                     .replace([np.inf, -np.inf], np.nan)
                     .fillna(0.0)
                     .astype(np.float32, copy=False))
        X_tr, X_cal, X_te = _clean_ci(X_tr), _clean_ci(X_cal), _clean_ci(X_te)

    def _dist(y):
        if y is None or len(y)==0: return "Vacío"
        u, c = np.unique(y, return_counts=True)
        return f"{dict(zip(u.tolist(), c.tolist()))} (N={len(y)})"
    print(f"[CHK] split train: {_dist(y_tr)}")
    print(f"[CHK] split calib: {_dist(y_cal)}")
    print(f"[CHK] split test : {_dist(y_te)}")

    return (X_tr.astype(np.float32), X_cal.astype(np.float32), X_te.astype(np.float32),
            y_tr.astype(int), y_cal.astype(int), y_te.astype(int))


# =============================================================================
# Carga/guardado de modelos
# =============================================================================
def _load_models_for_runtime(models_dir, calibrated_dir, dataset_type, keras_model_name=None):
    """
    funcio:
    Carga de los modelos .pkl y Keras para uso en streaming u orquestación.
    """
    models = {}
    names = set()
    if os.path.isdir(models_dir):
        names |= {f for f in os.listdir(models_dir) if f.endswith(".pkl")}
    if os.path.isdir(calibrated_dir):
        names |= {f for f in os.listdir(calibrated_dir) if f.endswith(".pkl")}

    for f in names:
        path = os.path.join(calibrated_dir, f) if os.path.exists(os.path.join(calibrated_dir, f)) \
               else os.path.join(models_dir, f)
        try:
            models[os.path.splitext(f)[0]] = safe_joblib_load(path)
            print(f"[Runtime] Modelo cargado: {path}")
        except Exception as e:
            print(f"[Runtime] Error cargando {path}: {e}")

    if keras_model_name:
        cand = os.path.join(calibrated_dir, keras_model_name)
        if not os.path.exists(cand):
            cand = os.path.join(models_dir, keras_model_name)
        if os.path.exists(cand):
            try:
                km = tf.keras.models.load_model(cand)
                d = dataset_type.lower()
                key = "simple_neural_network_model_unsw" if d=="unsw15" else ("simple_neural_network_model_cicids" if d=="cicids2018" else "simple_neural_network_model")
                models[key] = km
                print(f"[Runtime] Keras cargado: {cand}")
            except Exception as e:
                print(f"[Runtime] Error cargando Keras: {e}")
    return models


def _mk_calibrator(model, cv, method='sigmoid', prefit=False):
    """
    fucio:
    Envuelve CalibratedClassifierCV.
    """
    try:
        return CalibratedClassifierCV(estimator=model, cv=('prefit' if prefit else cv), method=method)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=model, cv=('prefit' if prefit else cv), method=method)


def _clean_xy_pair(Xin, yin):
    """
    fucion:
    Limpia y alinea X e y:
    - y va a ser numérico, se filtra NaN e inf y se castea a int
    - X aplica la misma máscara; y soporta DataFrame o ndarray
    Devuelve (X_clean, y_clean), si tras limpiar queda <2 clases, retorna (None, None).
    """
    y_s = pd.to_numeric(pd.Series(yin), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = y_s.notna()
    if isinstance(Xin, pd.DataFrame):
        X_clean = Xin.loc[mask].copy()
    else:
        Xin = np.asarray(Xin)
        X_clean = Xin[mask.values]
    y_clean = y_s.loc[mask].astype(int)

    if y_clean.nunique() < 2 or len(y_clean) == 0:
        return None, None
    return X_clean, y_clean


def calibrate_single_model(model_file, models_dir, calibrated_dir,
                           X_calib, y_calib, dataset_type,
                           X_train=None, y_train=None):
    """
    fucion:
    Carga un modelo base y lo calibra si es posible.
    - Si el modelo ya tiene predict_proba, se deja tal cual (calibración opcional).
    - Si no hay 2 clases tras limpieza o no hay datos, se devuelve modelo base.
    """
    try:
        model_path = os.path.join(models_dir, model_file)
        model = safe_joblib_load(model_path)
        print(f"[Info] Modelo cargado: {model_file} (dataset={dataset_type})")

        def _dist(y):
            if y is None or len(y)==0: return "Vacío"
            u, c = np.unique(y, return_counts=True)
            return f"{dict(zip(u.tolist(), c.tolist()))} (N={len(y)})"
        print(f"[CHK] {model_file} -> y_calib={_dist(y_calib)} | y_train={_dist(y_train)}")

        if hasattr(model, "predict_proba"):
            return (os.path.splitext(model_file)[0], model)

        y_fit = y_calib if (y_calib is not None and len(y_calib) > 0) else y_train
        if y_fit is None or np.unique(pd.Series(y_fit)).size < 2:
            print("[Info] Saltando calibración (solo 1 clase disponible). Se usa modelo base SIN calibrar.")
            return (os.path.splitext(model_file)[0], model)

        os.makedirs(calibrated_dir, exist_ok=True)

        if y_calib is not None and len(y_calib) > 0:
            print("[Info] Calibración con CALIB (cv='prefit').")
            cal = _mk_calibrator(model, cv='prefit', method='sigmoid', prefit=True)
            names = _get_model_feature_names(model)
            if names:
                Xc = X_calib.reindex(columns=names, fill_value=0.0)
            else:
                Xc = X_calib.to_numpy(dtype=np.float32, copy=False)

            Xc_clean, yc_clean = _clean_xy_pair(Xc, y_calib)
            if Xc_clean is None:
                print("[Info] Saltando calibración (y_calib tras limpieza < 2 clases). Modelo base.")
                return (os.path.splitext(model_file)[0], model)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The `cv='prefit'` option is deprecated",
                    category=UserWarning
                )
                cal.fit(Xc_clean, yc_clean.to_numpy())

        else:
            print("[Info] Calibración con CV=3 sobre TRAIN.")
            cal = _mk_calibrator(model, cv=3, method='sigmoid', prefit=False)
            names = _get_model_feature_names(model)
            if names:
                Xt = X_train.reindex(columns=names, fill_value=0.0)
            else:
                Xt = X_train.to_numpy(dtype=np.float32, copy=False)

            Xt_clean, yt_clean = _clean_xy_pair(Xt, y_train)
            if Xt_clean is None:
                print("[Info] Saltando calibración (y_train tras limpieza < 2 clases). Modelo base.")
                return (os.path.splitext(model_file)[0], model)

            cal.fit(Xt_clean, yt_clean.to_numpy())

        outp = os.path.join(calibrated_dir, model_file)
        dump(cal, outp)
        print(f"[Info] Modelo calibrado guardado en: {outp}")
        return (os.path.splitext(model_file)[0], cal)

    except Exception as e:
        print(f"[Error] calibrate_single_model({model_file}): {e}")
        return None


def calibrate_and_save_models(models_dir, calibrated_dir,
                              X_train, y_train, X_calib, y_calib,
                              keras_model_name, dataset_type):
    """
    fucio:
    Calibra en paralelo todos los .pkl de models_dir y carga Keras si existe.
    Devuelve un dict de {nombre_modelo: estimator}.
    """
    t0 = time.time()
    if not os.path.exists(models_dir):
        print(f"[Error] No existe {models_dir}")
        return {}
    os.makedirs(calibrated_dir, exist_ok=True)

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    models = {}

    with ThreadPoolExecutor(max_workers=int(os.getenv("CALIB_WORKERS", "4"))) as ex:
        futs = {ex.submit(calibrate_single_model, f, models_dir, calibrated_dir,
                          X_calib, y_calib, dataset_type, X_train, y_train): f
                for f in model_files}
        pbar = _progress(len(futs), "Calibrando modelos", unit="mdl")
        for fut in as_completed(futs):
            res = fut.result()
            pbar.update(1)
            if res: models[res[0]] = res[1]
        pbar.close()

    for f in os.listdir(calibrated_dir):
        if f.endswith(".pkl"):
            key = os.path.splitext(f)[0]
            if key not in models:
                try:
                    models[key] = safe_joblib_load(os.path.join(calibrated_dir, f))
                except Exception:
                    pass

    chosen = os.path.join(calibrated_dir, keras_model_name)
    if not os.path.exists(chosen):
        chosen = os.path.join(models_dir, keras_model_name)
    if os.path.exists(chosen):
        try:
            km = tf.keras.models.load_model(chosen)
            d = dataset_type.lower()
            if d == "unsw15":       models["simple_neural_network_model_unsw"] = km
            elif d == "cicids2018": models["simple_neural_network_model_cicids"] = km
            else:                   models["simple_neural_network_model"] = km
            print(f"[Info] Modelo Keras cargado: {chosen}")
        except Exception as e:
            print(f"[Warn] No se pudo cargar Keras: {e}")
    else:
        print(f"[Info] No se encontró modelo Keras en {chosen}")

    print(f"[Info] Calibración total (s): {time.time() - t0:.2f}")
    return models


# =============================================================================
# UNSW15 stacking (fallback completo) + NSL-KDD stacking
# =============================================================================
def _predict_in_chunks(model, X_df, batch_size=20000, show_progress=False, desc="KNN meta"):
    """
    funcio:
    Predice en batches para evitar picos de memoria (usado para KNN en metas demora mucho).
    """
    batch_size = int(os.getenv("STACK_META_BATCH", str(batch_size)))
    n = X_df.shape[0]
    out = []
    steps = (n + batch_size - 1) // batch_size
    pbar = _progress(steps, desc, unit="batch") if show_progress else None
    for i in range(0, n, batch_size):
        sl = X_df.iloc[i:i+batch_size]
        out.append(model.predict(sl))
        if pbar: pbar.update(1)
    if pbar: pbar.close()
    return np.concatenate(out, axis=0)


def _unsw15_prepare_stacking_matrix(
    X_df_in: pd.DataFrame,
    models: dict,
    stacking_model,
    max_knn_meta_rows=50000,
    show_progress=True
):
    """
    funcion:
    Construye la matriz meta esperada por el stacking de UNSW15:
    columnas meta -> predicciones de modelos base (DT/LR/SVC/KNN).
    Si no están presentes o no hay feature_names_in_ en el stacking, retorna X_df_in.
    """
    if not hasattr(stacking_model, "feature_names_in_") or stacking_model.feature_names_in_ is None:
        return X_df_in

    feat_names = list(stacking_model.feature_names_in_)
    X_stack = X_df_in.reindex(columns=feat_names, fill_value=0.0).copy()

    meta_to_base = {
        "decision_tree_pred":       "decision_tree_model_unsw",
        "logistic_regression_pred": "logistic_regression_model_unsw",
        "svc_pred":                 "svc_model_unsw",
        "knn_pred":                 "knn_model_unsw",
    }

    metas_presentes = [(mcol, models[mkey])
                       for mcol, mkey in meta_to_base.items()
                       if mcol in feat_names and mkey in models]

    pbar = _progress(len(metas_presentes), "Metas stacking UNSW15", unit="meta") if show_progress else None

    for meta_col, base_model in metas_presentes:
        try:
            is_knn = base_model.__class__.__name__.lower().startswith("knn")
            if (meta_col == "knn_pred" or is_knn):
                if os.getenv("SKIP_META_KNN", "0") == "1":
                    if pbar: pbar.update(1)
                    continue
                if X_df_in.shape[0] > max_knn_meta_rows:
                    print(f"[Info][UNSW15] Saltando meta '{meta_col}' (KNN) por N={X_df_in.shape[0]} > {max_knn_meta_rows}.")
                    if pbar: pbar.update(1)
                    continue

            base_names = _get_model_feature_names(base_model)
            if base_names:
                X_for_base = X_df_in.reindex(columns=base_names, fill_value=0.0)
            else:
                X_for_base = X_df_in.to_numpy(dtype=np.float32, copy=False)

            if meta_col == "knn_pred" or is_knn:
                preds_base = _predict_in_chunks(base_model, X_for_base, batch_size=20000, show_progress=show_progress, desc="Meta KNN")
            else:
                preds_base = base_model.predict(X_for_base)

            X_stack[meta_col] = np.asarray(preds_base, dtype=np.int32)

        except KeyboardInterrupt:
            print(f"[Warn][UNSW15] Meta '{meta_col}' cancelada por usuario.")
            raise
        except Exception as e:
            print(f"[Warn][UNSW15] No se pudo calcular meta '{meta_col}': {e}")
        finally:
            if pbar: pbar.update(1)

    if pbar: pbar.close()
    return X_stack


def _nslkdd_prepare_stacking_matrix(
    X_df_in: pd.DataFrame,
    models: dict,
):
    """
    fucion:
    [STACK-NSL]
    Construye la matriz meta esperada por el stacking de NSL-KDD:
    columnas meta -> predicciones de modelos base (DT/LR/SVC/KNN).

    igualito a _build_stacking_meta_nslkdd del autómata:
      - decision_tree_model       -> decision_tree_pred
      - logistic_regression_model -> logistic_regression_pred
      - svc_best_clf_model        -> svc_pred
      - knn_model                 -> knn_pred
    """
    base = {
        "decision_tree_model":       "decision_tree_pred",
        "logistic_regression_model": "logistic_regression_pred",
        "svc_best_clf_model":        "svc_pred",
        "knn_model":                 "knn_pred",
    }

    for mname in base.keys():
        if mname not in models:
            print(f"[NSL-KDD stacking] Falta modelo base '{mname}', no se aplica stacking.")
            return None

    stack_df = X_df_in.copy()
    for base_name, meta_col in base.items():
        mdl = models[base_name]
        try:
            preds, _ = predict_single_model(base_name, mdl, X_df_in, "nslkdd")
        except Exception as e:
            print(f"[NSL-KDD stacking] Error al predecir con '{base_name}': {e}")
            preds = np.zeros(X_df_in.shape[0], dtype=np.int32)
        stack_df[meta_col] = np.asarray(preds, dtype=np.int32)

    return stack_df


# =============================================================================
# Predicción por lote
# =============================================================================
def predict_single_model(model_name, model, X_test, dataset_type):
    """
    funcion:
    Predice con un modelo individual y devuelve (preds, probas o None).
    - Primero intenta usar shared_utils.predict_proba_and_pred.
    - Si falla o shared_utils no está disponible, cae al fallback (Keras, sklearn, wrappers, etc).

    es para garantizar que LightGBM, CatBoost, el stacking RF, KNN, etc. se
    comporten igual en calibrated_predict.py y en sub_automaton.py.
    """
    # === CAMINO PRINCIPAL: usar el mismo motor que el autómata ===
    if HAS_SU and getattr(su, "predict_proba_and_pred", None):
        try:
            preds, probas = su.predict_proba_and_pred(model, X_test)
            return preds, probas
        except Exception as e:
            print(f"[Warn] predict_proba_and_pred falló para {model_name}: {e}. "
                  f"Usando fallback local en predict_single_model.")

    # === FALLBACK LOCAL ===
    try:
        print(f"[Info] => predict_single_model: {model_name}, dataset_type={dataset_type}")
        d = dataset_type.lower()

        # --- KERAS ---
        if model_name in ["simple_neural_network_model",
                          "simple_neural_network_model_unsw",
                          "simple_neural_network_model_cicids"]:
            print("[Info] => modelo Keras, X->float32.")
            X_np = _keras_align_input(model, X_test)
            probas = model.predict(X_np, verbose=0)
            if probas.ndim == 2 and probas.shape[1] == 1:
                probas = np.hstack((1 - probas, probas))
            preds = (probas[:, 1] >= 0.5).astype(int)
            return preds, probas

        # --- SKLEARN / OTROS ---
        # 1) Si el modelo expone feature_names_in_, pasamos DataFrame alineado (no numpy).
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            names = [str(c) for c in list(model.feature_names_in_)]
            if isinstance(X_test, pd.DataFrame):
                X_input_df = X_test.reindex(columns=names, fill_value=0.0).copy()
            else:
                try:
                    X_input_df = pd.DataFrame(X_test, columns=names)
                except Exception:
                    X_input_df = pd.DataFrame(X_test)
            X_input_df = (X_input_df.apply(pd.to_numeric, errors='coerce')
                                       .replace([np.inf, -np.inf], np.nan)
                                       .fillna(0.0)
                                       .astype(np.float32))

            # KNN: trocear si es grande
            is_knn = model.__class__.__name__.lower().startswith("kneighbors")
            if is_knn and X_input_df.shape[0] > 20000:
                preds = _predict_in_chunks(model, X_input_df, batch_size=20000,
                                           show_progress=False, desc="KNN base")
                probas = None
                if hasattr(model, "predict_proba"):
                    bs = int(os.getenv("STACK_META_BATCH", "20000"))
                    outs = []
                    for i in range(0, X_input_df.shape[0], bs):
                        outs.append(model.predict_proba(X_input_df.iloc[i:i+bs]))
                    probas = np.vstack(outs)
                return preds, probas

            preds = model.predict(X_input_df)
            probas = model.predict_proba(X_input_df) if hasattr(model, "predict_proba") else None
            if probas is None and hasattr(model, "decision_function"):
                scores = model.decision_function(X_input_df)
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                p1 = np.clip(p1, 1e-7, 1 - 1e-7)
                probas = np.vstack([1 - p1, p1]).T
            return preds, probas

        # 2) Si no hay feature_names_in_ pero podemos inferir nombres, también DataFrame.
        names = _get_model_feature_names(model)
        if names:
            if not isinstance(X_test, pd.DataFrame):
                try:
                    if hasattr(X_test, "ndim") and getattr(X_test, "ndim") == 2 and X_test.shape[1] == len(names):
                        X_df = pd.DataFrame(X_test, columns=names)
                    else:
                        X_df = pd.DataFrame(X_test)
                except Exception:
                    X_df = pd.DataFrame(X_test)
            else:
                X_df = X_test
            X_input_df = (X_df.reindex(columns=names, fill_value=0.0)
                            .apply(pd.to_numeric, errors='coerce')
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                            .astype(np.float32))

            # KNN: trocear si es grande
            is_knn = model.__class__.__name__.lower().startswith("kneighbors")
            if is_knn and X_input_df.shape[0] > 20000:
                preds = _predict_in_chunks(model, X_input_df, batch_size=20000,
                                           show_progress=False, desc="KNN base")
                probas = None
                if hasattr(model, "predict_proba"):
                    bs = int(os.getenv("STACK_META_BATCH", "20000"))
                    outs = []
                    for i in range(0, X_input_df.shape[0], bs):
                        outs.append(model.predict_proba(X_input_df.iloc[i:i+bs]))
                    probas = np.vstack(outs)
                return preds, probas

            preds = model.predict(X_input_df)
            probas = model.predict_proba(X_input_df) if hasattr(model, "predict_proba") else None
            if probas is None and hasattr(model, "decision_function"):
                scores = model.decision_function(X_input_df)
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                p1 = np.clip(p1, 1e-7, 1 - 1e-7)
                probas = np.vstack([1 - p1, p1]).T
            return preds, probas

        # 3) Último recurso: ndarray (no hay nombres disponibles)
        X_input_np = (X_test.to_numpy(dtype=np.float32)
                      if isinstance(X_test, pd.DataFrame)
                      else np.asarray(X_test, dtype=np.float32))
        preds = model.predict(X_input_np)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_input_np)
        else:
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_input_np)
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                p1 = np.clip(p1, 1e-7, 1 - 1e-7)
                probas = np.vstack([1 - p1, p1]).T
            else:
                probas = None
        return preds, probas

    except Exception as e:
        print(f"[Error] => predict_single_model, {model_name}: {e}")
        return np.zeros(X_test.shape[0], dtype=int), None

def generate_model_predictions(models: dict, X_test: pd.DataFrame, dataset_type="nslkdd"):
    """
    funcion:
    Predice en paralelo con todos los modelos y devuelve (preds_all, registered_results).
    Aplica tratamiento especial a los modelos de stacking (NSL-KDD, UNSW15).
    """
    preds_all = {}
    registered = {i: {"__data_key__": repr(tuple(X_test.iloc[i].tolist()))} for i in range(X_test.shape[0])}

    def _worker(name, mdl):
        d = dataset_type.lower()
        # [STACK-UNSW] - usar matriz meta UNSW15
        if d == "unsw15" and name == STACKING_MODELS.get("unsw15"):
            try:
                X_stack = _unsw15_prepare_stacking_matrix(X_test, models, mdl, show_progress=False)
            except Exception as e:
                print(f"[UNSW15 stacking] Error preparando matriz meta: {e}")
                X_stack = None
            X_input = X_stack if X_stack is not None else X_test
            return predict_single_model(name, mdl, X_input, dataset_type)

        # [STACK-NSL] - usar matriz meta NSL-KDD
        if d == "nslkdd" and name == STACKING_MODELS.get("nslkdd"):
            try:
                X_stack = _nslkdd_prepare_stacking_matrix(X_test, models)
            except Exception as e:
                print(f"[NSL-KDD stacking] Error preparando matriz meta: {e}")
                X_stack = None
            X_input = X_stack if X_stack is not None else X_test
            return predict_single_model(name, mdl, X_input, dataset_type)

        # Modelos normales
        return predict_single_model(name, mdl, X_test, dataset_type)

    with ThreadPoolExecutor(max_workers=int(os.getenv("PRED_WORKERS", "4"))) as ex:
        futs = {ex.submit(_worker, name, mdl): name for name, mdl in models.items()}
        pbar = _progress(len(futs), "Predicción", unit="mdl")
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                preds, probas = fut.result()
            except Exception as e:
                print(f"[Warn] {name} lanzó excepción: {e}")
                pbar.update(1)
                continue
            preds_all[name] = preds
            for i in range(X_test.shape[0]):
                pred_i = int(preds[i])
                if probas is not None:
                    try:
                        p_i = probas[i].tolist() if hasattr(probas, "tolist") else (list(probas[i]) if isinstance(probas[i], (list, tuple)) else float(probas[i]))
                    except Exception:
                        p_i = None
                else:
                    p_i = None
                registered[i][name] = {"prediction": pred_i, "probabilities": p_i}
            pbar.update(1)
        pbar.close()

    return preds_all, registered


def save_registered_results_csv(results, path):
    """
    funcion:
    Guarda el diccionario 'registered_results' en CSV.
    """
    if HAS_SU and getattr(su, "save_registered_results_csv", None):
        try:
            su.save_registered_results_csv(results, path); return
        except Exception:
            pass
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["row_id", "data_key", "model_name", "prediction", "probabilities"])
            for row_id, model_dict in results.items():
                dk = model_dict.get("__data_key__", "")
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
                    w.writerow([row_id, dk, mname, pred, probs_str])
        print(f"[Info] CSV guardado en: {path}")
    except Exception as e:
        print(f"[Error] save_registered_results_csv: {e}")


# =============================================================================
# Procesos principales (por dataset)
# =============================================================================
def process_dataset_calibration(data_path, models_dir, calibrated_models_dir,
                                registered_results_path, keras_model_name,
                                dataset_type="nslkdd"):
    """
    funcio:
    el modo split/eval: split -> calibrar -> predecir TEST -> guardar CSV.
    """
    print(f"[Calibrated] dataset={dataset_type}")
    if not os.path.exists(data_path):
        print(f"[Calibrated] No existe CSV: {data_path}")
        return

    t0 = time.time()
    X_all, y_all = load_dataset_Xy(data_path, dataset_type)
    print(f"[Info] Dimensiones: {X_all.shape} (X), {y_all.shape} (y)")

    X_train, X_calib, X_test, y_train, y_calib, y_test = make_stratified_splits_dataset(X_all, y_all, dataset_type)
    print(f"[Calibrated] split -> train:{len(X_train)} calib:{len(X_calib)} test:{len(X_test)}")
    print(f"[Phase] Carga+Split ({dataset_type}) t={time.time() - t0:.2f}s")

    t1 = time.time()
    models = calibrate_and_save_models(models_dir, calibrated_models_dir,
                                       X_train, y_train, X_calib, y_calib,
                                       keras_model_name, dataset_type)
    print(f"[Phase] Calibración ({dataset_type}) t={time.time() - t1:.2f}s")

    t2 = time.time()
    preds, reg = generate_model_predictions(models, X_test, dataset_type=dataset_type)
    print(f"[Phase] Predicción TEST ({dataset_type}) t={time.time() - t2:.2f}s")

    save_registered_results_csv(reg, registered_results_path)
    print(f"[Calibrated] OK ({dataset_type})  total={time.time() - t0:.2f}s")


def process_dataset_no_split_calibration(data_path, models_dir, calibrated_models_dir,
                                         registered_results_path, keras_model_name,
                                         dataset_type="nslkdd"):
    """
    funcion:
    este modo no-split: usa todo el CSV para entreno/calibración base y para predecir (registro).
    """
    print(f"[Calibrated][NoSplit] => dataset={dataset_type}")
    if not os.path.exists(data_path):
        print(f"[Calibrated][NoSplit] No existe CSV: {data_path}")
        return

    t0 = time.time()
    X_all, y_all = load_dataset_Xy(data_path, dataset_type)
    print(f"[Info][NoSplit] Dimensiones: {X_all.shape} (X), {y_all.shape} (y)")

    X_train = X_all; y_train = y_all
    X_calib = pd.DataFrame(columns=X_all.columns); y_calib = pd.Series(dtype=int)

    t1 = time.time()
    models = calibrate_and_save_models(models_dir, calibrated_models_dir,
                                       X_train, y_train, X_calib, y_calib,
                                       keras_model_name, dataset_type)
    print(f"[Phase][NoSplit] Calibración ({dataset_type}) t={time.time() - t1:.2f}s")

    t2 = time.time()
    preds, reg = generate_model_predictions(models, X_all, dataset_type=dataset_type)
    print(f"[Phase][NoSplit] Predicción ({dataset_type}) t={time.time() - t2:.2f}s")

    save_registered_results_csv(reg, registered_results_path)
    print(f"[Calibrated][NoSplit] OK ({dataset_type})  total={time.time() - t0:.2f}s")


# =============================================================================
# main (CLI)
# =============================================================================
def main():
    """
    funcion:
    es el punto de entrada del script. Respeta flags y variables de entorno:
    - RUNTIME_MODE=<offline_split|offline_nosplit|online_live|offline_eval>
    - CALIB_WORKERS, PRED_WORKERS, STACK_META_BATCH, etc.
    """
    from textwrap import dedent

    ENV_HELP = dedent("""
    Variables de entorno para este:

      RUNTIME_MODE=<offline_split|offline_nosplit|online_live|offline_eval>

      # Splits (usados por offline_split/offline_eval)
      # Genéricas:
      SPLIT_TRAIN_FRACTION=0.6
      SPLIT_CALIB_FRACTION=0.2
      SPLIT_TEST_FRACTION=0.2
      EVAL_SEED=43

      # Específicas por dataset (prioridad sobre las genéricas):
      SPLIT_TRAIN_FRACTION_NSLKDD=0.85
      SPLIT_CALIB_FRACTION_NSLKDD=0.05
      SPLIT_TEST_FRACTION_NSLKDD=0.10
      EVAL_SEED_NSLKDD=43

      SPLIT_TRAIN_FRACTION_CICIDS2018=0.70
      SPLIT_CALIB_FRACTION_CICIDS2018=0.10
      SPLIT_TEST_FRACTION_CICIDS2018=0.20
      EVAL_SEED_CICIDS2018=42

      SPLIT_TRAIN_FRACTION_UNSW15=0.70
      SPLIT_CALIB_FRACTION_UNSW15=0.15
      SPLIT_TEST_FRACTION_UNSW15=0.15
      EVAL_SEED_UNSW15=57

      # Concurrencia / performance
      CALIB_WORKERS=4
      PRED_WORKERS=4
      STACK_META_BATCH=20000
      SKIP_OHE_UNSW=0
      SKIP_META_KNN=0
      DATA_KEY_FORMAT=json
    """).strip()

    parser = argparse.ArgumentParser(
        description="Calibrated runner",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=ENV_HELP
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["offline_split", "offline_eval", "offline_nosplit", "online_live"],
        default=None,
        help="Modo de ejecución (si no se pasa, se usa RUNTIME_MODE o 'offline_eval' por defecto)."
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Ignorar GUI (opcional, para que no habra mas ventanas; sin efecto en esta CLI)."
    )
    args, _unknown = parser.parse_known_args()

    mode = (args.mode or os.getenv("RUNTIME_MODE", "offline_eval")).lower()
    print(f"[Calibrated] mode='{mode}'")

    if mode == "online_live":
        print("[Calibrated] ONLINE: uso de StreamingScorer desde app.")
        return

    def run_split(data_path, models_dir, cal_dir, out_csv, keras_name, dtype):
        if not os.path.exists(data_path):
            print(f"[Info] {dtype} omitido: no existe CSV -> {data_path}")
            return
        process_dataset_calibration(
            data_path, models_dir, cal_dir, out_csv, keras_name, dtype
        )

    def run_nosplit(data_path, models_dir, cal_dir, out_csv, keras_name, dtype):
        if not os.path.exists(data_path):
            print(f"[Info] {dtype} omitido: no existe CSV -> {data_path}")
            return
        process_dataset_no_split_calibration(
            data_path, models_dir, cal_dir, out_csv, keras_name, dtype
        )

    if mode in ("offline_split", "offline_eval"):
        print("[Calibrated] => OFFLINE con split (train/calib/test)")
        run_split(PROCESSED_NSLKDD_CSV, MODELS_DIR, CALIBRATED_MODELS_DIR_NSLKDD,
                  REGISTERED_RESULTS_PATH_AUTOMATA, KERAS_MODEL_NAME_NSL, "nslkdd")
        run_split(PROCESSED_CICIDS_CSV, MODELS_DIR_CICIDS, CALIBRATED_MODELS_DIR_CICIDS,
                  REGISTERED_RESULTS_PATH_AUTOMATA_CICIDS, KERAS_MODEL_NAME_CICIDS, "cicids2018")
        run_split(PROCESSED_UNSW_CSV, MODELS_DIR_UNSW, CALIBRATED_MODELS_DIR_UNSW15,
                  REGISTERED_RESULTS_PATH_AUTOMATA_UNSW, KERAS_MODEL_NAME_UNSW, "unsw15")
        print("[Calibrated] => Proceso OFFLINE FIN (split)")
        return

    if mode == "offline_nosplit":
        print("[Calibrated] => OFFLINE sin split (todo el CSV)")
        run_nosplit(PROCESSED_NSLKDD_CSV, MODELS_DIR, CALIBRATED_MODELS_DIR_NSLKDD,
                    REGISTERED_RESULTS_PATH_AUTOMATA, KERAS_MODEL_NAME_NSL, "nslkdd")
        run_nosplit(PROCESSED_CICIDS_CSV, MODELS_DIR_CICIDS, CALIBRATED_MODELS_DIR_CICIDS,
                    REGISTERED_RESULTS_PATH_AUTOMATA_CICIDS, KERAS_MODEL_NAME_CICIDS, "cicids2018")
        run_nosplit(PROCESSED_UNSW_CSV, MODELS_DIR_UNSW, CALIBRATED_MODELS_DIR_UNSW15,
                    REGISTERED_RESULTS_PATH_AUTOMATA_UNSW, KERAS_MODEL_NAME_UNSW, "unsw15")
        print("[Calibrated] => Proceso OFFLINE FIN (nosplit)")
        return


if __name__ == "__main__":
    main()
