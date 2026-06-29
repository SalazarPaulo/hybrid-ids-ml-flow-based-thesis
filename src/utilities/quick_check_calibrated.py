#!/usr/bin/env python3
# tools/quick_check_calibrated.py
# -*- coding: utf-8 -*-

import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import tensorflow as tf

from utilities import shared_utils as su

def load_models(models_dir, calibrated_dir=None, keras_model_name=None):
    """
    Carga .pkl y Keras opcional. Prefiere modelo calibrado en calibrated_dir si existe.
    """
    models = {}
    pkl_names = set()
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pkl"):
                pkl_names.add(f)
    if calibrated_dir and os.path.isdir(calibrated_dir):
        for f in os.listdir(calibrated_dir):
            if f.endswith(".pkl"):
                pkl_names.add(f)
    for f in pkl_names:
        cand = os.path.join(calibrated_dir, f) if calibrated_dir else None
        if cand and os.path.exists(cand):
            path = cand
        else:
            path = os.path.join(models_dir, f)
        try:
            m = joblib.load(path)
            models[os.path.splitext(f)[0]] = m
            print(f"[loaded] {path}")
        except Exception as e:
            print(f"[err] loading {path}: {e}")
    # Keras
    if keras_model_name:
        chosen = None
        if calibrated_dir:
            candk = os.path.join(calibrated_dir, keras_model_name)
            if os.path.exists(candk):
                chosen = candk
        orgk = os.path.join(models_dir, keras_model_name)
        if chosen is None and os.path.exists(orgk):
            chosen = orgk
        if chosen:
            try:
                k = tf.keras.models.load_model(chosen)
                models["simple_neural_network"] = k
                print(f"[loaded keras] {chosen}")
            except Exception as e:
                print(f"[err] loading keras {chosen}: {e}")
    return models

def calibrate_if_needed(models, X_train, y_train):
    """
    Para cada modelo que no tenga predict_proba, intenta calibrar con CV=3.
    Guarda en memoria (no en disco) el calibrated_model devuelto por CalibratedClassifierCV.
    """
    out = {}
    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            out[name] = m
            continue
        try:
            calib = su.mk_calibrator(m, cv=3, method='sigmoid', prefit=False)
            Xt = X_train.copy()
            # si el modelo tiene feature_names_in_, reindex
            names = su.get_model_feature_names(m)
            if names:
                Xt = Xt.reindex(columns=names, fill_value=0.0)
            calib.fit(Xt, y_train)
            out[name] = calib
            print(f"[calibrated in-mem] {name}")
        except Exception as e:
            print(f"[calib error] {name}: {e}")
            out[name] = m
    return out

def generate_predictions_and_metrics(models, X_test, y_test, dataset_type, jsons_dir=None):
    # Harmonize if UNSW
    d = dataset_type.lower()
    if d == "unsw15":
        X_test_df = su.harmonize_unsw15_schema(pd.DataFrame(X_test), models, jsons_dir)
    elif d == "cicids2018":
        X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    else:
        X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

    X_test_df = (X_test_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32))

    n = X_test_df.shape[0]
    registered = {i: {"__data_key__": su.make_data_key(X_test_df.iloc[i].values)} for i in range(n)}

    results_by_model = {}
    # predecir en paralelo
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for mname, mdl in models.items():
            futures[ex.submit(su.predict_model, mdl, X_test_df)] = mname
        for fut in as_completed(futures):
            mname = futures[fut]
            try:
                preds, probas = fut.result()
            except Exception as e:
                print(f"[err predict] {mname}: {e}")
                continue
            results_by_model[mname] = (preds, probas)
            for i in range(n):
                p_i = None
                if probas is not None:
                    try:
                        p = probas[i]
                        if hasattr(p, "tolist"):
                            p_i = p.tolist()
                        else:
                            # scalar?
                            p_i = float(p)
                    except Exception:
                        p_i = None
                registered[i][mname] = {"prediction": int(preds[i]), "probabilities": p_i}

    # metrics per model
    metrics = {}
    for mname, (preds, probas) in results_by_model.items():
        try:
            ytrue = None
            if y_test is not None and len(y_test) == n:
                ytrue = y_test.to_numpy().astype(int)
            if ytrue is None:
                metrics[mname] = {"note": "no labels"}
            else:
                probs_pos = None
                if probas is not None:
                    # si probas (n,2) tomar columna 1
                    try:
                        pp = np.asarray(probas)
                        if pp.ndim == 2 and pp.shape[1] >= 2:
                            probs_pos = pp[:,1]
                        elif pp.ndim == 1:
                            probs_pos = pp
                    except Exception:
                        probs_pos = None
                metrics[mname] = su.compute_binary_metrics(ytrue, preds, probs_pos)
        except Exception as e:
            metrics[mname] = {"error": str(e)}
    return registered, metrics

def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="Quick check for calibrated with first N rows")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--calibrated_dir", default=None)
    parser.add_argument("--keras_model_name", default=None)
    parser.add_argument("--dataset_type", default="nslkdd")
    parser.add_argument("--nrows", type=int, default=5000)
    parser.add_argument("--jsons_dir", default=None)
    parser.add_argument("--out_registered", default="registered_quick.csv")
    parser.add_argument("--out_metrics", default="metrics_quick.json")
    args = parser.parse_args()

    X_all, y_all = su.load_test_data(args.data_csv, args.dataset_type, nrows=args.nrows)
    print(f"[quick] loaded X shape {X_all.shape}, y {len(y_all)}")
    # load models
    models = load_models(args.models_dir, args.calibrated_dir, args.keras_model_name)
    # calibrate if needed (CV=3 on this small set)
    models = calibrate_if_needed(models, X_all, y_all)
    reg, metrics = generate_predictions_and_metrics(models, X_all, y_all, args.dataset_type, jsons_dir=args.jsons_dir)
    # save
    su.export_normalized_registered_results(reg, args.out_registered)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[quick] done. registered ->", args.out_registered, "metrics ->", args.out_metrics)

if __name__ == "__main__":
    main()
