#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
#  E N L A C E   2 :   T E O R Í A   D E   C O L A S   ( S U B N I V E L )
# ============================================================================
"""
queues.py
---------
Subnivel de teoría de colas para priorizar resultados producidos por el MoE.

Este módulo lee el archivo MOE correspondiente (por ejemplo, moe_output_nslkdd.json,
moe_output_cicids.json o moe_output_unsw.json), ordena sus entradas por
confianza y escribe un archivo de salida (queues_output_*.json).

Formato de ENTRADA:
- Puede ser una lista de dicts o un dict (en cuyo caso se usan sus .values()).
- Cada elemento debería incluir:
    - score o confidence (float): confianza del MoE para esa fila.
    - final_prediction o prediction (int): clase predicha.
    - model_contributions (dict): aportes de cada modelo base (opcional).

Formato de SALIDA (estándar para la UI):
- Un dict indexado por strings "0", "1", … con cada item:
    {
      "score": <float>,                 # siempre presente (map de confidence->score si aplica)
      "prediction": <int>,              # clase predicha
      "model_contributions": <dict>     # copiado tal cual si existía
    }

Notas:
- Se normaliza confidence -> score y final_prediction -> prediction si hace falta.
- Se evita usar la clase como clave (para no sobrescribir entradas con misma predicción).
- El orden final es descendente por score.
"""

import os
import json

# setup_path.py se encarga de agregar el path automáticamente
from utilities.setup_path import setup_project_path
setup_project_path()

from utilities.global_paths import (
    MOE_OUTPUT_JSON_NSLKDD,
    QUEUES_OUTPUT_JSON_NSLKDD,
    MOE_OUTPUT_JSON_CICIDS,
    QUEUES_OUTPUT_JSON_CICIDS,
    MOE_OUTPUT_JSON_UNSW,
    QUEUES_OUTPUT_JSON_UNSW
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get(d, *keys, default=None):
    """ funcion: Devuelve d[k] para el primer k existente en keys; si ninguno existe, retorna default."""
    for k in keys:
        if k in d:
            return d[k]
    return default


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------
def process_results_with_priority(results):
    """
    funcion:
    Ordena la lista de resultados por mayor score/confidence (descendente).

    Parámetros
    ----------
    results : list[dict]
        Lista de resultados MoE. Cada dict puede contener:
        - score o confidence (float)
        - final_prediction o prediction (int)
        - model_contributions (dict), opcional
        Cualquier otro campo adicional se ignora para el ordenamiento.

    Returns
    -------
    list[dict]
        Nueva lista de resultados, ordenada por confianza descendente.
        No se modifica la lista de entrada.

    Detalles
    --------
    - Si un elemento no tiene score pero sí confidence, se usa confidence.
    - Si ninguno existe, se asume 0.0 para el ordenamiento.
    - El contenido de cada elemento no es transformado aquí: solo se ordena.

    Ejemplo
    -------
    >>> process_results_with_priority([
    ...   {"confidence": 0.7, "final_prediction": 1},
    ...   {"score": 0.9, "prediction": 0}
    ... ])
    [
      {"score": 0.9, "prediction": 0, ...},   # primero por mayor confianza
      {"confidence": 0.7, "final_prediction": 1, ...}
    ]
    """
    print("[Teoría de Colas] Iniciando el procesamiento de resultados para asignar prioridades...")
    ordered = sorted(
        results,
        key=lambda x: x.get("score", x.get("confidence", 0.0)),
        reverse=True
    )
    for idx, r in enumerate(ordered):
        sc = r.get("score", r.get("confidence", 0.0))
        print(f"[Teoría de Colas] Resultado {idx} procesado con score: {sc}")
    print("[Teoría de Colas] Resultados ordenados por prioridad.")
    return ordered


def process_queue_for_dataset(moe_json_path, queues_json_path):
    """
    funcion:
    Lee el archivo MOE, ordena sus entradas por confianza y genera un JSON
    listo para la UI con claves "0", "1", …, siempre incluyendo score.

    Parámetros
    ----------
    moe_json_path : str
        Ruta del archivo JSON de entrada (salida del MoE).
        Acepta:
          - Lista de dicts
          - Dict de {id: dict}, en cuyo caso se usan sus values()
    queues_json_path : str
        Ruta del archivo JSON de salida para colas (consumido por la UI).

    Returns
    -------
    dict
        Diccionario ya normalizado e indexado por strings ("0", "1", ...),
        con campos:
          - "score" (float)
          - "prediction" (int)
          - "model_contributions" (dict)

    Comportamiento
    --------------
    - Normaliza confidence -> score y final_prediction -> prediction.
    - Garantiza un contenedor dict (no lista) para que la UI pueda hacer .values().
    - Preserva model_contributions si existe.
    - Mantiene el orden de mayor a menor score.

    Errores
    -------
    - Si el JSON no existe, se loguea y se retorna None.
    - Si el JSON es inválido, se loguea y se retorna None.
    """
    if not os.path.exists(moe_json_path):
        print(f"[Teoría de Colas] No existe {moe_json_path}. Abortando el proceso.")
        return None

    print(f"[Teoría de Colas] Leyendo {moe_json_path}...")
    try:
        with open(moe_json_path, "r", encoding="utf-8") as f:
            moe_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Teoría de Colas] Error al leer {moe_json_path}: {e}")
        return None

    # Normaliza la entrada: lista o dict -> lista
    if isinstance(moe_data, dict):
        items = list(moe_data.values())
    elif isinstance(moe_data, list):
        items = moe_data
    else:
        items = []
    print(f"[Teoría de Colas] Se han leído {len(items)} resultados desde {moe_json_path}.")

    # Ordena por score/confidence
    ordered = process_results_with_priority(items)

    # Estructura de salida: siempre dict con claves "0","1",... y clave 'score' presente
    final_dict = {}
    print("[Teoría de Colas] Creando diccionario final con los resultados priorizados...")
    for i, entry in enumerate(ordered):
        score = _get(entry, "score", "confidence", default=0.0)
        pred  = _get(entry, "final_prediction", "prediction", default=None)
        contrib = entry.get("model_contributions", {})

        final_dict[str(i)] = {
            "score": score,
            "prediction": pred,
            "model_contributions": contrib
        }
    print("[Teoría de Colas] Diccionario final creado correctamente.")

    # Persistencia
    os.makedirs(os.path.dirname(queues_json_path), exist_ok=True)
    with open(queues_json_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
    print(f"[Teoría de Colas] Archivo {queues_json_path} generado exitosamente.")

    return final_dict


def main():
    """ uncion:
    Programa principal: procesa los resultados del MoE y genera los archivos de colas para:
      1. NSL-KDD
      2. CICIDS2018
      3. UNSW15

    Efectos
    -------
    - Lee archivos MOE_OUTPUT_JSON_* (desde utilities.global_paths)
    - Escribe archivos QUEUES_OUTPUT_JSON_* con el formato esperado por la UI.
    """
    try:
        print("[Teoría de Colas] Procesando dataset NSL-KDD")
        process_queue_for_dataset(MOE_OUTPUT_JSON_NSLKDD, QUEUES_OUTPUT_JSON_NSLKDD)

        print("\n[Teoría de Colas] Procesando dataset CICIDS2018")
        process_queue_for_dataset(MOE_OUTPUT_JSON_CICIDS, QUEUES_OUTPUT_JSON_CICIDS)

        print("\n[Teoría de Colas] Procesando dataset UNSW15")
        process_queue_for_dataset(MOE_OUTPUT_JSON_UNSW, QUEUES_OUTPUT_JSON_UNSW)

        print("[Teoría de Colas] Proceso finalizado para todos los datasets.")
    except Exception as e:
        print(f"[Teoría de Colas] Error: {e}")

if __name__ == "__main__":
    main()
