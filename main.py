#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Launcher de Interfaz (GUI)
- no orquesta scripts; solo abre la UI.
- Aplica defaults de entorno si no existen.
- Permite overrides: --mode / --env KEY=VAL (repetible)
- Reenvía todo lo que va tras `--` a la interfaz.

Ejemplos:
  # Sin más: abre la interfaz y aplica defaults si faltan
  python main.py

  # Fijar modo por CLI (equivale a export RUNTIME_MODE=offline_nosplit)
  python main.py --mode offline_nosplit

  # Añadir variables de entorno antes de abrir la UI
  python main.py --env SU_TIMEOUT_SECONDS=0 --env MODEL_TIMEOUT_SECONDS=0

  # Reenviar flags a la propia interfaz (todo tras -- pasa a interface.pyw)
  python main.py -- --dataset nslkdd --theme dark
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# setup_path
try:
    from utilities.setup_path import setup_project_path
    setup_project_path()
except Exception:
    pass

# global_paths para ubicar la UI
try:
    import utilities.global_paths as GP
    INTERFACE_DIR = Path(getattr(GP, "INTERFACE_DIR", str(SRC_DIR / "Container" / "interface")))
except Exception:
    INTERFACE_DIR = SRC_DIR / "Container" / "interface"

# -------------------- Ayuda de entorno --------------------
ENV_HELP = dedent("""
Variables de entorno comunes

[Generales]
  PYTHONUNBUFFERED=1
  TF_ENABLE_ONEDNN_OPTS=0
  LIGHTGBM_VERBOSITY=0
  DATA_KEY_FORMAT=json

[Calibrated / Predict]
  RUNTIME_MODE=<offline_split|offline_nosplit|online_live|offline_eval>
  STACK_META_BATCH=20000
  CALIB_WORKERS=4
  PRED_WORKERS=4
  SPLIT_TRAIN_FRACTION=0.6
  SPLIT_CALIB_FRACTION=0.2
  SPLIT_TEST_FRACTION=0.2
  SPLIT_RANDOM_STATE=43
  SKIP_OHE_UNSW=0
  SKIP_META_KNN=0

[Automaton]
  AUTOMATON_THREADS=4
  SU_TIMEOUT_SECONDS=0
  MODEL_TIMEOUT_SECONDS=0
  AUTOMATON_FORCE_LOCAL=0
  AUTOMATON_CHUNK=50000
  EVAL_OFFLINE=0
  EVAL_MAX_ROWS=0
  EVAL_SEED=43
""").strip()

# -------------------- Defaults de entorno --------------------
DEFAULT_ENV = {
    # Generales
    "PYTHONUNBUFFERED": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "LIGHTGBM_VERBOSITY": "0",
    "DATA_KEY_FORMAT": "json",
    # Calibrated
    "RUNTIME_MODE": "offline_nosplit",
    "STACK_META_BATCH": "20000",
    "CALIB_WORKERS": "4",
    "PRED_WORKERS": "4",
    "SPLIT_TRAIN_FRACTION": "0.6",
    "SPLIT_CALIB_FRACTION": "0.2",
    "SPLIT_TEST_FRACTION":  "0.2",
    "SPLIT_RANDOM_STATE":   "43",
    "SKIP_OHE_UNSW": "0",
    "SKIP_META_KNN": "0",
    # Automaton
    "AUTOMATON_THREADS": "4",
    "SU_TIMEOUT_SECONDS": "0",
    "MODEL_TIMEOUT_SECONDS": "0",
    "AUTOMATON_FORCE_LOCAL": "0",
    "AUTOMATON_CHUNK": "50000",
    "EVAL_OFFLINE": "0",
    "EVAL_MAX_ROWS": "0",
    "EVAL_SEED": "43",
}

def _apply_default_env():
    """Setea defaults solo si no existen en el entorno."""
    for k, v in DEFAULT_ENV.items():
        if os.environ.get(k) is None:
            os.environ[k] = v

def _apply_overrides(mode_arg, env_kv_list):
    """--mode -> RUNTIME_MODE; --env KEY=VAL (repetible)."""
    if mode_arg:
        os.environ["RUNTIME_MODE"] = mode_arg
    for kv in env_kv_list or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[str(k).strip()] = str(v).strip()

def _autodetect_ui():
    """Busca interface.pyw."""
    candidates = [
        INTERFACE_DIR / "interface.pyw",
        INTERFACE_DIR / "interface.py",
        SRC_DIR / "Container" / "interface" / "interface.pyw",
        SRC_DIR / "Container" / "interface" / "interface.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _run_ui(python_exe, script_path, args=None, env=None, cwd=None):
    """Lanza la UI heredando la consola para ver los logs."""
    cmd = [python_exe, str(script_path)]
    if args:
        cmd += list(args)
    myenv = os.environ.copy()
    if env:
        myenv.update(env)
    # Unir PYTHONPATH con src/
    src_abs = str(SRC_DIR.resolve())
    if myenv.get("PYTHONPATH"):
        parts = myenv["PYTHONPATH"].split(os.pathsep)
        if src_abs not in parts:
            myenv["PYTHONPATH"] = src_abs + os.pathsep + myenv["PYTHONPATH"]
    else:
        myenv["PYTHONPATH"] = src_abs
    return subprocess.call(cmd, env=myenv, cwd=str(cwd or SRC_DIR))


def main():
    if os.getenv("MAIN_VERBOSE", "1") == "1":
        print(str(SRC_DIR.resolve()))
    if os.getenv("MAIN_VERBOSE", "1") == "1":
        try:
            from utilities import __version__ as uver
            print(f"Paquete 'utilities' inicializado. Versión (MAIN): {uver}")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Launcher de Interfaz (GUI) — aplica entorno y abre la UI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=ENV_HELP
    )
    parser.add_argument("--ui-path", type=str, default=None,
                        help="Ruta al script de la interfaz (por defecto autodetecta interface.pyw).")
    parser.add_argument("--mode", "-m", type=str, default=None,
                        choices=["offline_split", "offline_nosplit", "online_live", "offline_eval"],
                        help="Atajo para exportar RUNTIME_MODE antes de abrir la UI.")
    parser.add_argument("--env", action="append", default=[],
                        help="Establecer variables de entorno KEY=VAL (repetible).")
    parser.add_argument("--env-help", action="store_true",
                        help="Mostrar guía de variables de entorno y salir.")

    # Todo lo que vaya tras `--` se reenvía tal cual a la UI
    args, passthrough = parser.parse_known_args()

    if args.env_help:
        print(ENV_HELP)
        return

    # 1) Defaults si no existen
    _apply_default_env()
    # 2) Overrides de CLI
    _apply_overrides(args.mode, args.env)

    # Log útil
    # print(str(SRC_DIR.resolve()))
    if os.environ.get("RUNTIME_MODE"):
        print(f"[Main] RUNTIME_MODE = '{os.environ['RUNTIME_MODE']}'")
    # lanza la interfaz no la logica, el resto lo maneja la gui
    ui_path = Path(args.ui_path) if args.ui_path else _autodetect_ui()
    if not ui_path or not ui_path.exists():
        print("[Main] Error: No se encontró el script de la interfaz.")
        print("       Pasa la ruta explícita con --ui-path o ajusta el autodetector en main.py")
        sys.exit(2)

    print(f"[Main] Lanzando interfaz: {ui_path}")
    exit_code = _run_ui(sys.executable, ui_path, args=passthrough)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
