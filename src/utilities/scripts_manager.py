# scripts_manager.py
import os
import sys
import subprocess
import importlib.util
from utilities.global_paths import (
    CAPTURE_SCRIPT,
    PREPROCESS_SCRIPT,
    CALIBRATED_PREDICT_SCRIPT,
    AUTOMATON_SCRIPT,
    MOE_SCRIPT,
    QUEUES_SCRIPT,
    NLP_SCRIPT,
    ARCHIVE_JSONS_SCRIPT,
    QUEUES_OUTPUT_JSON_NSLKDD  # Se asume que esta es la ruta para el JSON de evaluación
)

##############################################################################
# CARGA DINÁMICA DEL MÓDULO "capture" DESDE LA RUTA CAPTURE_SCRIPT
##############################################################################
spec = importlib.util.spec_from_file_location("capture", CAPTURE_SCRIPT)
capture = importlib.util.module_from_spec(spec)
sys.modules["capture"] = capture
spec.loader.exec_module(capture)

##############################################################################
# FUNCIONES PARA MANEJO DE CAPTURA
##############################################################################
def start_capture():
    capture.start_sniffing()
    print("[Orquestación] Captura iniciada.")
    
# frame 1 => _stop_sniffing_task // Solo detener nivel 1
def stop_capture():
    # Paso 1: Archivar y limpiar archivos JSON antes de procesar
    run_archive_script()
    # Paso 2: Comienza la magia
    capture.stop_sniffing()
    print("[Orquestación] Captura detenida.")

# frame 1 => _finish_level_1_task // Finalizar nivel 1
def stop_capture_and_run_full_pipeline():
    # Paso 1: Archivar y limpiar archivos JSON antes de procesar
    run_archive_script()
    # Paso 2: Comienza la magia
    capture.stop_sniffing()
    run_full_pipeline()

##############################################################################
# FUNCIONES PARA OBTENER ALERTAS Y FEATURES DESDE CAPTURE.PY
##############################################################################
def get_alerts():
    """
    Retorna la lista de alertas desde capture.py.
    """
    return capture.get_alerts()

def get_features():
    """
    Retorna la lista de características extraídas desde capture.py.
    """
    return capture.get_features()

# -----------------------------------------------------------
# FUNCIONES PARA EJECUTAR OTROS SCRIPTS EXTERNOS
# -----------------------------------------------------------
def run_script(script_path, script_name):
    """
    funcion:
    Ejecuta un script secundario asegurando que utilities esté en el PYTHONPATH
    y añadiendo el argumento --no-gui para evitar que la GUI se inicie en
    subprocesos dentro del ejecutable PyInstaller.
    """
    print(f"\n[Orquestación] Ejecutando script: {script_name}...")

    env = os.environ.copy()
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    utilities_path = os.path.abspath(os.path.dirname(__file__))  # Ruta de utilities

    # Agregar paths al PYTHONPATH
    env["PYTHONPATH"] = f"{src_path};{utilities_path}"

    try:
        # -------------- CAMBIO PARA QUE NO SE ABRA NUEVAS  -----------------
        # -------------- INTERFACES AL HACER SYS.EXECUTABLE -----------------
        subprocess.run([sys.executable, script_path, "--no-gui"], check=True, env=env)
        # -----------------------------------------------------------
        print(f"[Orquestación] {script_name} finalizado con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Fallo al ejecutar {script_name}: {e}")


def run_preprocess_script():
    run_script(PREPROCESS_SCRIPT, "Preprocesamiento")

def run_calibrated_predict_script():
    run_script(CALIBRATED_PREDICT_SCRIPT, "Calibración y Predicción")

def run_automaton_script():
    run_script(AUTOMATON_SCRIPT, "Autómata")

def run_moe_script():
    run_script(MOE_SCRIPT, "Mixture of Experts (MoE)")

def run_queues_script():
    run_script(QUEUES_SCRIPT, "Teoría de Colas")

def run_archive_script():
    """
    Ejecuta el script de archivado de JSONS antes de sobrescribir archivos.
    """
    run_script(ARCHIVE_JSONS_SCRIPT, "Archivado de JSONS")
    # run_preprocess_script()

##############################################################################
# FLUJO COMPLETO (SECUENCIA DE EJECUCIÓN)
##############################################################################
def run_full_pipeline():
    """
    funcion:
    Ejecuta la secuencia completa de procesamiento de datos en el orden correcto.
    1. Archiva los archivos JSON y los limpia (excepto dataset.json).
    2. Ejecuta los scripts de preprocesamiento, predicción, autómata, MoE y teoría de colas.
    """
    print("\n[Orquestación] Iniciando el pipeline completo...")

    # Paso 2: Ejecutar la secuencia de procesamiento
    run_preprocess_script()
    run_calibrated_predict_script()
    run_automaton_script()
    run_moe_script()
    run_queues_script()

    print("\n[Orquestación] Pipeline completo finalizado :D.")

##############################################################################
# NLP (Carga dinámica del módulo NLP)
##############################################################################
def load_nlp_module(callback=None):
    print("[Orquestación] Cargando módulo NLP_SCRIPT dinámicamente...")
    try:
        importlib_spec = importlib.util.spec_from_file_location("nlp", NLP_SCRIPT)
        nlp_mod = importlib.util.module_from_spec(importlib_spec)
        importlib_spec.loader.exec_module(nlp_mod)
        print("[Orquestación] Módulo NLP cargado exitosamente.")
        return nlp_mod
    except Exception as e:
        msg = f"[Error] No se pudo cargar {NLP_SCRIPT}: {e}"
        print(msg)
        if callback:
            callback(msg)
        return None

##############################################################################
# FUNCIONES PARA MOSTRAR CONTENIDO EN INTERFAZ
##############################################################################
def show_json_content_in_interface(callback=None):
    """
    funcion:
    Lee y retorna el contenido de los tres JSON de evaluación (rutas:
    QUEUES_OUTPUT_JSON_NSLKDD, QUEUES_OUTPUT_JSON_CICIDS y QUEUES_OUTPUT_JSON_UNSW)
    mediante el callback. Esto separa la lógica de presentación de la interfaz.
    """
    from utilities.global_paths import QUEUES_OUTPUT_JSON_NSLKDD, QUEUES_OUTPUT_JSON_CICIDS, QUEUES_OUTPUT_JSON_UNSW
    content = {}

    rutas = {
        "nslkdd": QUEUES_OUTPUT_JSON_NSLKDD,
        "cicids": QUEUES_OUTPUT_JSON_CICIDS,
        "unsw": QUEUES_OUTPUT_JSON_UNSW
    }

    for key, path in rutas.items():
        if not os.path.exists(path):
            msg = f"[Error] No se encontró el archivo JSON para {key}: {path}"
            if callback:
                callback(msg)
            content[key] = f"[Error] Archivo no encontrado: {path}"
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                content[key] = data
            except Exception as e:
                msg = f"[Error] No se pudo mostrar el contenido del JSON para {key}: {e}"
                if callback:
                    callback(msg)
                content[key] = f"[Error] {e}"

    if callback:
        callback(f"Contenido de los JSON de evaluaciones:\n{json.dumps(content, indent=4)}")
    return content


##############################################################################
# EJECUCIÓN DIRECTA DESDE CONSOLA
##############################################################################
if __name__ == "__main__":
    print("[Orquestación] Iniciando scripts_manager...")
