import os

# Ajustar BASE_DIR para que apunte a la raíz del proyecto (src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("GLOBAL_PATHS_DEBUG") == "1":
    print(BASE_DIR)  # Ejemplo: "C:\Users\paulo\Downloads\TESIS\tesis_env\src"

# Carpetas principales
CONTAINER_DIR = os.path.join(BASE_DIR, "Container")
UTILITIES_DIR = os.path.join(BASE_DIR, "utilities")  # carpeta utilities
EXTENSIONS_DIR = os.path.join(CONTAINER_DIR, "extensions")
JSONS_DIR = os.path.join(CONTAINER_DIR, "jsons")
WEB_JSON_DIR = os.path.join(JSONS_DIR, "web")
INTERFACE_DIR = os.path.join(CONTAINER_DIR, "interface")
IMG_DIR = os.path.join(BASE_DIR, "img")  # Carpeta para imágenes
NIVEL_1_DIR = os.path.join(CONTAINER_DIR, "nivel_1")
NIVEL_2_DIR = os.path.join(CONTAINER_DIR, "nivel_2")
NIVEL_3_DIR = os.path.join(CONTAINER_DIR, "nivel_3")
EXTRA_DIR = os.path.join(CONTAINER_DIR, "Extra")
AUTO_DIR = os.path.join(EXTENSIONS_DIR, "Auto")
QUEUE_DIR = os.path.join(EXTENSIONS_DIR, "Queue")

# Scripts principales
CAPTURE_SCRIPT = os.path.join(NIVEL_1_DIR, "capture.py")
MOE_SCRIPT = os.path.join(NIVEL_2_DIR, "moe.py")
NLP_SCRIPT = os.path.join(NIVEL_3_DIR, "nlp.py")

# Scripts adicionales
AUTOMATON_SCRIPT = os.path.join(AUTO_DIR, "sub_automaton.py")
PREPROCESS_SCRIPT = os.path.join(AUTO_DIR, "preprocess.py")
CALIBRATED_PREDICT_SCRIPT = os.path.join(AUTO_DIR, "calibrated_predict.py")
QUEUES_SCRIPT = os.path.join(QUEUE_DIR, "sub_queues.py")

# Archivos JSON
DETECTIONS_JSON = os.path.join(JSONS_DIR, "data", "detections.json")
FINAL_DATA_JSON = os.path.join(JSONS_DIR, "data", "final_dataset.json")
# QUEUES_OUTPUT_JSON = os.path.join(WEB_JSON_DIR, "queues_output.json")
DATASET_JSON = os.path.join(WEB_JSON_DIR, "dataset.json")
OUTPUT_JSON_PATH = os.path.join(WEB_JSON_DIR, "moe_output.json")  # Para MOE NSL-KDD (por defecto)

# Archivos CSV
REGISTERED_RESULTS_PATH = os.path.join(JSONS_DIR, "test", "results.csv")

# === Secciones NSL-KDD ===
# NSLKDD_CSV = os.path.join(JSONS_DIR, "res", "nslkdd.csv")  # CSV con nslkdd sin procesar
# PROCESSED_NSLKDD_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "nslkdd_processed.csv")  # Preprocesado NSL-KDD
# PROCESSED_NSLKDD_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "test_data_cont.csv")
PROCESSED_NSLKDD_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "nsl_augmented.csv")
# Directorios de modelos para NSL-KDD
MODELS_DIR = os.path.join(NIVEL_2_DIR, "models", "nslkdd")

NSL_REF_FEATURES_JSON = os.path.join(JSONS_DIR, "res", "preprocessed", "nslkdd_feature_list.json")

# Directorio base para modelos calibrados
CALIBRATED_MODELS_BASE_DIR = os.path.join(NIVEL_2_DIR, "models", "calibrated_models")

# Subdirectorios específicos para cada dataset
CALIBRATED_MODELS_DIR_NSLKDD = os.path.join(CALIBRATED_MODELS_BASE_DIR, "nslkdd")
CALIBRATED_MODELS_DIR_CICIDS = os.path.join(CALIBRATED_MODELS_BASE_DIR, "cicids2018")
CALIBRATED_MODELS_DIR_UNSW15 = os.path.join(CALIBRATED_MODELS_BASE_DIR, "unsw15")

# Subdirectorios para Sub_Queues
QUEUES_OUTPUT_JSON_NSLKDD = os.path.join(WEB_JSON_DIR, "queues_output_nslkdd.json")
QUEUES_OUTPUT_JSON_CICIDS = os.path.join(WEB_JSON_DIR, "queues_output_cicids.json")
QUEUES_OUTPUT_JSON_UNSW = os.path.join(WEB_JSON_DIR, "queues_output_unsw.json")

# === Secciones NSL-KDD ===
# Archivos de datos de prueba para NSL-KDD
# TEST_DATA_PATH = os.path.join(JSONS_DIR, "res", "preprocessed", "test_data_cont.csv")
# TEST_DATA_PATH = os.path.join(JSONS_DIR, "res", "nslkdd.csv")
REGISTERED_RESULTS_PATH_AUTOMATA = os.path.join(JSONS_DIR, "test", "registered_results.csv")
MOE_OUTPUT_JSON_NSLKDD = os.path.join(JSONS_DIR, "test", "moe_output_nslkdd.json")

# === Secciones CICIDS2018 ===
PROCESSED_CICIDS_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "test_data_cont_cicids2018.csv")
# TEST_DATA_PATH_CICIDS = os.path.join(JSONS_DIR, "res", "cicids2018.csv")
MODELS_DIR_CICIDS = os.path.join(NIVEL_2_DIR, "models", "cicids2018")
REGISTERED_RESULTS_PATH_AUTOMATA_CICIDS = os.path.join(JSONS_DIR, "test", "registered_results_cicids.csv")
MOE_OUTPUT_JSON_CICIDS = os.path.join(JSONS_DIR, "test", "moe_output_cicids.json")

# CSV preprocesado final para CICIDS (AGREGADO)
CICIDS2018_CSV = os.path.join(JSONS_DIR, "res", "cicids2018.csv")  # CSV con cicids2018 sin procesar
# PROCESSED_CICIDS_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "cicids_processed.csv")

# === Secciones UNSW15 ===
# === Referencias persistentes para UNSW15 (esquema y stats del escalado) ===
UNSW15_SCHEMA_PATH = os.path.join(JSONS_DIR, "res", "preprocessed", "unsw15_feature_list.json")
UNSW15_STATS_PATH  = os.path.join(JSONS_DIR, "res", "preprocessed", "unsw15_scaler_stats.json")

# PROCESSED_UNSW_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "test_data_cont_unsw15.csv")
# TEST_DATA_PATH_UNSW = os.path.join(JSONS_DIR, "res", "unsw15.csv")
MODELS_DIR_UNSW = os.path.join(NIVEL_2_DIR, "models", "unsw15")
REGISTERED_RESULTS_PATH_AUTOMATA_UNSW = os.path.join(JSONS_DIR, "test", "registered_results_unsw.csv")
MOE_OUTPUT_JSON_UNSW = os.path.join(JSONS_DIR, "test", "moe_output_unsw.json")

# CSV preprocesado final para UNSW (AGREGADO)
UNSW15_CSV = os.path.join(JSONS_DIR, "res", "unsw15.csv")  # CSV con unsw15 sin procesar
PROCESSED_UNSW_CSV = os.path.join(JSONS_DIR, "res", "preprocessed", "unsw15_processed.csv")

# Recursos de interfaz
INTERFACE_IMAGE = os.path.join(IMG_DIR, "interface.jpeg")

# Carpeta para gráficos
OUTPUT_DIR = IMG_DIR  # Las imágenes se guardan en la carpeta img

# Rutas de scripts de utilidad
ARCHIVE_JSONS_SCRIPT = os.path.join(UTILITIES_DIR, "archive_json_history.py")
GLOBAL_PATHS_SCRIPT = os.path.join(UTILITIES_DIR, "global_paths.py")
SCRIPTS_MANAGER_SCRIPT = os.path.join(UTILITIES_DIR, "scripts_manager.py")
