import os
import shutil
import datetime
import json

# Importar las rutas globales
from global_paths import (
    BASE_DIR,
    CONTAINER_DIR,
    JSONS_DIR,       # Carpeta de origen de los JSON (con subdirectorios)
    WEB_JSON_DIR,    # Subdirectorio "web" dentro de JSONS_DIR
)

# Configuración: carpeta "history" se creará en CONTAINER_DIR
HISTORY_DIR = os.path.join(CONTAINER_DIR, "history")

def get_history_subfolder():
    """ funcion:
    Crea (o determina) una nueva carpeta de historial basada en la fecha actual.
    Si ya existe una carpeta con el mismo nombre, se le añade un sufijo numérico.
    Devuelve la ruta completa del nuevo directorio.
    """
    # Formato de fecha
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    base_name = today_str
    dest_folder = os.path.join(HISTORY_DIR, base_name)
    counter = 1
    # Si ya existe, agregar sufijo numérico
    while os.path.exists(dest_folder):
        dest_folder = os.path.join(HISTORY_DIR, f"{base_name}_{counter}")
        counter += 1
    return dest_folder

def copy_jsons_to_history():
    """
    funcion:
    Copia toda la carpeta JSONS_DIR (con sus subdirectorios y archivos)
    en un nuevo directorio dentro de HISTORY_DIR, excepto los archivos "dataset.json".
    Devuelve la ruta al directorio de historial creado.
    """
    # Asegurarse de que la carpeta de historial existe
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    dest_folder = get_history_subfolder()
    
    # Función ignore que evita copiar "dataset.json"
    def ignore_dataset(dir, files):
        # Si estamos en el directorio "web", no copiar dataset.json
        if os.path.basename(dir).lower() == "web":
            return ["dataset.json"]
        return []  # En otros directorios, no ignorar nada

    try:
        shutil.copytree(JSONS_DIR, dest_folder, ignore=ignore_dataset)
        print(f"[History] Se archivó JSONS_DIR en: {dest_folder}")
    except Exception as e:
        print(f"[Error] Al copiar a history: {e}")
        raise
    return dest_folder

def clear_jsons_folder():
    """
    funcion
    Elimina todos los archivos en JSONS_DIR (y subdirectorios) excepto 'dataset.json' 
    y algunos archivos CSV específicos.
    """
    exceptions = {
        "dataset.json",
        "test_data_cont_unsw15.csv",
        "test_data_cont_cicids2018.csv",
        "df_all_preprocessed.csv",
        "test_data_cont.csv",
        "nsl_augmented.csv",
            # ---> protege los JSON de referencia UNSW15:
        "unsw15_feature_list.json",
        "unsw15_scaler_stats.json",
        "unsw15.csv",
        "unsw15_processed.csv",
        "nslkdd_feature_list.json",
    }

    for root, dirs, files in os.walk(JSONS_DIR):
        for file in files:
            if file.lower() in exceptions:
                continue
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"[Clear] Eliminado archivo: {file_path}")
            except Exception as e:
                print(f"[Error] No se pudo eliminar {file_path}: {e}")


def archive_and_clear_jsons():
    """
    funcion
    Función principal que primero archiva la carpeta JSONS_DIR en history (ignorando dataset.json)
    y luego limpia (elimina) los archivos en JSONS_DIR (excepto dataset.json).
    """
    print("[History] Iniciando proceso de archivado de JSONS_DIR...")
    history_folder = copy_jsons_to_history()
    print("[History] Archivado completado. Ahora limpiando JSONS_DIR...")
    clear_jsons_folder()
    print("[History] Limpieza de JSONS_DIR completada.")
    return history_folder

if __name__ == "__main__":
    # Al ejecutar este script, se realizará el archivado y limpieza
    try:
        archive_and_clear_jsons()
    except Exception as e:
        print(f"[Error] Proceso de archivado y limpieza falló: {e}")
