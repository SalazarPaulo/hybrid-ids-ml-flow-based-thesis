# _init_placeholder.py
# Este archivo marca este directorio como un paquete de Python.
# paraimportar módulos.

try:
    from . import scripts_manager
    from . import global_paths 
    from . import setup_path 

except ImportError as e:
    print(f"[Error] No se pudo importar módulos en _init_.py: {e}")

PACKAGE_NAME = "utilities"
VERSION = "1.0.0"

print(f"Paquete '{PACKAGE_NAME}' inicializado. Versión: {VERSION}")
