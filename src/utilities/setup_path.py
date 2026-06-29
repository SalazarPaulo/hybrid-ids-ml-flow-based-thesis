import os
import sys

def setup_project_path():
    """Asegura que la ruta del proyecto (`src`) esté en sys.path."""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    utilities_path = os.path.abspath(os.path.dirname(__file__))  # Ruta de utilities
    
    # Agregar rutas si no están en sys.path
    for path in [src_path, utilities_path]:
        if path not in sys.path:
            sys.path.append(path)

# Ejecutar automáticamente cuando se importe este módulo
setup_project_path()
