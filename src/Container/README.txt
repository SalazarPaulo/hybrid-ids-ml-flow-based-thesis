# Sistema de Detección y Prevención de Intrusiones (IDPS)
python -m venv venv
venv\Scripts\activate  # En Windows
pip install matplotlib numpy transformers

Este proyecto implementa un **Sistema de Detección y Prevención de Intrusiones (IDPS)** por modulos y es escalable, 
diseñado para analizar tráfico de red y detectar anomalías en tiempo real. 
---

## Estructura del Proyecto

text
Container/
├── extensions/
│   ├── Auto/
│   │   ├── calibrated_predict.py
│   │   ├── preprocess.py
│   │   └── sub_automaton.py
│   └── Queue/
│       └── sub_queues.py
├── interface/
│   └── interface.pyw
├── nivel_1/
│   ├── capture.py
├── nivel_2/
│   ├── moe.py
│   └── models/
│       ├── svc.pkl
│       ├── svc_unsw15.pkl
│       ├── ...
│       └── README.txt
├── nivel_3/
│   ├── nlp.py
└── README.txt

---

1. **Nivel 1** (nivel_1): Captura y preprocesamiento de datos de red.  
2. **Nivel 2** (nivel_2): Análisis y procesamiento avanzado (MoE, modelos de IA).  
3. **Nivel 3** (nivel_3): Interpretación y presentación de resultados (NLP).  
4. **Enlaces** (extensions):
   - **Auto**: Contiene el subnivel de autómata avanzado.
   - **Queue**: Contiene el subnivel de teoría de colas.
---

## Descripción de Cada Nivel y Enlace

### Nivel 1: Captura (Carpeta nivel_1)
- **Responsabilidad**: Capturar paquetes de red y realizar un preprocesamiento inicial (limpieza, normalización, transformación de datos).  
- **Archivo Principal**: capture.py  
- **Detalles**:  
  - Usa bibliotecas **Scapy** y **Pyshark** para la captura en tiempo real.  

### Nivel 2: Procesamiento Avanzado (Carpeta nivel_2)
- **Responsabilidad**: Análisis avanzado del tráfico de red.  
- **Archivo Principal**: moe.py  
- **Detalles**:  
  - Implementa múltiples algoritmos de IA (XGBoost, LightGBM, CatBoost, SVC, Redes Neuronales, etc.).  
  - Mezcla de Expertos (MoE) para combinar resultados de varios modelos.  
  - Carpeta models almacena los modelos preentrenados de los dataset (NSL-KDD, CICIDS2018, UNSW-NB15).

### Nivel 3: Interpretación y Presentación (Carpeta nivel_3)
- **Responsabilidad**: Emplear técnicas de **NLP** y visualización para interpretar y presentar los resultados.  
- **Archivo Principal**: nlp.py  
- **Detalles**:  
  - Usa la biblioteca **Transformers** (modelo GPT-2) y **matplotlib** para generar gráficos comparativos.

### Enlaces (Carpeta extensions)
1. **Auto** (Enlace/Carpeta Primero):
   - **Archivo**: sub_automaton.py y calibrated_predict.py 
   - **Función**: calcular metricas y  realizar la calibracion. 
   - **Responsable**: dar las predicciones y probabilidades.

2. **Queue** (Enlace/Carpeta Segundo):
   - **Archivo**: sub_queues.py  
   - **Función**: Implementación de teoría de colas, priorizando resultados según la precisión o relevancia.  
   - **Responsable**: Manejar el flujo de datos y asignar prioridad a los análisis.

---
