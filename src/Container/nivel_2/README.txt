# Nivel 2: Algoritmo MoE (Mezcla de Expertos)

Este contenedor implementa el Nivel 2 del sistema de detección y prevención de intrusiones (IDPS). 
Su objetivo es utilizar algoritmos de Mezcla de Expertos (MoE) para combinar y analizar los resultados 
de múltiples modelos de aprendizaje automático, proporcionando una clasificación avanzada.

---

## Descripción General

El script principal `moe.py` implementa la técnica de ponderacion
 de **votación dura**, **votación blanda**, y **stacking** para la combinación de predicciones de diferentes modelos. 
Este nivel actúa como el núcleo de procesamiento avanzado en el sistema, utilizando múltiples expertos para mejorar la precisión de las detecciones.

---

## Funcionalidades

1. **Votación Dura (Hard Voting)**:
   - Combina las predicciones de múltiples modelos mediante mayoría de votos.

2. **Votación Blanda (Soft Voting)**:
   - Promedia las probabilidades predichas por los modelos para determinar la clase final.

3. **Stacking**:
   - Utiliza un meta-modelo que toma las salidas de los modelos base como entrada para hacer predicciones más precisas.

4. **Gestión de Modelos**:
   - Accede a una carpeta compartida donde están almacenados los modelos preentrenados, facilitando la actualización y personalización de estos.

---

## Requisitos Previos

### Dependencias de Python  
Instala las siguientes dependencias:  
```bash
pip install numpy scikit-learn joblib


