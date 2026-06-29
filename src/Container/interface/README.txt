# Interfaz del Sistema IDPS

Esta carpeta (`interface`) contiene los archivos y scripts necesarios para **orquestar** la ejecución y brindar una **interfaz gráfica** al Sistema de Detección y Prevención de Intrusiones (IDPS). Aquí destacan dos elementos principales:

1. **Archivo `docker-compose.yml`**: define y arranca los contenedores correspondientes a los tres niveles (Captura, MoE, NLP) y los enlaces (Autómata, Teoría de Colas).
2. **Archivo(s) `interface.pyw` / `interface.py` / `main.py`**: proveen una interfaz gráfica o lógica de control sobre los contenedores, permitiendo la ejecución de tareas específicas en cada servicio.

---

## Contenido de la Carpeta

1. **`docker-compose.yml`**  
   - Orquesta los contenedores del IDPS, especificando dependencias, puertos y salud de los servicios.  
   - Permite arrancar y detener todos los niveles (Captura, MoE, NLP) y los enlaces (Autómata, Teoría de Colas) con un solo comando.

2. **`interface.jpeg`** *(opcional)*  
   - Imagen que puede ilustrar la arquitectura global del sistema o el diagrama de interacción entre contenedores.

3. **`interface.pyw`** *(o `interface.py` / `main.py`)*  
   - Script que implementa la interfaz gráfica (usando **CustomTkinter**, etc.).
   - Ofrece pestañas para cada nivel y subnivel (Captura, IDPS/MoE, NLP, Autómata, Teoría de Colas).
   - Permite ejecutar tareas específicas en cada contenedor vía `docker-compose exec`.

4. **`README.txt`** *(este documento)*  
   - Explica la función de la carpeta, la forma de usar el `docker-compose.yml`, y la interfaz gráfica si se dispone de ella.

---

## Uso del `docker-compose.yml`

1. **Instalar Docker y Docker Compose**  
   - Asegúrate de tener Docker y Docker Compose instalados en tu sistema.

2. **Navegar a la Carpeta `interface`**  
   - Ubícate donde reside el archivo `docker-compose.yml`.

3. **Construir y Levantar los Servicios**  
   ```bash
   docker-compose build
    ```bash
   docker-compose up -d

---

## Verificar el Estado de los Contenedores
1. **Muestra los servicios en ejecución y su estado** 
   ```bash
    docker-compose ps

2. **Detiener y eliminar los contenedores, redes y volúmenes asociados** 
   ```bash
    docker-compose down

---

## Interfaz Gráfica (interface.pyw)
Este script ofrece una ventana con pestañas (o marcos) para cada nivel y enlace:
1. **Nivel 1 (Captura)**  
   - Permite iniciar o detener captura de paquetes, escaneo de puertos, etc.
2. **Nivel 2 (IDPS/MoE)**  
   - Opciones para analizar el riesgo de intrusión, probar la mezcla de expertos, etc.
3. **Nivel 3 (NLP)**  
   - Genera interpretaciones en lenguaje natural, gráficos comparativos, etc.
4. **Enlace Autómata**  
   - Registra patrones, ejecuta análisis en paralelo con varios modelos.
5. **Enlace Teoría de Colas**  
   - Prioriza los resultados según precisión y relevancia.

---

## Ejecución de Tareas en Cada Contenedor
1. **Comunicarse con los contenedores activos** 
Muestra la salida en un cuadro de texto dentro de la GUI para monitorear la actividad.
   ```bash
    docker-compose exec <servicio> python <script> <args...> 

---

## Ejecución de interface.pyw (Opcional)
1. **Lanzar la interfaz gráfica en local** 
   ```bash
    python interface.pyw

2. **Iniciar Servicios desde la raiz de la carpeta** 
   ```bash
    docker-compose up -d.

3. **Abrir la Interfaz** 
    - Ejecuta python interface.pyw. 
    - Aparece una ventana con pestañas para cada nivel/enlace.

4. **Operar Servicios** 

1. Haz clic en botones (Capturar Paquetes, Escanear Puertos, etc.) para ejecutar tareas en capture_service.
2. Haz clic en “Riesgo de Intrusión” o “Prueba MoE” para interactuar con moe_service (Nivel 2).
3. Navega a la pestaña “NLP (N3)” para demos o interpretaciones.
4. O entra a las pestañas de “Autómata” y “Teoría de Colas” para gestionarlos.

---

## Monitoreo:
1. **La salida de cada comando se mostrará en un cuadro de texto** 

2. **Nombres de los Servicios** 
    - **capture_service** - Contenedor para el Nivel 1 (Captura).
    - **automaton_service** - Contenedor para el enlace Autómata.
    - **moe_service** - Contenedor para el Nivel 2 (MoE).
    - **queue_service** - Contenedor para el enlace Teoría de Colas.
    - **nlp_service** - Contenedor para el Nivel 3 (NLP).

3. **Logs adicionales se pueden ver con en la terminal** 
   ```bash
   docker-compose logs <nombre_servicio> 
   
---

## Recomendaciones y Notas
1. **Healthchecks** 
    - Cada contenedor define un test en docker-compose.yml para verificar que su proceso principal (python) esté corriendo.

2. **Permisos Especiales** 
    - capture_service requiere cap_add: NET_ADMIN y NET_RAW para capturar paquetes en bajo nivel.

3. **Volúmenes Compartidos**    
    - El contenedor moe_service mapea la carpeta de models para actualizar modelos sin reconstruir la imagen.

4. **Actualización Continua** 
    - Eventualmente se debe reentrena o ajustar hiperparámetros (Optuna, etc.) para reflejar nuevas amenazas.
