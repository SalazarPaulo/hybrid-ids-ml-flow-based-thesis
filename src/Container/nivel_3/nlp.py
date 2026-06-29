import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
import re
import json
import os
import logging
import matplotlib.pyplot as plt

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Importar rutas de global_paths
from utilities.global_paths import (
    QUEUES_OUTPUT_JSON_NSLKDD,  # NSL-KDD
    DATASET_JSON,               # DATASET_JSON unificado con toda la información
    OUTPUT_DIR,
    QUEUES_OUTPUT_JSON_CICIDS,  # CICIDS2018
    QUEUES_OUTPUT_JSON_UNSW     # UNSW15
)

# Configurar el nivel de registro para depuración
logging.basicConfig(level=logging.INFO)
document_store = None


# ====================================================================-----
# 1. Función auxiliar para enviar mensajes
# ====================================================================-----
def send_message(callback, message):
    """funcion: Envía un mensaje al callback de la interfaz, o lo imprime si no está definido."""
    if callback:
        callback(message)
    else:
        print(message)

# ====================================================================-----
# 2. Función auxiliar para "split_command"
# ====================================================================-----
def split_command(cmd: str, min_parts: int, err_msg: str, callback):
    """
    funcion:
    Divide el comando por ':' y comprueba que tenga al menos min_parts,si 
    no, entonces va a mostrar err_msg y devolver None.
    """
    parts = cmd.split(":")
    if len(parts) < min_parts:
        send_message(callback, err_msg)
        return None
    return parts

# ====================================================================-----
# 3. Cargar dataset (DATASET_JSON) de información complementaria
# ====================================================================-----
def load_dataset(dataset_path, callback=None):
    """funion: Carga un archivo de dataset desde la ruta proporcionada."""
    if not os.path.exists(dataset_path):
        send_message(callback, f"[Error] No se encontró el archivo del dataset: {dataset_path}")
        return {}
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        send_message(callback, f"Dataset cargado exitosamente desde: {dataset_path}")
        return dataset
    except Exception as e:
        send_message(callback, f"[Error] No se pudo cargar el dataset: {e}")
        return {}

# ====================================================================-----
# 4. Función para explicar un modelo a partir del dataset complementario
# ====================================================================-----
def explain_model_from_dataset(dataset_section, key, callback=None):
    """
    funcion:
    Explica una clave dentro de dataset_section (puede ser un modelo
    o la sección 'prevención', realmente seria mas que todo el id del diccionario).
    """
    info = dataset_section.get(key)
    if not info:
        send_message(callback, f"No se encontró información para '{key}'.")
        return None

    # Si es "prevención", mostrar description + normal + alerta
    if key == "prevención":
        text = (
            f"{info['description']}\n"
            f"- normal: {info['normal']}\n"
            f"- alerta: {info['alerta']}"
        )
    else:
        # modelo: description, interpretation, example
        text = (
            f"Descripción: {info['description']}\n"
            f"Interpretación: {info.get('interpretation','')}\n"
            f"Ejemplo: {info.get('example','')}"
        )

    send_message(callback, text)
    return text

# ====================================================================-----
# 5. Cargar modelo GPT-2 (español)
# ====================================================================-----
def load_spanish_gpt2(callback=None):
    """Carga el modelo mrm8488/spanish-gpt2."""
    send_message(callback, "Cargando modelo mrm8488/spanish-gpt2 en CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/spanish-gpt2")
        model = AutoModelForCausalLM.from_pretrained("mrm8488/spanish-gpt2")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cpu"  # Forzar uso de CPU
        )
        send_message(callback, "Modelo cargado exitosamente.")
        return generator
    except Exception as e:
        send_message(callback, f"[Error] No se pudo cargar el modelo: {e}")
        return None

# ====================================================================-----
# 6. Procesamiento de los JSON (colas)
# ====================================================================-----
def load_json_data(json_path, callback=None):
    """Carga un archivo JSON desde la ruta proporcionada."""
    if not os.path.exists(json_path):
        send_message(callback, f"[Error] No se encontró el archivo JSON: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        send_message(callback, f"JSON cargado exitosamente desde: {json_path}")
        return data
    except Exception as e:
        send_message(callback, f"[Error] No se pudo cargar el archivo JSON: {e}")
        return None

def create_document_store(queue_data, dataset_info, callback=None):
    """
    funcion:
    es el document store para escribir o eliminar recibe el
    queue_data que es un dict de colas (nslkdd, cicids, unsw) y el
    dataset_info que es un dict cargado de DATASET_JSON
    """
    docs = []
    # 1) Indexar cada entrada de cola
    for ds_name, entries in queue_data.items():
        for entry_id, entry in entries.items():
            docs.append({
                "content": json.dumps(entry, indent=4),
                "meta": {"type": "queue", "dataset": ds_name, "id": entry_id}
            })
    # 2) Indexar cada sección de dataset.json
    for key, info in dataset_info.items():
        docs.append({
            "content": json.dumps(info, indent=4),
            "meta": {"type": "dataset_info", "key": key}
        })

    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.delete_documents()
    document_store.write_documents(docs)
    return document_store




def list_document_store(document_store, callback=None):
    """
    funcion:
    Lista todos los documentos que hay en el document_store,
    mostrando su dataset e ID para que se pueda verificar.
    """
    try:
        docs = document_store.get_all_documents()
        if not docs:
            send_message(callback, "El document_store está vacío.")
            return

        send_message(callback, f"Hay {len(docs)} documentos en el store:")
        for doc in docs:
            ds = doc.meta.get("dataset", "desconocido")
            id_ = doc.meta.get("id", doc.meta.get("key", "sin clave"))
            send_message(callback, f"- Dataset: {ds}, ID: {id_}")
    except Exception as e:
        send_message(callback, f"[Error] No pude listar el document_store: {e}")

def show_json_content(json_path, callback=None):
    """
    funcion:
    Lee y formatea el contenido del archivo JSON indicado.
    Envía el contenido formateado mediante el callback o lo imprime.
    """
    if not os.path.exists(json_path):
        send_message(callback, f"[Error] No se encontró el archivo JSON: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        formatted_json = json.dumps(data, indent=4)
        send_message(callback, f"Contenido del JSON:\n{formatted_json}")
        return formatted_json
    except Exception as e:
        send_message(callback, f"[Error] No se pudo mostrar el contenido del JSON: {e}")
        return None

# ====================================================================-----
# 7. Haystack QA Pipeline (legacy)
# ====================================================================-----
def setup_qa_pipeline(document_store, callback=None):
    """Configura el pipeline de QA con Haystack."""
    send_message(callback, "Configurando el pipeline de QA...")
    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", use_gpu=False)
    pipeline_obj = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    send_message(callback, "Pipeline de QA configurado exitosamente.")
    return pipeline_obj

def get_answer_from_pipeline(pipeline, query, callback=None):
    """Recupera y responde usando el pipeline de QA."""
    send_message(callback, f"Procesando la consulta: '{query}'...")
    try:
        prediction = pipeline.run(
            query=query,
            params={"Retriever": {"top_k": 4}, "Reader": {"top_k": 1}}
        )
        if prediction["answers"]:
            answer = prediction["answers"][0].answer
            send_message(callback, f"Respuesta: {answer}")
            return answer
        else:
            send_message(callback, "No encontré información relevante para tu consulta.")
            return "No encontré información relevante para tu consulta."
    except Exception as e:
        send_message(callback, f"[Error] Error en el pipeline de QA: {e}")
        return f"[Error] Error en el pipeline de QA: {e}"

# ====================================================================-----
# 8. Funciones RAG y QA extractivo en paralelo
# ====================================================================-----
def generate_rag_answer(query, document_store, text_generator, callback=None):
    """
    funcion:
    Ejecuta un pipeline RAG: recupera los documentos más relevantes con Haystack,
    formatea la info clave incluyendo dataset+ID, y genera una respuesta
    con GPT-2 basada en ese contexto.
    """

    try:
        retriever = BM25Retriever(document_store=document_store)
        # top_k para más contexto elevar 
        docs = retriever.retrieve(query, top_k=4)
        if not docs:
            msg = "No se encontró contexto relevante para tu consulta."
            send_message(callback, msg)
            return msg

        # Construir contexto legible de todos los docs
        context_lines = []
        for doc in docs:
            entry = json.loads(doc.content)
            ds_name = doc.meta.get("dataset", "desconocido")
            entry_id = doc.meta.get("id", "desconocido")
            score = entry.get("score", "N/A")
            pred = entry.get("prediction", None)
            pred_label = "alerta" if pred == 1 else "normal" if pred == 0 else "desconocido"

            context_lines.append(f"--- Dataset: {ds_name}, ID: {entry_id} ---")
            context_lines.append(f"Puntaje: {score}")
            context_lines.append(f"Predicción: {pred_label}")
            context_lines.append("Contribuciones de modelos (prob. clase 1):")
            for model_name, model_data in entry.get("model_contributions", {}).items():
                probs = model_data.get("probabilities", [])
                prob1 = probs[1] if len(probs) > 1 else "N/A"
                context_lines.append(f"- {model_name}: {prob1}")
            context_lines.append("")  # línea en blanco entre documentos

        context = "\n".join(context_lines).strip()

        # este es el Prompt para GPT-2
        prompt = (
            f"{context}\n\n"
            f"Pregunta: {query}\n"
            f"Respuesta:"
        )

        out = text_generator(prompt, max_new_tokens=60, truncation=True)
        generated = out[0]["generated_text"].replace(prompt, "").strip()
        send_message(callback, generated)
        return generated

    except Exception as e:
        msg = f"[Error] Falló la generación RAG: {e}"
        send_message(callback, msg)
        return msg

def get_qa_answer(query, qa_pipeline, callback=None):
    """
    funcion:
    Ejecuta una consulta con el pipeline extractivo de Haystack y devuelve
    la respuesta literal extraída.
    """
    try:
        pred = qa_pipeline.run(
            query=query,
            params={"Retriever": {"top_k": 4}, "Reader": {"top_k": 1}}
        )
        if pred["answers"]:
            ans = pred["answers"][0].answer
            send_message(callback, f"Respuesta QA:\n{ans}")
            return ans
        msg = "No se encontró una respuesta clara para tu consulta."
        send_message(callback, msg)
        return msg
    except Exception as e:
        msg = f"[Error] Falló el pipeline QA: {e}"
        send_message(callback, msg)
        return msg

# ====================================================================-----
# 9. Ejemplos de comandos (ayuda)
# ====================================================================-----
def show_command_examples(callback=None):
    """
    funcion: Mostrar los ejemplos de uso respectivo para cada comando.
    """
    examples = {
        "mostrar:<dataset>:<clave>:json": "mostrar:nslkdd:0:json",
        "mostrar:all:json": "mostrar:all:json",
        "explica:<dataset>:<clave>:<modelo>": "explica:nslkdd:0:svc_best_clf_model.pkl",
        "explica:modelos:<dataset>:<modelo>": "explica:modelos:unsw15:svc_best_clf_model.pkl",
        "explica:<dataset>:<clave>:<modelo>:probabilities": "explica:cicids2018:0:XGB_model.pkl:probabilities",
        "explica:<dataset>:<clave>:prevención": "explica:unsw15:0:prevención",
        "pregunta:<texto>": "pregunta:¿Qué modelo detectó la alerta?",
        "qa:<texto>": "qa:¿Cuál es la predicción para el ID 0?",
        "graficar:1:<dataset>": "graficar:1:nslkdd",
        "graficar:2:<dataset>:<ID>": "graficar:2:cicids2018:0"
    }
    for command, example in examples.items():
        send_message(callback, f"{command} == {example}")

def show_command_guide(callback=None):
    guide = """
Guía de Uso de Comandos:
----------------------------------------
1. mostrar:<dataset>:<clave>:json
2. mostrar:all:json
3. explica:<dataset>:<clave>:<modelo>
4. explica:modelos:<dataset>:<modelo>
5. explica:<dataset>:<clave>:<modelo>:probabilities
6. explica:<dataset>:<clave>:prevención
7. pregunta:<texto> -> RAG + GPT-2
8. qa:<texto>       -> QA extractivo
9. graficar:1:<dataset>
10. graficar:2:<dataset>:<ID>
11. listar:store
----------------------------------------
"""
    send_message(callback, guide)

# ====================================================================-----
# 10. Graficación
# ====================================================================-----
def ensure_output_dir_exists():
    """Verifica que el directorio de salida para las imágenes exista."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_scores(data, callback=None):
    """Genera un gráfico de puntajes por entrada."""
    try:
        ids = list(data.keys())
        scores = [data[id]["score"] for id in ids]
        plt.figure(figsize=(8, 4))
        plt.bar(ids, scores)
        plt.xlabel('IDs')
        plt.ylabel('Scores')
        plt.title('Puntajes por Entrada')
        plt.tight_layout()
        ensure_output_dir_exists()
        output_path = os.path.join(OUTPUT_DIR, "scores_plot.png")
        plt.savefig(output_path)
        send_message(callback, f"Gráfica de puntajes generada y guardada en '{output_path}'")
    except Exception as e:
        send_message(callback, f"[Error] Error al generar gráfico de puntajes: {e}")

def plot_model_contributions(data, entry_id, callback=None):
    """Genera un gráfico comparando las probabilidades de los modelos dentro de un ID."""
    try:
        entry = data.get(entry_id)
        if not entry:
            send_message(callback, f"[Error] No se encontró la entrada con ID '{entry_id}'.")
            return
        contributions = entry.get("model_contributions")
        if not contributions:
            send_message(callback, f"[Error] No se encontraron contribuciones en el ID '{entry_id}'.")
            return
        model_names = list(contributions.keys())
        probabilities = [model["probabilities"][1] for model in contributions.values()]
        plt.figure(figsize=(8, 4))
        plt.bar(model_names, probabilities)
        plt.xlabel('Modelos')
        plt.ylabel('Probabilidades (Clase 1)')
        plt.title(f'Probabilidades por Modelo para ID {entry_id}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        ensure_output_dir_exists()
        output_path = os.path.join(OUTPUT_DIR, f"model_contributions_plot_{entry_id}.png")
        plt.savefig(output_path)
        send_message(callback, f"Gráfica de contribuciones generada y guardada en '{output_path}'")
    except Exception as e:
        send_message(callback, f"[Error] Error al generar gráfico de contribuciones: {e}")

# ====================================================================-----
# 11. Interacción principal
# ====================================================================-----
def main(user_input=None, callback=None):
    
    global document_store
    """Modo de interacción principal sin input() (usado por la interfaz)."""
    send_message(callback, "=== Configuración del Chatbot ===")
    send_message(callback, "1. Usar JSON por defecto.")
    send_message(callback, "2. Proporcionar una ruta personalizada.")
    send_message(callback, "3. Salir.")
    if not user_input:
        return

    choice = user_input.strip()
    if choice == "3":
        send_message(callback, "Saliendo del programa...")
        return
    elif choice not in ["1", "2"]:
        send_message(callback, "Por favor selecciona una opción válida (1/2/3).")
        return

    # Cargar JSONs de colas
    if choice == "1":
        json_data_nslkdd = load_json_data(QUEUES_OUTPUT_JSON_NSLKDD, callback)
        json_data_cicids = load_json_data(QUEUES_OUTPUT_JSON_CICIDS, callback)
        json_data_unsw   = load_json_data(QUEUES_OUTPUT_JSON_UNSW, callback)
        json_data = {
            "nslkdd": json_data_nslkdd,
            "cicids": json_data_cicids,
            "unsw":   json_data_unsw
        }
    else:
        send_message(callback, "Proporciona la ruta del archivo JSON:")
        return

    # Cargar dataset complementario
    dataset = load_dataset(DATASET_JSON, callback)
    if not json_data or not dataset:
        send_message(callback, "[Error] No se pudo cargar el JSON o el dataset.")
        return


    # Paso: crear el document_store (store) y los pipelines 
    # json_data (colas) y dataset (dataset.json)
    document_store = create_document_store(json_data, dataset, callback)
    qa_pipeline     = setup_qa_pipeline(document_store, callback)
    text_generator  = load_spanish_gpt2(callback)

    send_message(callback, "=== Modo Chatbot Activo ===")
    send_message(callback, "Escribe 'salir' para terminar la conversación.")
    send_message(callback, "Comandos disponibles:")
    send_message(callback, "- mostrar:<dataset>:<clave>:json")
    send_message(callback, "- mostrar:all:json")
    send_message(callback, "- listar:store")
    send_message(callback, "- explica:<dataset>:<clave>:<modelo>")
    send_message(callback, "- explica:modelos:<dataset>:<modelo>")
    send_message(callback, "- explica:<dataset>:<clave>:<modelo>:probabilities")
    send_message(callback, "- explica:<dataset>:<clave>:prevención")
    send_message(callback, "- pregunta:<texto> (RAG + GPT-2)")
    send_message(callback, "- qa:<texto>       (QA extractivo)")
    send_message(callback, "- graficar:1:<dataset>")
    send_message(callback, "- graficar:2:<dataset>:<ID>")
    
    return qa_pipeline, text_generator, dataset, json_data

# ========================================
# Función para cálculo determinista de contribución de los modelos
# ========================================
def handle_contribuciones_id(query, json_data, callback=None):
    """
    funcion:
    Recupera la entrada por dataset e ID directamente de json_data
    y devuelve el modelo con mayor probabilidad de clase 1.
    """
    import re

    try:
        # 1. Extraer el ID
        m_id = re.search(r"ID\s*(\d+)", query, re.IGNORECASE)
        if not m_id:
            send_message(callback, "No encontré un ID en tu pregunta.")
            return None
        entry_id = m_id.group(1)

        # 2. Extraer el nombre del dataset
        m_ds = re.search(r"dataset\s+(\w+)", query, re.IGNORECASE)
        if not m_ds:
            send_message(callback, "No especificaste el dataset.")
            return None
        ds_name = m_ds.group(1).lower()

        # 3. Tomar la sección correspondiente de json_data
        entries = json_data.get(ds_name, {})
        entry   = entries.get(entry_id)
        if not entry:
            send_message(callback, f"No hay la entrada ID {entry_id} en el dataset {ds_name}.")
            return None

        # 4. Calcular el modelo con mayor probabilidad de clase 1
        contribs = entry.get("model_contributions", {})
        if not contribs:
            send_message(callback, "No hay contribuciones para esa entrada.")
            return None

        best_model, best_data = max(
            contribs.items(),
            key=lambda kv: kv[1].get("probabilities", [0,0])[1]
        )
        best_prob = best_data["probabilities"][1]

        resp = (
            f"En el dataset **{ds_name}**, para la entrada ID {entry_id}, "
            f"el modelo **{best_model}** contribuyó más con probabilidad {best_prob}."
        )
        send_message(callback, resp)
        return resp

    except Exception as e:
        send_message(callback, f"[Error] Al calcular contribuciones: {e}")
        return None



def process_user_input(user_input, qa_pipeline, text_generator, dataset, json_data, callback=None):
    """Procesa el input del usuario en el chat."""
    global document_store
    user_input = user_input.strip()

    # Salida
    if user_input in ["salir", "3"]:
        send_message(callback, "Sesión terminada.")
        return None

    # Ayuda y guía
    if user_input == "ayuda":
        show_command_examples(callback)
        return True
    if user_input == "guia":
        show_command_guide(callback)
        return True

    # mostrar:<dataset>:<clave>:json o mostrar:all:json
    if user_input.startswith("mostrar:"):
        try:
            parts = user_input.split(":")
            if parts[1].strip() == "all":
                send_message(callback, f"\nJSON de todos los datasets:\n{json.dumps(json_data, indent=4)}")
            elif len(parts) >= 4:
                ds_key      = parts[1].strip()
                subset      = json_data.get(ds_key)
                if not subset:
                    send_message(callback, f"[Error] No se encontró info para '{ds_key}'.")
                else:
                    key = parts[2].strip()
                    if key.isdigit():
                        keys_list = list(subset.keys())
                        idx = int(key)
                        if idx < len(keys_list):
                            key = keys_list[idx]
                        else:
                            send_message(callback, "[Error] Índice fuera de rango.")
                            return True
                    send_message(callback, f"\nJSON:\n{json.dumps(subset.get(key, 'Clave no encontrada.'), indent=4)}")
            else:
                send_message(callback, "[Error] Formato incorrecto para 'mostrar'.")
        except Exception:
            send_message(callback, "[Error] Fallo al procesar 'mostrar'.")
        return True

    # lista el contenido en document sotre
    if user_input.strip() == "listar:store":
        list_document_store(document_store, callback)
        return True

    # explica:modelos:<dataset>:<modelo>
    if user_input.startswith("explica:modelos:"):
        parts = split_command(user_input, 3, "[Error] Formato explica:modelos", callback)
        if parts:
            modelo = ":".join(parts[2:]).replace("explica:modelos:", "").strip()
            explain_model_from_dataset(dataset, modelo, callback)
        return True

    # explica:<dataset>:<clave>:<modelo> (sin sufijos)
    if user_input.startswith("explica:") and all(x not in user_input for x in [":probabilities", ":prevención"]):
        try:
            parts = user_input.split(":")
            if len(parts) >= 4:
                explain_model_from_dataset(dataset, parts[3].strip(), callback)
            else:
                send_message(callback, "[Error] Formato incorrecto para 'explica'.")
        except Exception:
            send_message(callback, "[Error] Fallo al procesar 'explica'.")
        return True

    # explica:<dataset>:<clave>:<modelo>:probabilities
    if user_input.startswith("explica:") and ":probabilities" in user_input:
        try:
            parts = user_input.split(":")
            if len(parts) == 5:
                ds_key = parts[1].strip()
                key    = parts[2].strip()
                model  = parts[3].strip()
                subset = json_data.get(ds_key, {})
                if key.isdigit():
                    keys_list = list(subset.keys())
                    idx = int(key)
                    if idx < len(keys_list):
                        key = keys_list[idx]
                probs = subset.get(key, {}).get("model_contributions", {}).get(model, {}).get("probabilities", [])
                if probs:
                    send_message(callback, f"Probabilidades: {probs}")
                else:
                    send_message(callback, "[Error] No hay probabilidades para ese modelo.")
            else:
                send_message(callback, "[Error] Formato incorrecto para probabilities.")
        except Exception as e:
            send_message(callback, f"[Error] Fallo al procesar probabilities: {e}")
        return True

    # explica:<dataset>:<clave>:prevención
    if user_input.startswith("explica:") and ":prevención" in user_input:
        try:
            parts = split_command(user_input, 3, "[Error] Formato prevención", callback)
            if parts:
                ds_key = parts[1].strip()
                key    = parts[2].strip()
                subset = json_data.get(ds_key, {})
                if key.isdigit():
                    keys = list(subset.keys())
                    idx = int(key)
                    if idx < len(keys):
                        key = keys[idx]
                entry = subset.get(key)
                if entry:
                    explain_model_from_dataset(dataset, "prevención", callback)
                    pred = entry.get("prediction")
                    if pred == 0:
                        send_message(callback, "(Resultado: normal)")
                    elif pred == 1:
                        send_message(callback, "(Resultado: alerta)")
                    else:
                        send_message(callback, "(Resultado desconocido)")
        except Exception as e:
            send_message(callback, f"[Error] Fallo al procesar prevención: {e}")
        return True

    # graficar:1:<dataset>
    if user_input.startswith("graficar:1:"):
        try:
            parts = user_input.split(":")
            if len(parts) >= 3:
                ds_key = parts[2].strip()
                js_map = {
                    "nslkdd": QUEUES_OUTPUT_JSON_NSLKDD,
                    "cicids": QUEUES_OUTPUT_JSON_CICIDS,
                    "unsw":   QUEUES_OUTPUT_JSON_UNSW
                }
                js = js_map.get(ds_key)
                data = load_json_data(js, callback) if js else None
                if data:
                    plot_scores(data, callback)
            else:
                send_message(callback, "[Error] Formato incorrecto para graficar:1.")
        except Exception as e:
            send_message(callback, f"[Error] Fallo graficar puntajes: {e}")
        return True

    # graficar:2:<dataset>:<ID>
    if user_input.startswith("graficar:2:"):
        try:
            parts = user_input.split(":")
            if len(parts) >= 4:
                ds_key   = parts[2].strip()
                entry_id = parts[3].strip()
                js_map = {
                    "nslkdd": QUEUES_OUTPUT_JSON_NSLKDD,
                    "cicids": QUEUES_OUTPUT_JSON_CICIDS,
                    "unsw":   QUEUES_OUTPUT_JSON_UNSW
                }
                js = js_map.get(ds_key)
                data = load_json_data(js, callback) if js else None
                if data:
                    plot_model_contributions(data, entry_id, callback)
            else:
                send_message(callback, "[Error] Formato incorrecto para graficar:2.")
        except Exception as e:
            send_message(callback, f"[Error] Fallo graficar contribuciones: {e}")
        return True

    # pregunta:<texto> -> RAG + GPT-2
    if user_input.startswith("pregunta:"):
        # Vía determinista
        try:
            result = handle_contribuciones_id(user_input, json_data, callback)
        except Exception as e:
            send_message(callback, f"[Error] Al procesar contribuciones: {e}")
            result = None

        # Si no obtuvo nada, usa RAG + GPT-2
        if not result:
            try:
                generate_rag_answer(user_input, document_store, text_generator, callback)
            except Exception as e:
                send_message(callback, f"[Error] Falló generación RAG: {e}")
        return True



    # qa:<texto> -> QA extractivo
    if user_input.startswith("qa:"):
        try:
            question = user_input.replace("qa:", "").strip()
            get_qa_answer(question, qa_pipeline, callback)
        except Exception as e:
            send_message(callback, f"[Error] Fallo en pregunta QA: {e}")
        return True

    # fallback GPT-2 básico
    try:
        out = text_generator(user_input, max_new_tokens=30, truncation=True)
        send_message(callback, out[0]["generated_text"].strip())
    except Exception as e:
        send_message(callback, f"[Error] Generación básica fallida: {e}")

    return True  # aqui diec que el chatbot sigue activo xd
"""
explica:modelos:nslkdd:svc_best_clf_model.pkl
explica:modelos:unsw15:XGB_model.pkl XX
explica:nslkdd:0:svc_best_clf_model.pkl
explica:unsw15:0:prevención
explica:cicids2018:0:XGB_model_bin_cicids2018_cicids2018:probabilities
-> el código vuelve todo a minuscula y luego busca en el dataset_json
-> generaba error en el match se ha comentado el lower solo se deha el strip.
"""
