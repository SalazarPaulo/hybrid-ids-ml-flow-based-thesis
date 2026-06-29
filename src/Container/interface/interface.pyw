#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import customtkinter as ctk
import importlib.util
import json
import threading
from customtkinter import CTkImage
from tkinter import ttk, Text, END
import tkinter as tk

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# se agregar la ruta 'src' al sys.path si no lo detecta
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if src_path not in sys.path:
    sys.path.append(src_path)
    
# Importa las funciones de utilities.scripts_manager
from utilities.scripts_manager import (
    start_capture,
    stop_capture,
    stop_capture_and_run_full_pipeline,
    get_alerts,
    get_features,
    run_calibrated_predict_script,
    run_automaton_script,
    run_moe_script, 
    run_queues_script,
    load_nlp_module
)
import utilities.scripts_manager as scripts_manager

# Importa utilities.global_paths
import utilities.global_paths as paths
from utilities.global_paths import (
    CAPTURE_SCRIPT, 
    PREPROCESS_SCRIPT,
    MOE_SCRIPT,
    NLP_SCRIPT,
    AUTOMATON_SCRIPT,
    QUEUES_SCRIPT,
    QUEUES_OUTPUT_JSON_NSLKDD,
    INTERFACE_IMAGE,
    QUEUES_OUTPUT_JSON_NSLKDD,
    QUEUES_OUTPUT_JSON_CICIDS,
    QUEUES_OUTPUT_JSON_UNSW,
)



# Funciones para el frame 4 : terminal
class TerminalRedirector:
    """
    funcion:
    Redirige stdout y stderr a un CTkTextbox (o similar) 
    y colorea el texto según ciertas palabras clave.
    """
    def __init__(self, text_widget, original_stream):
        self.text_widget = text_widget       # El CTkTextbox (o Frame con ._textbox)
        self.original_stream = original_stream

    def write(self, text):
        # 1) Habilitar la inserción en el widget
        self.text_widget.configure(state="normal")

        # 2) Determinar el tag a usar
        #    (estos son patrones)
        if "[Error]" in text:
            tag = "error"
        elif "[INFO]" in text:
            tag = "info"
        elif "[Orquestación]" in text:
            tag = "script1"
        else:
            tag = "default"

        # 3) Insertar el texto usando el widget interno _textbox (para CustomTkinter)
        self.text_widget._textbox.insert("end", text, tag)
        self.text_widget._textbox.see("end")

        # 4) Volver a deshabilitar
        self.text_widget.configure(state="disabled")

        # 5) Además, imprimir en el stream original (es útil para la depuración)
        self.original_stream.write(text)

    def flush(self):
        # Asegura que el stream original también se vacíe correctamente
        self.original_stream.flush()

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE CustomTkinter
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ---------------------------------------------------------------------------
# FUNCIONES PARA EJECUTAR SCRIPTS LOCALES
# ---------------------------------------------------------------------------
def execute_local_script(script_path, *args):
    cmd = [sys.executable, script_path] + list(args)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(proc.stdout.readline, ''):
            print(line, end='')
        proc.stdout.close()
        proc.wait()
    except Exception as e:
        print(f"Error ejecutando {script_path}: {e}")

def execute_local_script_with_output(script_path, *args):
    cmd = [sys.executable, script_path] + list(args)
    output_lines = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            output_lines.append(line.rstrip('\n'))
        proc.stdout.close()
        proc.wait()
    except Exception as e:
        output_lines.append(f"Error ejecutando {script_path}: {e}")
    return output_lines

# ---------------------------------------------------------------------------
# CLASE PRINCIPAL
# ---------------------------------------------------------------------------
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Interfaz - IDS")
        self.geometry("1000x700")

        # Cargar imagen ( logo )
        self.logo_image = None
        if os.path.exists(INTERFACE_IMAGE):
            from PIL import Image
            img = Image.open(INTERFACE_IMAGE)
            self.logo_image = CTkImage(img, size=(50, 50))

        # Contenedor principal para frames
        self.container_frame = ctk.CTkFrame(self)
        self.container_frame.pack(side="right", fill="both", expand=True)

        # Diccionario de frames
        self.frames = {}

        # Registra los frames que se va a usar
        for F in (HomeFrame, Nivel1Frame, Nivel2Frame, Nivel3Frame, TerminalFrame):
            frame = F(parent=self.container_frame, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Barra de navegación (izquierda)
        self.nav_frame = ctk.CTkFrame(self, width=150)
        self.nav_frame.pack(side="left", fill="y")

        ctk.CTkButton(self.nav_frame, text="Home",
                      command=lambda: self.show_frame(HomeFrame)).pack(pady=5, padx=5)
        ctk.CTkButton(self.nav_frame, text="Nivel 1",
                      command=lambda: self.show_frame(Nivel1Frame)).pack(pady=5, padx=5)
        ctk.CTkButton(self.nav_frame, text="Nivel 2",
                      command=lambda: self.show_frame(Nivel2Frame)).pack(pady=5, padx=5)
        ctk.CTkButton(self.nav_frame, text="Nivel 3 (NLP)",
                      command=lambda: self.show_frame(Nivel3Frame)).pack(pady=5, padx=5)
        ctk.CTkButton(self.nav_frame, text="Terminal",
              command=lambda: self.show_frame(TerminalFrame)).pack(pady=5, padx=5)

        # Mostrar Home al inicio
        self.show_frame(HomeFrame)

    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        frame.tkraise()

    # -----------------------------------------------------------------------
    # MÉTODOS PARA EJECUTAR SCRIPTS SECUNDARIOS
    # -----------------------------------------------------------------------
    def run_preprocess_script(self):
        print("[INFO] Ejecutando preprocess.py...")
        execute_local_script(PREPROCESS_SCRIPT)

    def run_moe_and_queues(self):
        print("[INFO] Ejecutando moe.py...")
        execute_local_script(MOE_SCRIPT)
        print("[INFO] Finalizó moe.py, ahora sub_queues.py...")
        execute_local_script(QUEUES_SCRIPT)

    def run_automaton_script(self):
        print("[INFO] Ejecutando sub_automaton.py...")
        execute_local_script(AUTOMATON_SCRIPT)

# ---------------------------------------------------------------------------
# FRAME: HOME
# ---------------------------------------------------------------------------
class HomeFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Título de bienvenida
        label = ctk.CTkLabel(
            self,
            text="BIENVENIDO A LA INTERFAZ \n\nSelecciona un nivel a la izquierda",
            font=("Arial", 16, "bold"),
            justify="center"
        )
        label.pack(pady=40)

        # Mensaje de explicación de los badges
        badges_label = ctk.CTkLabel(
            self,
            text=(
                "🎖️ **Criterios para Badges de la Tabla Probabilidades Por Modelo**:\n"
                "🥇 Oro: Probabilidad > 90%\n"
                "🥈 Plata: Probabilidad entre 60% y 90%\n"
                "🥉 Bronce: Probabilidad entre 50% y 60%"
            ),
            font=("Arial", 14),
            justify="left",
            text_color="white",
            anchor="w",
            fg_color="#2c2c2c",
            corner_radius=8,
            padx=15,
            pady=10,
            width=500  # Ajusta el ancho del cuadro si es necesario
        )
        badges_label.pack(pady=20)

        # Ejemplo visual de los badges con colores destacados
        example_frame = ctk.CTkFrame(self, fg_color="transparent")
        example_frame.pack(pady=10)

        # Ejemplo: Badge Oro
        ctk.CTkLabel(
            example_frame,
            text="🥇 Oro",
            font=("Arial", 14, "bold"),
            fg_color="#FFA500",  # Naranja brillante
            text_color="#2A2A2A",
            corner_radius=8,
            width=100,
            height=40,
            anchor="center"
        ).grid(row=0, column=0, padx=10)

        # Ejemplo: Badge Plata
        ctk.CTkLabel(
            example_frame,
            text="🥈 Plata",
            font=("Arial", 14, "bold"),
            fg_color="#C0C0C0",  # Plateado
            text_color="#2A2A2A",
            corner_radius=8,
            width=100,
            height=40,
            anchor="center"
        ).grid(row=0, column=1, padx=10)

        # Ejemplo: Badge Bronce
        ctk.CTkLabel(
            example_frame,
            text="🥉 Bronce",
            font=("Arial", 14, "bold"),
            fg_color="#CD7F32",  # Bronce
            text_color="#2A2A2A",
            corner_radius=8,
            width=100,
            height=40,
            anchor="center"
        ).grid(row=0, column=2, padx=10)

# ---------------------------------------------------------------------------
# FRAME: NIVEL 1
# ---------------------------------------------------------------------------
class Nivel1Frame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="#2A2A2A")
        self.controller = controller

        self.status_label = ctk.CTkLabel(
            self,
            text="Estado: Captura detenida",
            text_color="red",
            font=("Arial", 14),
            fg_color="#2A2A2A"
        )
        self.status_label.pack(pady=10)

        self.button_frame = ctk.CTkFrame(self, fg_color="#2A2A2A")
        self.button_frame.pack(pady=5)

        # Botón para INICIAR captura
        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Iniciar Captura",
            command=self.start_sniffing,
            fg_color="green",
            text_color="white"
        )
        self.start_button.pack(side="left", padx=10)

        # Botón para DETENER captura
        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Detener Captura",
            command=self.stop_sniffing,
            fg_color="red",
            text_color="white"
        )
        self.stop_button.pack(side="left", padx=10)

        # Botón para FINALIZAR NIVEL 1 (detener + pipeline)
        self.finish_button = ctk.CTkButton(
            self.button_frame,
            text="Finalizar Nivel 1",
            command=self.finish_level_1,
            fg_color="#555555",
            text_color="white"
        )
        self.finish_button.pack(side="left", padx=10)

        # -------------------------
        # ALERTAS
        # -------------------------
        self.alert_frame = ctk.CTkFrame(self, fg_color="#2A2A2A")
        self.alert_frame.pack(pady=10)

        self.alert_label = ctk.CTkLabel(
            self.alert_frame,
            text="Alertas Detectadas",
            font=("Arial", 14, "bold"),
            fg_color="#2A2A2A"
        )
        self.alert_label.pack(pady=5)

        self.alert_list = ttk.Treeview(
            self.alert_frame,
            columns=("timestamp", "src_ip", "dst_ip", "attack"),
            show="headings",
            height=8
        )
        self.alert_list.pack()
        self.alert_list.heading("timestamp", text="Hora")
        self.alert_list.heading("src_ip", text="IP Origen")
        self.alert_list.heading("dst_ip", text="IP Destino")
        self.alert_list.heading("attack", text="Ataque")

        # -------------------------
        # CARACTERÍSTICAS
        # -------------------------
        self.feature_frame = ctk.CTkFrame(self, fg_color="#2A2A2A")
        self.feature_frame.pack(pady=10)

        self.feature_label = ctk.CTkLabel(
            self.feature_frame,
            text="Características Extraídas",
            font=("Arial", 14, "bold"),
            fg_color="#2A2A2A"
        )
        self.feature_label.pack(pady=5)

        self.feature_list = ttk.Treeview(
            self.feature_frame,
            columns=("timestamp", "src_ip", "dst_ip", "protocol", "length"),
            show="headings",
            height=8
        )
        self.feature_list.pack()
        self.feature_list.heading("timestamp", text="Hora")
        self.feature_list.heading("src_ip", text="IP Origen")
        self.feature_list.heading("dst_ip", text="IP Destino")
        self.feature_list.heading("protocol", text="Protocolo")
        self.feature_list.heading("length", text="Tamaño")

        # Actualizaciones periódicas
        self.update_alerts()
        self.update_features()

    def start_sniffing(self):
        """
        funcion: Inicia la captura usando scripts_manager.
        """
        start_capture()
        self.status_label.configure(text="Estado: Capturando tráfico", text_color="green")

    def stop_sniffing(self):
        """Inicia un hilo para detener la captura sin bloquear la interfaz."""
        threading.Thread(target=self._stop_sniffing_task, daemon=True).start()

    def _stop_sniffing_task(self):
        stop_capture()  # Llamada a la función de scripts_manager
        # Programar la actualización de la interfaz en el hilo principal
        self.after(0, lambda: self.status_label.configure(text="Estado: Captura detenida", text_color="red"))

    def finish_level_1(self):
        """Inicia un hilo para detener la captura, ejecutar el pipeline y cambiar de frame."""
        threading.Thread(target=self._finish_level_1_task, daemon=True).start()

    def _finish_level_1_task(self):
        stop_capture_and_run_full_pipeline()  # Llamada a la función de scripts_manager
        # Actualizar la interfaz en el hilo principal
        self.after(0, lambda: self.status_label.configure(text="Estado: Captura detenida (y preprocesada)", text_color="red"))
        self.after(0, lambda: self.controller.show_frame(Nivel2Frame))

    def update_alerts(self):
        """Refresca la lista de alertas cada 1 segundo."""
        self.alert_list.delete(*self.alert_list.get_children())
        alerts_list = get_alerts()  # Llamamos a scripts_manager.get_alerts()
        for a in alerts_list:
            vals = (
                a.get("timestamp", ""),
                a.get("src_ip", ""),
                a.get("dst_ip", ""),
                a.get("attack", "")
            )
            self.alert_list.insert("", "end", values=vals)
        self.after(1000, self.update_alerts)

    def update_features(self):
        """Refresca la lista de características cada 1 segundo."""
        self.feature_list.delete(*self.feature_list.get_children())
        feats = get_features()[-10:]  # scripts_manager.get_features()
        for feat in feats:
            vals = (
                feat.get("timestamp", ""),
                feat.get("srcip", ""),
                feat.get("dstip", ""),
                feat.get("protocol_abbreviation", ""),
                feat.get("TotLen Fwd Pkts", 0)
            )
            self.feature_list.insert("", "end", values=vals)
        self.after(1000, self.update_features)

# ---------------------------------------------------------------------------
# FRAME: NIVEL 2 (MoE)
# ---------------------------------------------------------------------------
class Nivel2Frame(ctk.CTkFrame):
    """
    Frame para Nivel 2 (MOE + Detalles de Modelos).
    """
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="#2A2A2A")
        self.controller = controller
        self.data = []  # Para uso cruzado de datos en tablas

        # Opciones de dataset
        self.DATASET_OPTIONS = {
            "NSL-KDD": ("nslkdd", QUEUES_OUTPUT_JSON_NSLKDD),
            "CICIDS 2018": ("cicids", QUEUES_OUTPUT_JSON_CICIDS),
            "UNSW-NB15": ("unsw", QUEUES_OUTPUT_JSON_UNSW),
        }
        self.selected_dataset_id = "nslkdd"
        self.selected_queue_path = QUEUES_OUTPUT_JSON_NSLKDD

        self.pack_propagate(False)
        self.configure(width=800, height=600)

        title_label = ctk.CTkLabel(self, text="IDPS con MoE (Nivel 2)", font=("Arial", 16, "bold"), fg_color="#2A2A2A")
        title_label.pack(pady=10)

        top_controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_controls_frame.pack(pady=5, padx=20, anchor="n")

        btn_risk = ctk.CTkButton(
            top_controls_frame,
            text="Riesgo de Intrusión",
            command=self.show_intrusion_risk,
            fg_color="#1E90FF",
            text_color="white",
            corner_radius=8,
        )
        btn_risk.grid(row=0, column=0, padx=10, sticky="w")

        self.avg_risk_label = ctk.CTkLabel(
            top_controls_frame,
            text="Riesgo Promedio:\n0.00%",
            font=("Arial", 14, "bold"),
            fg_color="#4B4B4B",
            text_color="#FFD700",
            corner_radius=8,
            padx=15,
            pady=8,
        )
        self.avg_risk_label.grid(row=0, column=1, padx=10, sticky="w")

        btn_moe_queues = ctk.CTkButton(
            top_controls_frame,
            text="Ejecutar MoE + Queues",
            command=self.run_moe_and_queues,
            fg_color="#FF8C00",
            text_color="white",
            corner_radius=8,
        )
        btn_moe_queues.grid(row=0, column=2, padx=10, sticky="w")

        # ------- Selector de dataset --------
        ctk.CTkLabel(top_controls_frame, text="Dataset:").grid(row=0, column=3, padx=10, sticky="e")

        self.dataset_menu = ctk.CTkOptionMenu(
            top_controls_frame,
            values=list(self.DATASET_OPTIONS.keys()),
            command=self.on_dataset_change,
        )
        self.dataset_menu.grid(row=0, column=4, padx=5)
        self.dataset_menu.set("NSL-KDD")

        # ---------- Tablas ----------
        tables_frame = ctk.CTkFrame(self, fg_color="transparent")
        tables_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.moe_table_frame = self.create_table(
            parent=tables_frame,
            title="Puntajes del Algoritmo MoE",
            headers=["Ítem", "Puntaje (%)", "Badge"],
            column_widths=[80, 100, 70]
        )
        self.moe_table_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.model_table_frame = self.create_table(
            parent=tables_frame,
            title="Probabilidades por Modelo",
            headers=["Modelo", "Clase 0 (%)", "Clase 1 (%)", "Badge"],
            column_widths=[120, 100, 100, 70]
        )
        self.model_table_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        tables_frame.grid_columnconfigure(0, weight=1)
        tables_frame.grid_columnconfigure(1, weight=1)

    def on_dataset_change(self, choice: str):
        self.selected_dataset_id, self.selected_queue_path = self.DATASET_OPTIONS[choice]
        print(f"[Info Nivel 2] Dataset seleccionado: {choice} → {self.selected_dataset_id}")

    def run_moe_and_queues(self):
        threading.Thread(target=self._run_moe_and_queues_task, daemon=True).start()

    def _run_moe_and_queues_task(self):
        # aqui esta el boton de prueba manual
        # scripts_manager.run_archive_script()
        # scripts_manager.run_preprocess_script()
        scripts_manager.run_calibrated_predict_script()
        scripts_manager.run_automaton_script()
        scripts_manager.run_moe_script()
        scripts_manager.run_queues_script()
        # self.after(0, self.show_intrusion_risk)

    def create_table(self, parent, title, headers, column_widths):
        table_card = ctk.CTkFrame(parent, corner_radius=8, fg_color="#1F1F1F")
        ctk.CTkLabel(table_card, text=title, font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=10, pady=5)

        header_frame = ctk.CTkFrame(table_card, fg_color="transparent")
        header_frame.pack(fill="x")
        for idx, (header, width) in enumerate(zip(headers, column_widths)):
            ctk.CTkLabel(
                header_frame,
                text=header,
                font=("Arial", 11, "bold"),
                text_color="white",
                anchor="center",
                width=width,
            ).grid(row=0, column=idx, padx=5, pady=3)

        table_canvas = ctk.CTkCanvas(table_card, bg="#1F1F1F", highlightthickness=0)
        table_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(table_card, orientation="vertical", command=table_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        table_canvas.configure(yscrollcommand=scrollbar.set)

        rows_container = ctk.CTkFrame(table_canvas, fg_color="#2c2c2c")
        window_id = table_canvas.create_window((0, 0), window=rows_container, anchor="nw")

        def _on_configure_rows(event):
            table_canvas.configure(scrollregion=table_canvas.bbox("all"))
        rows_container.bind("<Configure>", _on_configure_rows)

        def _on_canvas_configure(event):
            table_canvas.itemconfig(window_id, width=event.width)
        table_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            table_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        table_canvas.bind("<MouseWheel>", _on_mousewheel)

        table_card.rows_container = rows_container
        return table_card

    def show_intrusion_risk(self):
        threading.Thread(target=self._show_intrusion_risk_task, daemon=True).start()

    def _show_intrusion_risk_task(self):
        try:
            with open(self.selected_queue_path, "r") as f:
                data = json.load(f)  # dict: {"0": {...}, "1": {...}, ...}
            moe_rows = sorted(data.values(), key=lambda x: x["score"], reverse=True)
            avg_score = sum(row["score"] for row in moe_rows) / len(moe_rows) if moe_rows else 0
            self.after(0, lambda: self._update_intrusion_risk_ui(data, moe_rows, avg_score))
        except FileNotFoundError:
            self.after(0, lambda: print("El archivo seleccionado no fue encontrado."))
        except Exception as e:
            self.after(0, lambda: print(f"Error al procesar los datos: {e}"))

    def _update_intrusion_risk_ui(self, data, moe_rows, avg_score):
        self.data = data
        self.update_moe_table(moe_rows)
        if moe_rows:
            self.update_model_table(moe_rows[0].get("model_contributions", {}))
        self.avg_risk_label.configure(text=f"Riesgo Promedio:\n{avg_score * 100:.2f}%")

    def update_model_table(self, model_contributions):
        container = self.model_table_frame.rows_container
        for widget in container.winfo_children():
            widget.destroy()

        # Orden por la prob. más alta (desc)
        sorted_models = sorted(
            model_contributions.items(),
            key=lambda x: max(x[1].get("probabilities", [0.0, 0.0])),
            reverse=True
        )

        for idx, (model_name, contributions) in enumerate(sorted_models):
            probabilities = contributions.get("probabilities", [0.0, 0.0])
            # Tolerancia: lista/tupla/ndarray a list (asegurar 2 elementos)
            try:
                if hasattr(probabilities, "tolist"):
                    probabilities = probabilities.tolist()
            except Exception:
                pass
            if not isinstance(probabilities, (list, tuple)) or len(probabilities) < 2:
                probabilities = [0.0, 0.0]

            class_0_prob = float(probabilities[0]) * 100.0
            class_1_prob = float(probabilities[1]) * 100.0

            highest_prob = max(class_0_prob, class_1_prob)
            if highest_prob > 90:
                badge_color = "#FFD700"; badge_text = "🥇"
            elif highest_prob > 75:
                badge_color = "#C0C0C0"; badge_text = "🥈"
            elif highest_prob > 50:
                badge_color = "#CD7F32"; badge_text = "🥉"
            else:
                badge_color = None; badge_text = ""

            ctk.CTkLabel(
                container, text=model_name.replace(".pkl", ""),
                text_color="white", anchor="center", width=90
            ).grid(row=idx, column=0, padx=5, pady=2)

            ctk.CTkLabel(
                container, text=f"{class_0_prob:.2f}%",
                text_color="white", anchor="center", width=75
            ).grid(row=idx, column=1, padx=5, pady=2)

            ctk.CTkLabel(
                container, text=f"{class_1_prob:.2f}%",
                text_color="white", anchor="center", width=95
            ).grid(row=idx, column=2, padx=5, pady=2)

            if badge_text:
                ctk.CTkLabel(
                    container, text=badge_text, font=("Arial", 18),
                    width=50, height=30, fg_color=badge_color,
                    corner_radius=15, text_color="#2A2A2A", anchor="center"
                ).grid(row=idx, column=3, padx=10, pady=5)


    def update_moe_table(self, rows, limit=500):
        container = self.moe_table_frame.rows_container
        for widget in container.winfo_children():
            widget.destroy()

        total = len(rows)
        rows = rows[:limit]  # limitar filas para evitar "row out of bounds"

        badges = ["🥇", "🥈", "🥉"]
        for idx, row in enumerate(rows):
            score = row.get("score", 0.0) * 100.0
            badge = badges[idx] if idx < 3 else ""

            ctk.CTkButton(
                container,
                text=f"{idx} [Ver Modelos]",
                text_color="lightgray",
                command=lambda r=row: self.update_model_table(r.get('model_contributions', {})),
                fg_color="#333333",
                hover_color="#444444",
                corner_radius=8,
                anchor="w",
            ).grid(row=idx, column=0, padx=5, pady=2)

            ctk.CTkLabel(container, text=f"{score:.2f}%", text_color="white")\
            .grid(row=idx, column=1, padx=5, pady=2)

            if badge:
                ctk.CTkLabel(
                    container,
                    text=badge, font=("Arial", 18), width=50, height=30,
                    fg_color="#FFD700" if idx == 0 else ("#C0C0C0" if idx == 1 else "#CD7F32"),
                    corner_radius=15, text_color="#2A2A2A",
                ).grid(row=idx, column=2, padx=10, pady=5)

        if total > limit:
            ctk.CTkLabel(
                container,
                text=f"Mostrando {limit} de {total} items",
                text_color="gray"
            ).grid(row=limit, column=0, columnspan=3, padx=5, pady=(8, 4), sticky="w")

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FRAME: NIVEL 3
# ---------------------------------------------------------------------------
class Nivel3Frame(ctk.CTkFrame):
    """
    Frame para el Nivel 3 (NLP - GPT-2 + QA).
    """
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="#2A2A2A")
        self.controller = controller
        
        # Referencias a objetos NLP
        self.nlp_module = None
        self.gpt2_generator = None
        self.qa_pipeline = None
        self.dataset = None
        self.json_data = None
        self.plot_mode = None  # Estado para manejar la selección de gráficos

        # Título principal
        title = ctk.CTkLabel(
            self,
            text="NLP - Spanish GPT-2 (Nivel 3)",
            font=("Arial", 16, "bold"),
            fg_color="#2A2A2A"
        )
        title.pack(pady=5)

        # Tarjeta principal
        nlp_card_main = ctk.CTkFrame(self, corner_radius=8)
        nlp_card_main.pack(pady=10, padx=10, fill="x")

        # Sub-tarjeta (chat)
        chat_frame = ctk.CTkFrame(nlp_card_main, corner_radius=8)
        chat_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # Área de visualización del chat
        self.chat_display = ctk.CTkTextbox(chat_frame, width=450, height=220)
        self.chat_display.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.chat_display.insert("end", "Bienvenido al ChatSolver!\n")
        self.chat_display.configure(state="disabled")

        # Etiqueta "Mensaje"
        ctk.CTkLabel(chat_frame, text="Mensaje:", font=("Arial", 12)).grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )

        # Caja de texto para escribir el mensaje
        self.nlp_input_box = ctk.CTkEntry(chat_frame, width=260, placeholder_text="Escribe tu mensaje aquí...")
        self.nlp_input_box.grid(row=1, column=1, padx=5, pady=5)

        # Al presionar Enter en la caja de texto, se gestiona según el contexto
        self.nlp_input_box.bind("<Return>", self.on_enter_send_message)

        # Botón "Enviar"
        nlp_send_button = ctk.CTkButton(chat_frame, text="Enviar", command=self.send_nlp_message)
        nlp_send_button.grid(row=1, column=2, padx=5, pady=5)

        # Botón "Limpiar"
        clear_button = ctk.CTkButton(chat_frame, text="Limpiar", fg_color="red", command=self.clear_chat)
        clear_button.grid(row=2, column=2, padx=5, pady=(0, 10), sticky="e")

        # Frame adicional a la derecha del chat (para botones de carga y demo)
        nlp_right_frame = ctk.CTkFrame(nlp_card_main, corner_radius=0)
        nlp_right_frame.grid(row=2, column=1, sticky="nw", padx=10, pady=(0, 10))

        # Botón para Cargar GPT-2
        btn_load_gpt2 = ctk.CTkButton(nlp_right_frame, text="Cargar GPT-2", command=self.load_gpt2)
        btn_load_gpt2.grid(row=0, column=0, padx=(0, 5), pady=5)

        # Botón para iniciar la sesión NLP/QA
        btn_nlp_demo = ctk.CTkButton(nlp_right_frame, text="Iniciar Chatbot NLP", command=self.start_nlp_session)
        btn_nlp_demo.grid(row=1, column=0, padx=(0, 5), pady=5)

        # Botón para ver evaluaciones (se delega la función a scripts_manager)
        btn_eval = ctk.CTkButton(nlp_right_frame, text="Ver Evaluaciones", command=self.show_evaluations)
        btn_eval.grid(row=2, column=0, padx=(0, 5), pady=5)

        btn_stop_task = ctk.CTkButton(nlp_right_frame, text="Detener", command=self.stop_task)
        btn_stop_task.grid(row=4, column=0, padx=(0, 5), pady=5)

    # ----------------------------------------------------------------------
    # Métodos de Chat (solo para manejo de interfaz)
    # ----------------------------------------------------------------------
    def append_chat(self, message):
        """Agrega un mensaje al área de visualización del chat."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{message}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def clear_chat(self):
        """Limpia el área de visualización del chat y la entrada."""
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.nlp_input_box.delete(0, "end")

    # ----------------------------------------------------------------------
    # Integración con scripts_manager y módulo NLP
    # ----------------------------------------------------------------------
    def on_enter_send_message(self, event):
        """
        Método que se llama cuando se presiona Enter en self.nlp_input_box.
        Llama a send_nlp_message() para enviar el mensaje.
        """
        self.send_nlp_message()

    def load_gpt2(self):
        """
        Carga el módulo NLP vía scripts_manager por si no se cargo,
        luego llama a nlp.load_spanish_gpt2 para cargar GPT-2.
        """
        self.append_chat("Intentando cargar el modelo Spanish GPT-2...")

        def task():
            try:
                if not self.nlp_module:
                    self.append_chat("[Info] Cargando módulo NLP desde scripts_manager...")
                    self.nlp_module = scripts_manager.load_nlp_module(callback=self.append_chat)
                    if not self.nlp_module:
                        self.append_chat("[Error] No se pudo cargar el módulo NLP.")
                        return

                load_gpt2_func = self.nlp_module.load_spanish_gpt2
                self.gpt2_generator = load_gpt2_func(callback=self.append_chat)

                if self.gpt2_generator:
                    self.append_chat("El modelo Spanish GPT-2 se cargó exitosamente.")
                else:
                    self.append_chat("[Error] No se pudo cargar el modelo Spanish GPT-2.")
            except Exception as e:
                self.append_chat(f"[Error] Ocurrió un problema al cargar el modelo GPT-2: {e}")

        threading.Thread(target=task, daemon=True).start()

    def start_nlp_session(self):
        """
        Inicia la sesión NLP (pipeline QA, dataset, etc.)
        llamando a nlp.main(user_input="1").
        """
        self.append_chat("Iniciando el modo NLP...")

        def task():
            try:
                if not self.nlp_module:
                    self.append_chat("[Info] Cargando módulo NLP primero...")
                    self.nlp_module = scripts_manager.load_nlp_module(callback=self.append_chat)
                    if not self.nlp_module:
                        self.append_chat("[Error] No se pudo cargar el módulo NLP.")
                        return

                main_func = self.nlp_module.main
                result = main_func(user_input="1", callback=self.append_chat)

                if result and len(result) == 4:
                    self.qa_pipeline, self.text_generator, self.dataset, self.json_data = result
                    self.append_chat("El chatbot está listo para recibir mensajes.")
                else:
                    self.append_chat("[Error] No se pudo iniciar el chatbot.")
            except Exception as e:
                self.append_chat(f"[Error] Ocurrió un problema al iniciar el chatbot: {e}")

        threading.Thread(target=task, daemon=True).start()

    def send_nlp_message(self):
        """
        Envía un mensaje a nlp.py para procesarlo el usará self.qa_pipeline, self.text_generator, etc.
        Se llama tanto al hacer clic en el botón 'Enviar' como al presionar Enter.
        """
        if not self.qa_pipeline or not self.text_generator:
            self.append_chat("[Error] El chatbot no está inicializado. Primero inicia una sesión NLP.")
            return

        user_input = self.nlp_input_box.get().strip()
        if not user_input:
            return

        self.append_chat(f"Usuario: {user_input}")
        self.nlp_input_box.delete(0, "end")

        def task():
            try:
                process_user_input = self.nlp_module.process_user_input
                is_running = process_user_input(
                    user_input,
                    self.qa_pipeline,
                    self.text_generator,
                    self.dataset,
                    self.json_data,
                    callback=self.append_chat
                )
                if not is_running:
                    self.append_chat("El chatbot ha finalizado la sesión.")
            except Exception as e:
                self.append_chat(f"[Error] No se pudo procesar el mensaje: {e}")

        threading.Thread(target=task, daemon=True).start()

    def show_evaluations(self):
        """
        Muestra el contenido completo del JSON de evaluaciones utilizando
        la función show_json_content_in_interface de scripts_manager.
        """
        content = scripts_manager.show_json_content_in_interface(callback=self.append_chat)
        if content is None:
            self.append_chat("[Error] No se pudo mostrar el contenido del JSON.")

    def stop_task(self):
        """
        Detiene la ejecución del programa NLP enviando la opción de salida ("3").
        nlp.main(...) maneja user_input="3" para salir, pero no sirve bien.
        """
        self.append_chat("Deteniendo el programa NLP...")

        def task():
            try:
                if not self.nlp_module:
                    self.append_chat("[Error] No se ha cargado el módulo NLP.")
                    return
                main_func = self.nlp_module.main
                result = main_func(user_input="3", callback=self.append_chat)
                if result is None:
                    self.append_chat("El programa NLP se ha detenido correctamente.")
                else:
                    self.append_chat("[Error] No se pudo detener el programa NLP.")
            except Exception as e:
                self.append_chat(f"[Error] Ocurrió un problema al detener NLP: {e}")

        threading.Thread(target=task, daemon=True).start()

# ---------------------------------------------------------------------------
# FRAME 4 : TERMINAL
# ---------------------------------------------------------------------------

class TerminalFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        """
        Frame card que simula una terminal para imprimir stdout/stderr.
        """
        super().__init__(parent, fg_color="#2A2A2A", width=800, height=300, corner_radius=15)
        self.controller = controller
        # Evita que el frame se redimensione automáticamente por el contenido
        self.grid_propagate(False)

        # --- Header: bolitas de colores + título "Terminal" ---
        header_frame = ctk.CTkFrame(self, fg_color="#2A2A2A")
        header_frame.pack(fill="x", padx=10, pady=(10, 0))

        # Frame para las bolitas de colores (alineadas a la izquierda)
        circles_frame = ctk.CTkFrame(header_frame, fg_color="#2A2A2A")
        circles_frame.pack(side="left", padx=0, pady=0)

        # Usamos tk.Canvas para cada bolita
        red_circle = tk.Canvas(circles_frame, width=12, height=12, bg="#2A2A2A", highlightthickness=0)
        red_circle.create_oval(0, 0, 12, 12, fill="#FF5F57", outline="")
        red_circle.pack(side="left", padx=2)

        yellow_circle = tk.Canvas(circles_frame, width=12, height=12, bg="#2A2A2A", highlightthickness=0)
        yellow_circle.create_oval(0, 0, 12, 12, fill="#FFBD2E", outline="")
        yellow_circle.pack(side="left", padx=2)

        green_circle = tk.Canvas(circles_frame, width=12, height=12, bg="#2A2A2A", highlightthickness=0)
        green_circle.create_oval(0, 0, 12, 12, fill="#28C840", outline="")
        green_circle.pack(side="left", padx=2)

        # Etiqueta "Terminal" a la derecha de las bolitas
        title_label = ctk.CTkLabel(
            header_frame,
            text="Terminal",
            font=("Arial", 16, "bold"),
            fg_color="#2A2A2A",
            text_color="white"
        )
        title_label.pack(side="left", padx=10)

        # --- Separador horizontal (simulado con un Frame de 2px) ---
        separator = ctk.CTkFrame(self, fg_color="#3A3A3A", height=2)
        separator.pack(fill="x", padx=10, pady=(5, 5))

        # --- Área de texto que simula la terminal ---
        self.terminal_text = ctk.CTkTextbox(self, width=780, height=200, fg_color="#3A3A3A")
        self.terminal_text.pack(padx=10, pady=5, fill="both", expand=False)
        self.terminal_text.configure(state="normal")

        # Configurar “tags” para colorear texto
        self.terminal_text._textbox.tag_configure("error", foreground="red")
        self.terminal_text._textbox.tag_configure("info", foreground="lime")
        self.terminal_text._textbox.tag_configure("script1", foreground="cyan")
        self.terminal_text._textbox.tag_configure("default", foreground="white")
        self.terminal_text.configure(state="disabled")

        # --- Botón para limpiar la terminal ---
        clear_button = ctk.CTkButton(self, text="Limpiar Terminal", command=self.clear_terminal)
        clear_button.pack(pady=(5, 10))

        # --- Redirigir stdout y stderr a este widget ---
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TerminalRedirector(self.terminal_text, self.original_stdout)
        sys.stderr = TerminalRedirector(self.terminal_text, self.original_stderr)

    def clear_terminal(self):
        """Limpia el contenido del área de texto de la terminal."""
        self.terminal_text.configure(state="normal")
        self.terminal_text._textbox.delete("1.0", "end")
        self.terminal_text.configure(state="disabled")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
