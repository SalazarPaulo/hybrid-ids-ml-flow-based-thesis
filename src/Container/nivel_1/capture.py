# -*- coding: utf-8 -*-
"""
IDS Flow-based  soporte para los datasets:
- NSL-KDD
- UNSW-NB15
- CICIDS2018

Mantiene las funciones y campos  para NSL-KDD y UNSW-NB15,
y añade las columnas de CICIDS2018. Se generan 4 secciones en el JSON final:
"alerts", "nslkdd", "unsw15", "cicids2018".

Incluye docstrings y comentarios, y secciones claramente diferenciadas.
"""

import os
import json
import time
import threading
import logging
from collections import defaultdict, deque
from utilities.global_paths import DETECTIONS_JSON, FINAL_DATA_JSON
from scapy.all import sniff, IP, TCP, UDP, Raw, ICMP, DNS

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ids_flow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("flow_ids")

# ============================================================================
# CONFIGURACIONES Y PARÁMETROS GLOBALES
# ============================================================================

# Parámetros de DDoS y ventanas de tiempo
THRESHOLD = 20
TIME_WINDOW = 3
COOLDOWN = 30

# Parámetros de timeout y limpieza
FLOW_TIMEOUT = 120
FRAGMENT_TIMEOUT = 30
CLEANUP_INTERVAL = 60

# Máx. de flujos en memoria
MAX_FLOWS = 10000

# ============================================================================
# LISTAS DE CARACTERÍSTICAS (FEATURES) POR DATASET
# ============================================================================
# NSL-KDD 
FIELD_NAMES_NSLKDD = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "wrong_fragment", "hot",
    "logged_in", "num_compromised", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "attack"
]

# UNSW-NB15 
FIELD_NAMES_UNSW15 = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
    "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    "attack"
]

# CICIDS2018
FIELD_NAMES_CIC2018 = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd URG Flags",
    "Bwd Pkts/s", "Pkt Len Var",
    "FIN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "Down/Up Ratio", "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Seg Size Min",
    "Active Mean", "Active Std",
    "Idle Min",
    "Label",      # Ejemplo: "FTP-BruteForce"
    "Threat",     # 0/1
    "Attack Type" # Ej. "Brute-force"
]

# Unificamos en una sola para el procesamiento interno
ALL_FIELDS = sorted(set(
    FIELD_NAMES_NSLKDD +
    FIELD_NAMES_UNSW15 +
    FIELD_NAMES_CIC2018
))

# ============================================================================
# MAPEOS DE PROTOCOLOS Y SERVICIOS
# ============================================================================
# PROTOCOL_MAP: asocia números de protocolo (ICMP=1, TCP=6, etc.) a strings
PROTOCOL_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    41: "IPv6",
    47: "GRE",
    50: "ESP",
    51: "AH",
    58: "ICMPv6",
    89: "OSPF",
}

# SERVICE_MAP: mapea un nombre de servicio (ej. "HTTP") a un código numérico
SERVICE_MAP = {
    "HTTP": 1,
    "HTTPS": 2,
    "FTP": 3,
    "SSH": 4,
    "DNS": 5,
    "TELNET": 6,
    "SMTP": 7,
    "POP3": 8,
    "IMAP": 9,
    "SMB": 10,
    "DHCP": 11,
    "NTP": 12,
    "SNMP": 13,
    "RDP": 14,
    "LDAP": 15,
    "UNKNOWN": 0
}

# PORT_SERVICE_MAP: mapeos típicos de puertos a nombres de servicio (ej. 21->"FTP")
PORT_SERVICE_MAP = {
    21: "FTP",
    22: "SSH",
    23: "TELNET",
    25: "SMTP",
    53: "DNS",
    67: "DHCP",
    68: "DHCP",
    69: "TFTP",
    80: "HTTP",
    110: "POP3",
    123: "NTP",
    143: "IMAP",
    161: "SNMP",
    162: "SNMP",
    389: "LDAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP"
}

# Diccionario de flags TCP
TCP_FLAGS_MAP = {
    "F": 0x01,  # FIN
    "S": 0x02,  # SYN
    "R": 0x04,  # RST
    "P": 0x08,  # PSH
    "A": 0x10,  # ACK
    "U": 0x20,  # URG
    "E": 0x40,  # ECE
    "C": 0x80   # CWR
}

# ============================================================================
# ESTRUCTURAS GLOBALES DE DATOS
# ============================================================================
alerts_list = deque(maxlen=1000)

# IMPORTANTE: en lugar de flows_closed, ahora usaremos 3 listas
# para poder exportar cada dataset por separado en el JSON final.
nslkdd_rows = []
unsw15_rows = []
cicids2018_rows = []

extracted_features = deque(maxlen=10000)

connection_tracker = defaultdict(lambda: deque(maxlen=1000))
ddos_alerted = {}

host_connection_stats = defaultdict(lambda: {
    "connections": 0,
    "last_seen": time.time(),
    "ports_used": set(),
    "destinations": set()
})

flow_state = {}

# Contadores globales para ct_* (UNSW)
srv_src_counter = defaultdict(int)
state_ttl_counter = defaultdict(int)
dst_ltm_counter = defaultdict(int)
src_dport_counter = defaultdict(int)
dst_sport_counter = defaultdict(int)
dst_src_counter = defaultdict(int)
src_ltm_counter = defaultdict(int)
srv_dst_counter = defaultdict(int)

# Fragmentos IP
fragment_buffer = defaultdict(lambda: {
    "fragments": [],
    "last_updated": time.time(),
    "done": False,
    "total_length": 0,
    "wrong_fragment": False
})

sniffing_active = False
sniffing_thread = None
cleanup_thread = None

# ============================================================================
# FUNCIONES DE INICIALIZACIÓN DE FLOW Y AUXILIARES
# ============================================================================
def init_flow_data():
    """ funcion:
    Crea un diccionario que inicializa en 0 todas las columnas de todos los
    datasets (NSL-KDD, UNSW-NB15 y CICIDS2018).
    Retorna el diccionario para usarse en flow_state[flow_id].
    """
    data = {
        **{col: 0 for col in ALL_FIELDS},

        # Tiempos
        "start_time": time.time(),
        "end_time": time.time(),
        "last_activity": time.time(),

        # Timestamps (src, dst y global para Flow IAT)
        "src_timestamps": deque(maxlen=100),
        "dst_timestamps": deque(maxlen=100),
        "all_timestamps": deque(maxlen=200),

        # Historial TCP (sec/ack)
        "src_seq_history": deque(maxlen=10),
        "dst_seq_history": deque(maxlen=10),
        "src_ack_history": deque(maxlen=10),
        "dst_ack_history": deque(maxlen=10),

        # Handshake times
        "syn_time": None,
        "synack_time": None,
        "ack_time": None,

        # Flags
        "tcp_flags_seen": set(),

        # Contadores de error (NSL-KDD)
        "serror_count": 0,
        "srv_error_count": 0,
        "rerror_count": 0,

        # HTTP
        "http_methods": set(),
        "http_requests_count": 0,
        "http_responses_count": 0,
        "http_response_body_len": 0,

        # Payload
        "payload_history": [],

        # TTL
        "src_ttls": [],
        "dst_ttls": [],

        # --- Campos auxiliares CICIDS2018 ---
        "fwd_pkt_len_list": [],
        "bwd_pkt_len_list": [],
        "all_pkt_len_list": [],
        "fwd_psh_flags_cnt": 0,
        "fwd_urg_flags_cnt": 0,
        "fin_flag_cnt": 0,
        "rst_flag_cnt": 0,
        "psh_flag_cnt": 0,
        "ack_flag_cnt": 0,
        "urg_flag_cnt": 0,
        "init_fwd_win_bytes": None,
        "init_bwd_win_bytes": None,
        "fwd_seg_size_min": None,

        # Etiquetas (CICIDS2018)
        "Label": "Benign",
        "Threat": 0,
        "Attack Type": "None"
    }
    return data

def get_flow_id(packet):
    """funcion:
    Identifica de forma única un flow basado en src_ip, dst_ip, puertos
    y protocolo. Soporta TCP, UDP e ICMP.
    Retorna una tupla (src_ip, dst_ip, sport, dport, proto).
    """
    if IP not in packet:
        return None
    ip_layer = packet[IP]
    proto = ip_layer.proto
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst

    # TCP
    if proto == 6 and TCP in packet:
        tcp_layer = packet[TCP]
        return (src_ip, dst_ip, tcp_layer.sport, tcp_layer.dport, proto)
    # UDP
    elif proto == 17 and UDP in packet:
        return (src_ip, dst_ip, packet[UDP].sport, packet[UDP].dport, proto)
    # ICMP
    elif proto == 1 and ICMP in packet:
        icmp_layer = packet[ICMP]
        return (src_ip, dst_ip, icmp_layer.type, icmp_layer.code, proto)
    else:
        # Otros protocolos
        return (src_ip, dst_ip, 0, 0, proto)

def identify_service(packet, proto, sport, dport):
    """ funcion:
    Heurística para identificar servicio (NSL-KDD) basándose en
    puerto, protocolo y posibles strings en la capa Raw.
    """
    if sport in PORT_SERVICE_MAP:
        return SERVICE_MAP.get(PORT_SERVICE_MAP[sport], 0)
    if dport in PORT_SERVICE_MAP:
        return SERVICE_MAP.get(PORT_SERVICE_MAP[dport], 0)

    # Regla simple si es TCP con payload
    if proto == 6 and Raw in packet:
        payload = packet[Raw].load
        try:
            payload_str = payload.decode('utf-8', errors='ignore').lower()
        except:
            payload_str = str(payload).lower()
        if b'http' in payload or b'host:' in payload:
            return SERVICE_MAP["HTTP"]
        if b'ftp' in payload.lower() or b'220 ftp' in payload.lower():
            return SERVICE_MAP["FTP"]
        if b'ssh-' in payload:
            return SERVICE_MAP["SSH"]
        if b'mail from:' in payload or b'rcpt to:' in payload:
            return SERVICE_MAP["SMTP"]
    elif proto == 17:
        if DNS in packet:
            return SERVICE_MAP["DNS"]
        if dport in [67, 68] or sport in [67, 68]:
            return SERVICE_MAP["DHCP"]
        if dport == 123 or sport == 123:
            return SERVICE_MAP["NTP"]
        if dport in [161,162] or sport in [161,162]:
            return SERVICE_MAP["SNMP"]
    return 0

# ============================================================================
# MANEJO DE FRAGMENTOS IP
# ============================================================================
def process_fragment(packet):
    """
    funcion:
    Maneja el caso de paquetes IP fragmentados:
    - Almacena los fragmentos en fragment_buffer.
    - Intenta reensamblar en reassemble_fragments.
    - Luego limpia fragmentos caducados con cleanup_fragments.
    """
    ip_layer = packet[IP]
    key = (ip_layer.src, ip_layer.dst, ip_layer.id, ip_layer.proto)
    offset = ip_layer.frag * 8
    mf_flag = bool(ip_layer.flags & 0x1)
    payload = b""

    if Raw in packet:
        payload = bytes(packet[Raw].load)

    fragment_buffer[key]["last_updated"] = time.time()
    fragment_buffer[key]["fragments"].append({
        "offset": offset,
        "mf": mf_flag,
        "data": payload,
        "timestamp": time.time()
    })
    reassemble_fragments(key)
    cleanup_fragments()

def reassemble_fragments(key):
    """
    funcion:
    Ordena los fragmentos por offset y verifica solapamientos o huecos.
    Si logra reconstruir todo, marca 'done' en fragment_buffer[key].
    """
    fb = fragment_buffer[key]
    if fb["done"] or fb["wrong_fragment"]:
        return
    frags = sorted(fb["fragments"], key=lambda x: x["offset"])
    reassembled_data = b""
    expected_offset = 0
    last_fragment_seen = False

    for frag in frags:
        offset = frag["offset"]
        data = frag["data"]
        if offset > expected_offset:
            # hueco => no se puede reensamblar aún
            return
        if offset < expected_offset:
            overlap_sz = expected_offset - offset
            overlapped_data = data[:overlap_sz]
            existing_data = reassembled_data[-overlap_sz:] if overlap_sz <= len(reassembled_data) else None
            if existing_data is not None and overlapped_data != existing_data:
                fb["wrong_fragment"] = True
                logger.warning(f"Fragmento inconsistente: {key}")
                return
            data = data[overlap_sz:]
            offset = expected_offset
        reassembled_data += data
        expected_offset = offset + len(data)
        if not frag["mf"]:
            last_fragment_seen = True

    if last_fragment_seen:
        fb["done"] = True
        fb["total_length"] = len(reassembled_data)
        logger.debug(f"Reensamblado completo: {key}, len={fb['total_length']}")

def cleanup_fragments():
    """
    funcion:
    Elimina fragmentos expirados o marcados 'done' de fragment_buffer
    basándose en FRAGMENT_TIMEOUT.
    """
    now = time.time()
    to_remove = []
    for key, data in fragment_buffer.items():
        if (now - data["last_updated"] > FRAGMENT_TIMEOUT) or data["done"]:
            to_remove.append(key)
    for k in to_remove:
        del fragment_buffer[k]

# ============================================================================
# FUNCIÓN update_flow_state (Procesa cada paquete en su flow)
# ============================================================================
def update_flow_state(flow_id, packet):
    """
    funcion:
    Actualiza el estado de un flujo (flow_id) con la información del
    paquete actual, para NSL-KDD, UNSW-NB15, CICIDS2018.

    - Calcula bytes, timestamps, contadores de flags, etc.
    - Maneja contadores globales ct_* (UNSW).
    - Prepara datos para Fwd/Bwd packet len (CICIDS).
    """
    # Si no existe el flow, crearlo
    if flow_id not in flow_state:
        flow_state[flow_id] = init_flow_data()
        logger.debug(f"[INFO] Creando flow: {flow_id}")

        # Incrementar contadores globales
        src_ip, dst_ip, sport, dport, proto = flow_id
        host_connection_stats[src_ip]["connections"] += 1
        host_connection_stats[src_ip]["last_seen"] = time.time()
        host_connection_stats[src_ip]["ports_used"].add(sport)
        host_connection_stats[src_ip]["destinations"].add(dst_ip)

        # UNSW: ct_srv_src, etc.
        service = identify_service(packet, proto, sport, dport)
        srv_key = (src_ip, service)
        srv_src_counter[srv_key] += 1
        dst_ltm_counter[dst_ip] += 1
        src_dport_key = (src_ip, dport)
        src_dport_counter[src_dport_key] += 1
        dst_sport_key = (dst_ip, sport)
        dst_sport_counter[dst_sport_key] += 1
        dst_src_key = (dst_ip, src_ip)
        dst_src_counter[dst_src_key] += 1
        src_ltm_counter[src_ip] += 1
        srv_dst_key = (service, dst_ip)
        srv_dst_counter[srv_dst_key] += 1

    fdata = flow_state[flow_id]
    fdata["last_activity"] = time.time()
    fdata["end_time"] = time.time()
    dur = fdata["end_time"] - fdata["start_time"]
    fdata["duration"] = dur
    fdata["dur"] = dur

    ip_layer = packet[IP]
    pkt_len = len(packet)

    src_ip, dst_ip, sport, dport, proto2 = flow_id
    is_src = (ip_layer.src == src_ip)

    # Para CICIDS2018: timestamps globales de todos los pkts
    now_ts = time.time()
    fdata["all_timestamps"].append(now_ts)
    fdata["all_pkt_len_list"].append(pkt_len)

    # Direcciones forward/backward
    if is_src:
        fdata["src_bytes"] += pkt_len
        fdata["spkts"] += 1
        fdata["sbytes"] += pkt_len
        fdata["src_timestamps"].append(now_ts)
        fdata["src_ttls"].append(ip_layer.ttl)

        # CICIDS: fwd pkt lens
        fdata["fwd_pkt_len_list"].append(pkt_len)

    else:
        fdata["dst_bytes"] += pkt_len
        fdata["dpkts"] += 1
        fdata["dbytes"] += pkt_len
        fdata["dst_timestamps"].append(now_ts)
        fdata["dst_ttls"].append(ip_layer.ttl)

        # CICIDS: bwd pkt lens
        fdata["bwd_pkt_len_list"].append(pkt_len)

    fdata["protocol_type"] = float(ip_layer.proto)
    fdata["service"] = identify_service(packet, ip_layer.proto, sport, dport)

    # Verifica fragmento incorrecto
    key_frag = (ip_layer.src, ip_layer.dst, ip_layer.id, ip_layer.proto)
    if key_frag in fragment_buffer and fragment_buffer[key_frag]["wrong_fragment"]:
        fdata["wrong_fragment"] = 1

    # -------------------------------------------------------------------------
    # Si es TCP, procesar flags, handshake, etc.
    # -------------------------------------------------------------------------
    if ip_layer.proto == 6 and TCP in packet:
        tcp_layer = packet[TCP]
        flags_str = str(tcp_layer.flags)

        # RST => error
        if "R" in flags_str:
            # Ejemplo: serror_count si RST desde lado "server" (not is_src)
            if not is_src:
                fdata["serror_count"] += 1
            fdata["rerror_count"] += 1

        # flag valor
        flag_val = 0
        for ff, val in TCP_FLAGS_MAP.items():
            if ff in flags_str:
                fdata["tcp_flags_seen"].add(ff)
                flag_val |= val
        fdata["flag"] = float(flag_val)

        # CICIDS: contadores de flags
        if "F" in flags_str:
            fdata["fin_flag_cnt"] += 1
        if "R" in flags_str:
            fdata["rst_flag_cnt"] += 1
        if "P" in flags_str:
            fdata["psh_flag_cnt"] += 1
            if is_src:
                fdata["fwd_psh_flags_cnt"] += 1
        if "A" in flags_str:
            fdata["ack_flag_cnt"] += 1
        if "U" in flags_str:
            fdata["urg_flag_cnt"] += 1
            if is_src:
                fdata["fwd_urg_flags_cnt"] += 1

        # Ventanas init forward/backward
        if is_src:
            fdata["sttl"] = ip_layer.ttl
            fdata["stcpb"] = tcp_layer.seq
            fdata["swin"] = tcp_layer.window
            if fdata["init_fwd_win_bytes"] is None:
                fdata["init_fwd_win_bytes"] = tcp_layer.window
            if fdata["fwd_seg_size_min"] is None or (pkt_len < fdata["fwd_seg_size_min"]):
                fdata["fwd_seg_size_min"] = pkt_len
        else:
            fdata["dttl"] = ip_layer.ttl
            fdata["dtcpb"] = tcp_layer.seq
            fdata["dwin"] = tcp_layer.window
            if fdata["init_bwd_win_bytes"] is None:
                fdata["init_bwd_win_bytes"] = tcp_layer.window

        # Handshake times
        if "S" in flags_str and "A" not in flags_str and not fdata["syn_time"]:
            fdata["syn_time"] = time.time()
        elif "S" in flags_str and "A" in flags_str and not fdata["synack_time"]:
            fdata["synack_time"] = time.time()
        elif "A" in flags_str and fdata["synack_time"] and not fdata["ack_time"]:
            fdata["ack_time"] = time.time()

        # Historial seq/ack
        if is_src:
            fdata["src_seq_history"].append(tcp_layer.seq)
            if tcp_layer.ack:
                fdata["src_ack_history"].append(tcp_layer.ack)
        else:
            fdata["dst_seq_history"].append(tcp_layer.seq)
            if tcp_layer.ack:
                fdata["dst_ack_history"].append(tcp_layer.ack)

        # Procesar posible HTTP
        if Raw in packet and (tcp_layer.sport == 80 or tcp_layer.dport == 80):
            process_http_data(packet, flow_id, fdata, is_src)

    # -------------------------------------------------------------------------
    # UDP
    # -------------------------------------------------------------------------
    elif ip_layer.proto == 17 and UDP in packet:
        fdata["flag"] = 0

    # -------------------------------------------------------------------------
    # ICMP
    # -------------------------------------------------------------------------
    elif ip_layer.proto == 1 and ICMP in packet:
        icmp_layer = packet[ICMP]
        val = (icmp_layer.type << 8) | icmp_layer.code
        fdata["flag"] = float(val)
    else:
        fdata["flag"] = 0

    # Rate (UNSW)
    if dur > 0:
        total_b = fdata["sbytes"] + fdata["dbytes"]
        fdata["rate"] = total_b / dur
    else:
        fdata["rate"] = 0

    # Payload checks (NSL-KDD)
    if Raw in packet:
        payload = packet[Raw].load
        try:
            p_str = payload.decode('utf-8', errors='ignore').lower()
        except:
            p_str = str(payload).lower()

        if len(fdata["payload_history"]) < 5:
            fdata["payload_history"].append(payload[:100])

        # NSL-KDD
        if "login" in p_str:
            fdata["hot"] += 1
        if "auth_ok" in p_str:
            fdata["logged_in"] = 1
        if "exploit" in p_str:
            fdata["num_compromised"] += 1
        if "malicious" in p_str:
            fdata["attack"] = 1

        # UNSW (FTP)
        if "ftp" in p_str:
            fdata["is_ftp_login"] = 1
            if "user" in p_str:
                fdata["ct_ftp_cmd"] += 1
            if "pass" in p_str:
                fdata["ct_ftp_cmd"] += 1

    # contadores nsl-kdd
    fdata["count"] += 1
    fdata["srv_count"] += 1

    # ct_* (UNSW)
    s_ip, d_ip, s_p, d_p, pr = flow_id
    srv_key = (s_ip, fdata["service"])
    fdata["ct_srv_src"] = srv_src_counter[srv_key]

    ttl_key = (s_ip, ip_layer.ttl)
    fdata["ct_state_ttl"] = state_ttl_counter[ttl_key]

    fdata["ct_dst_ltm"] = dst_ltm_counter[d_ip]

    src_dport_key = (s_ip, d_p)
    fdata["ct_src_dport_ltm"] = src_dport_counter[src_dport_key]

    dst_sport_key = (d_ip, s_p)
    fdata["ct_dst_sport_ltm"] = dst_sport_counter[dst_sport_key]

    dst_src_key = (d_ip, s_ip)
    fdata["ct_dst_src_ltm"] = dst_src_counter[dst_src_key]

    fdata["ct_src_ltm"] = src_ltm_counter[s_ip]

    srv_dst_key = (fdata["service"], d_ip)
    fdata["ct_srv_dst"] = srv_dst_counter[srv_dst_key]

    if (s_ip == d_ip) and (s_p == d_p):
        fdata["is_sm_ips_ports"] = 1

# ============================================================================
# PROCESAR DATOS HTTP (EJEMPLO BÁSICO, AUN FALTA)
# ============================================================================
def process_http_data(packet, flow_id, fdata, is_src_to_dst):
    """
    funcion:
    Intenta detectar métodos HTTP y contabilizar 'Content-Length'
    para trans_depth y response_body_len.
    """
    tcp_layer = packet[TCP]
    raw_data = packet[Raw].load
    try:
        data_str = raw_data.decode('utf-8', errors='ignore')
    except:
        data_str = str(raw_data)

    lines = data_str.split('\r\n')
    if lines:
        first_line = lines[0]
        if first_line.startswith('GET '):
            fdata["http_methods"].add('GET')
            fdata["ct_flw_http_mthd"] += 1
        elif first_line.startswith('POST '):
            fdata["http_methods"].add('POST')
            fdata["ct_flw_http_mthd"] += 1
        elif first_line.startswith('PUT '):
            fdata["http_methods"].add('PUT')
            fdata["ct_flw_http_mthd"] += 1

    if is_src_to_dst and tcp_layer.dport == 80:
        fdata["http_requests_count"] += 1
        fdata["trans_depth"] += 1
    elif (not is_src_to_dst) and tcp_layer.sport == 80:
        fdata["http_responses_count"] += 1
        for line in lines:
            if 'Content-Length:' in line:
                try:
                    val = int(line.split(':')[1].strip())
                    fdata["response_body_len"] = val
                except:
                    pass

# ============================================================================
# FUNCIÓN DE CÁLCULO DE INTERVALOS Y JITTER
# ============================================================================
def compute_intervals_and_jitter(timestamps):
    """
    funcion:
    Retorna (mean, jitter) para la lista de timestamps.
    - mean => promedio de intervalos consecutivos
    - jitter => desviación absoluta promedio
    """
    if len(timestamps) < 2:
        return 0.0, 0.0
    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i - 1])
    mean_val = sum(intervals) / len(intervals)
    diffs = [abs(x - mean_val) for x in intervals]
    jitter = sum(diffs) / len(diffs)
    return mean_val, jitter

# ============================================================================
# FUNCIÓN close_flow (cierra y separa en 3 filas: nslkdd, unsw15, cicids2018)
# ============================================================================
def close_flow(flow_id):
    """
    funcion:
    Cierra el flujo, calcula las métricas finales de NSL-KDD, UNSW-NB15
    y CICIDS2018. Luego separa en 3 filas (diccionarios) para guardarlas
    en las listas globales correspondientes (nslkdd_rows, unsw15_rows,
    cicids2018_rows).
    """
    if flow_id not in flow_state:
        return

    fdata = flow_state.pop(flow_id)

    # ----------------------------------
    # Cálculos finales (NSL-KDD)
    # ----------------------------------
    dur = fdata["end_time"] - fdata["start_time"]
    fdata["duration"] = dur

    # Contadores y tasas de error NSL-KDD
    total_pkts = max(fdata["count"], 1)
    fdata["serror_rate"] = fdata["serror_count"] / total_pkts
    fdata["rerror_rate"] = fdata["rerror_count"] / total_pkts
    srv_count = max(fdata["srv_count"], 1)
    fdata["srv_serror_rate"] = fdata["srv_error_count"] / srv_count

    # ----------------------------------
    # Cálculos finales (UNSW-NB15)
    # ----------------------------------
    fdata["dur"] = dur  # 'dur' difiere de 'duration'
    sinpkt, sjit = compute_intervals_and_jitter(fdata["src_timestamps"])
    dinpkt, djit = compute_intervals_and_jitter(fdata["dst_timestamps"])
    fdata["sinpkt"] = sinpkt
    fdata["sjit"] = sjit
    fdata["dinpkt"] = dinpkt
    fdata["djit"] = djit

    if dur > 0:
        total_b = fdata["sbytes"] + fdata["dbytes"]
        fdata["rate"] = total_b / dur
        fdata["sload"] = (fdata["sbytes"] * 8) / dur
        fdata["dload"] = (fdata["dbytes"] * 8) / dur
    else:
        fdata["rate"] = 0.0
        fdata["sload"] = 0.0
        fdata["dload"] = 0.0

    spkts = max(fdata["spkts"], 1)
    dpkts = max(fdata["dpkts"], 1)
    fdata["smean"] = fdata["sbytes"] / spkts
    fdata["dmean"] = fdata["dbytes"] / dpkts

    # Handshake times (UNSW)
    if fdata["syn_time"] and fdata["synack_time"]:
        fdata["synack"] = fdata["synack_time"] - fdata["syn_time"]
    if fdata["synack_time"] and fdata["ack_time"]:
        fdata["ackdat"] = fdata["ack_time"] - fdata["synack_time"]
    if fdata["syn_time"] and fdata["ack_time"]:
        fdata["tcprtt"] = fdata["ack_time"] - fdata["syn_time"]

    # sloss/dloss con TCP
    if fdata["protocol_type"] == 6.0:  # TCP
        lost_src = 0
        for seqnum in fdata["src_seq_history"]:
            acked = any(ack >= seqnum for ack in fdata["dst_ack_history"])
            if not acked:
                lost_src += 1
        fdata["sloss"] = lost_src
        lost_dst = 0
        for seqnum in fdata["dst_seq_history"]:
            acked = any(ack >= seqnum for ack in fdata["src_ack_history"])
            if not acked:
                lost_dst += 1
        fdata["dloss"] = lost_dst
    else:
        fdata["sloss"] = 0
        fdata["dloss"] = 0

    # ----------------------------------
    # Cálculos finales (CICIDS2018)
    # ----------------------------------
    dst_port = flow_id[3]
    fdata["Dst Port"] = dst_port
    proto_int = flow_id[4]
    fdata["Protocol"] = proto_int
    fdata["Flow Duration"] = dur

    fdata["Tot Fwd Pkts"] = fdata["spkts"]
    fdata["Tot Bwd Pkts"] = fdata["dpkts"]

    if fdata["fwd_pkt_len_list"]:
        fdata["Fwd Pkt Len Max"] = max(fdata["fwd_pkt_len_list"])
        fdata["Fwd Pkt Len Min"] = min(fdata["fwd_pkt_len_list"])
        fdata["Fwd Pkt Len Mean"] = sum(fdata["fwd_pkt_len_list"]) / len(fdata["fwd_pkt_len_list"])
    else:
        fdata["Fwd Pkt Len Max"] = 0
        fdata["Fwd Pkt Len Min"] = 0
        fdata["Fwd Pkt Len Mean"] = 0

    if fdata["bwd_pkt_len_list"]:
        fdata["Bwd Pkt Len Max"] = max(fdata["bwd_pkt_len_list"])
        fdata["Bwd Pkt Len Min"] = min(fdata["bwd_pkt_len_list"])
        fdata["Bwd Pkt Len Mean"] = sum(fdata["bwd_pkt_len_list"]) / len(fdata["bwd_pkt_len_list"])
    else:
        fdata["Bwd Pkt Len Max"] = 0
        fdata["Bwd Pkt Len Min"] = 0
        fdata["Bwd Pkt Len Mean"] = 0

    total_pkts_flow = fdata["spkts"] + fdata["dpkts"]
    total_bytes_flow = fdata["sbytes"] + fdata["dbytes"]
    if dur > 0:
        fdata["Flow Byts/s"] = total_bytes_flow / dur
        fdata["Flow Pkts/s"] = total_pkts_flow / dur
    else:
        fdata["Flow Byts/s"] = 0
        fdata["Flow Pkts/s"] = 0

    if len(fdata["all_timestamps"]) > 1:
        intervals = []
        atimes = list(fdata["all_timestamps"])
        for i in range(1, len(atimes)):
            intervals.append(atimes[i] - atimes[i-1])
        fdata["Flow IAT Mean"] = sum(intervals)/len(intervals)
    else:
        fdata["Flow IAT Mean"] = 0

    if len(fdata["dst_timestamps"]) > 1:
        intervals = []
        dtimes = list(fdata["dst_timestamps"])
        for i in range(1, len(dtimes)):
            intervals.append(dtimes[i] - dtimes[i-1])
        bwd_iat_tot = sum(intervals)
        bwd_iat_mean = bwd_iat_tot / len(intervals)
        variance = sum((x - bwd_iat_mean)**2 for x in intervals)/len(intervals)
        bwd_iat_std = variance**0.5
        bwd_iat_max = max(intervals)
        bwd_iat_min = min(intervals)

        fdata["Bwd IAT Tot"] = bwd_iat_tot
        fdata["Bwd IAT Mean"] = bwd_iat_mean
        fdata["Bwd IAT Std"] = bwd_iat_std
        fdata["Bwd IAT Max"] = bwd_iat_max
        fdata["Bwd IAT Min"] = bwd_iat_min
    else:
        fdata["Bwd IAT Tot"] = 0
        fdata["Bwd IAT Mean"] = 0
        fdata["Bwd IAT Std"] = 0
        fdata["Bwd IAT Max"] = 0
        fdata["Bwd IAT Min"] = 0

    fdata["Fwd PSH Flags"] = fdata["fwd_psh_flags_cnt"]
    fdata["Fwd URG Flags"] = fdata["fwd_urg_flags_cnt"]

    if dur > 0:
        fdata["Bwd Pkts/s"] = fdata["dpkts"] / dur
    else:
        fdata["Bwd Pkts/s"] = 0

    if len(fdata["all_pkt_len_list"]) > 1:
        mean_len = sum(fdata["all_pkt_len_list"])/len(fdata["all_pkt_len_list"])
        var_sum = sum((x - mean_len)**2 for x in fdata["all_pkt_len_list"])
        fdata["Pkt Len Var"] = var_sum / len(fdata["all_pkt_len_list"])
    else:
        fdata["Pkt Len Var"] = 0

    fdata["FIN Flag Cnt"] = fdata["fin_flag_cnt"]
    fdata["RST Flag Cnt"] = fdata["rst_flag_cnt"]
    fdata["PSH Flag Cnt"] = fdata["psh_flag_cnt"]
    fdata["ACK Flag Cnt"] = fdata["ack_flag_cnt"]
    fdata["URG Flag Cnt"] = fdata["urg_flag_cnt"]

    fwd_pkts = max(fdata["spkts"], 1)
    fdata["Down/Up Ratio"] = fdata["dpkts"] / fwd_pkts

    fdata["Init Fwd Win Byts"] = fdata["init_fwd_win_bytes"] if fdata["init_fwd_win_bytes"] else 0
    fdata["Init Bwd Win Byts"] = fdata["init_bwd_win_bytes"] if fdata["init_bwd_win_bytes"] else 0

    fdata["Fwd Seg Size Min"] = fdata["fwd_seg_size_min"] if fdata["fwd_seg_size_min"] else 0

    fdata["Active Mean"] = 0
    fdata["Active Std"] = 0
    fdata["Idle Min"] = 0

    # Etiquetas
    if fdata["attack"] == 1:
        fdata["Label"] = "Malicious"
        fdata["Threat"] = 1
        fdata["Attack Type"] = "Generic"
    else:
        fdata["Label"] = "Benign"
        fdata["Threat"] = 0
        fdata["Attack Type"] = "None"

    # -------------------------------------------------------------------------
    # Generamos "row" final con todos los campos
    # -------------------------------------------------------------------------
    row = {}
    for c in ALL_FIELDS:
        row[c] = fdata.get(c, 0)

    # -------------------------------------------------------------------------
    # Ahora separamos en 3 filas para cada dataset
    # -------------------------------------------------------------------------
    row_nslkdd = {k: row[k] for k in FIELD_NAMES_NSLKDD}
    row_unsw15 = {k: row[k] for k in FIELD_NAMES_UNSW15}
    row_cicids = {k: row[k] for k in FIELD_NAMES_CIC2018}

    # Añadimos a las listas globales
    nslkdd_rows.append(row_nslkdd)
    unsw15_rows.append(row_unsw15)
    cicids2018_rows.append(row_cicids)

    logger.info(f"[INFO] Cerrando flujo: {flow_id}, dur={dur:.2f}s")
    print(f"[INFO] Cerrando flujo: {flow_id}, dur={dur:.2f}s")

# ============================================================================
# DETECCIÓN DE DDoS
# ============================================================================
def check_ddos(src_ip, dst_ip):
    """
    funcion.
    Verifica si la IP origen excede el THRESHOLD de conexiones
    en la ventana TIME_WINDOW, y enciende alerta si es así, no 
    tomar al pie de la letra.
    """
    now = time.time()
    while connection_tracker[src_ip] and (now - connection_tracker[src_ip][0] > TIME_WINDOW):
        connection_tracker[src_ip].popleft()
    if len(connection_tracker[src_ip]) > THRESHOLD:
        if src_ip not in ddos_alerted:
            alert = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "attack": "Possible DDoS"
            }
            alerts_list.append(alert)
            ddos_alerted[src_ip] = now
            logger.warning(f"[DDoS] Detectado posible DDoS: {alert}")
            log_alert(alert)
            t = threading.Timer(COOLDOWN, remove_ddos_alert, args=[src_ip])
            t.start()

def remove_ddos_alert(ip):
    """
    funcion:
    Remueve la IP del mapa ddos_alerted una vez cumplido el cooldown, falta.
    """
    ddos_alerted.pop(ip, None)
    logger.info(f"[DDoS] Cooldown finalizado para IP: {ip}")
    print(f"[DDoS] Cooldown finalizado para IP: {ip}")

def log_alert(alert):
    """
    funcion:
    Registra la alerta en detections.json (append a la lista JSON).
    """
    data = []
    if os.path.exists(DETECTIONS_JSON):
        try:
            with open(DETECTIONS_JSON, "r") as f:
                data = json.load(f)
        except:
            data = []
    data.append(alert)
    with open(DETECTIONS_JSON, "w") as f:
        json.dump(data, f, indent=4)

# ============================================================================
# PROCESAR PAQUETE
# ============================================================================
def process_packet(packet):
    """
    Función principal para cada paquete capturado:
    - Maneja fragmentos si está fragmentado.
    - Identifica flow y actualiza su estado.
    - Cierra flow si detecta FIN/RST.
    - Registra info para DDoS.
    """
    if IP not in packet:
        return
    ip_layer = packet[IP]

    # Fragmentos
    if (ip_layer.flags & 0x1) or (ip_layer.frag > 0):
        process_fragment(packet)
        return

    fid = get_flow_id(packet)
    if fid:
        update_flow_state(fid, packet)
        # Si TCP con FIN o RST => cerrar flow
        if TCP in packet:
            flags_str = str(packet[TCP].flags)
            if "F" in flags_str or "R" in flags_str:
                close_flow(fid)

    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    connection_tracker[src_ip].append(time.time())
    check_ddos(src_ip, dst_ip)

    # Ejemplo: guardamos features mínimas en extracted_features
    feats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "srcip": src_ip,
        "dstip": dst_ip,
        "protocol": PROTOCOL_MAP.get(ip_layer.proto, str(ip_layer.proto)),
        "pkt_len": len(packet)
    }
    extracted_features.append(feats)

# ============================================================================
# BUCLes DE CAPTURA Y LIMPIEZA
# ============================================================================
def sniff_loop():
    """
    funcion:
    Bucle para capturar paquetes en lotes (count=50, timeout=1)
    usando el filtro 'ip' a su vez llama a process_packet.
    """
    while sniffing_active:
        sniff(filter="ip", prn=process_packet, store=0, count=50, timeout=1)

def cleanup_loop():
    """funcion:
    Bucle para limpieza periódica cada CLEANUP_INTERVAL.
    - Cierra flows inactivos (FLOW_TIMEOUT).
    - Limpia fragmentos.
    """
    while sniffing_active:
        time.sleep(CLEANUP_INTERVAL)
        now = time.time()
        to_close = []
        for fid, data in flow_state.items():
            if (now - data["last_activity"]) > FLOW_TIMEOUT:
                to_close.append(fid)
        for fid in to_close:
            close_flow(fid)
        cleanup_fragments()

# ============================================================================
# start_sniffing / stop_sniffing
# ============================================================================
def start_sniffing():
    """
    funcion:
    Inicializa la captura en hilos separados (sniff_loop y cleanup_loop).
    """
    global sniffing_active, sniffing_thread, cleanup_thread
    sniffing_active = True
    sniffing_thread = threading.Thread(target=sniff_loop, daemon=True)
    sniffing_thread.start()

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("[INFO] Captura iniciada.")
    print("[INFO] Captura iniciada.")

def stop_sniffing():
    """
    funcion:
    Detiene la captura y el hilo de limpieza.
    Cierra flows restantes y exporta datos finales.
    """
    global sniffing_active, sniffing_thread, cleanup_thread
    sniffing_active = False
    if sniffing_thread:
        sniffing_thread.join()
    if cleanup_thread:
        cleanup_thread.join()

    # Cerrar flows
    for fid in list(flow_state.keys()):
        close_flow(fid)

    export_all_data()
    logger.info("[INFO] Captura detenida.")
    print("[INFO] Captura detenida.")

# ============================================================================
# EXPORTAR DATOS A JSON
# ============================================================================
def export_all_data():
    data_out={
        "alerts":list(alerts_list),
        "nslkdd":nslkdd_rows,
        "unsw15":unsw15_rows,
        "cicids2018":cicids2018_rows
    }
    with open(FINAL_DATA_JSON,"w")as f:
        json.dump(data_out,f,indent=4)
    logger.info(
        f"[INFO] Se generó {FINAL_DATA_JSON} con "
        f"{len(nslkdd_rows)} flows NSL-KDD, "
        f"{len(unsw15_rows)} flows UNSW15, "
        f"{len(cicids2018_rows)} flows CICIDS2018. "
        f"Alertas: {len(alerts_list)}."
    )
    print(
        f"[INFO] Se generó {FINAL_DATA_JSON} con "
        f"{len(nslkdd_rows)} flows NSL-KDD, "
        f"{len(unsw15_rows)} flows UNSW15, "
        f"{len(cicids2018_rows)} flows CICIDS2018. "
        f"Alertas: {len(alerts_list)}."
    )

# ============================================================================
# get_alerts / get_features
# ============================================================================
def get_alerts():
    """funcion: Devuelve la lista de alertas DDoS."""
    return list(alerts_list)

def get_features():
    """funcion: Devuelve la lista de features extraídos (compatibilidad con interfaz)."""
    return list(extracted_features)

# ============================================================================
# FUNCIÓN PRINCIPAL (MAIN)
# ============================================================================
def main():
    """
    Función principal que inicia la captura (start_sniffing) y
    mantiene el script corriendo hasta que se reciba Ctrl+C.
    """
    logger.info("[INFO] Iniciando IDS Flow-based con NSL-KDD, UNSW-NB15 y CICIDS2018.")
    print("[INFO] Iniciando IDS Flow-based con NSL-KDD, UNSW-NB15 y CICIDS2018.")
   
    start_sniffing()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[INFO] Interrupción detectada, deteniendo...")
        print("[INFO] Interrupción detectada, deteniendo...")
        stop_sniffing()
        logger.info("[INFO] IDS detenido.")
        print("[INFO] IDS detenido.")

# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    main()
