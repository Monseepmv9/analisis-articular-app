"""
M.M. - MotionMetrics
=====================================================================
Análisis cinemático 2D (MediaPipe). Soporta análisis simultáneo de
múltiples articulaciones (incluyendo muñeca y rotaciones de cadera), 
recorte de video, velocidad angular, dibujo de arco de ángulo, estandarización clínica AAOS
e integración de Diagnóstico y Observaciones Clínicas en los reportes.
"""

import io
import math
import os
import tempfile
import time
import urllib.request
from datetime import date

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import av

# ----------------------------------------------------------------------------
# 1. Configuración de Modelos y Datos Clínicos
# ----------------------------------------------------------------------------

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), 
    (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), 
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

BODY_PART_LABELS = {
    "hombro": "Hombro", 
    "codo": "Codo", 
    "muneca": "Muñeca",
    "cadera": "Cadera", 
    "rodilla": "Rodilla", 
    "tobillo": "Tobillo",
}

NORMATIVE_RANGES = {
    "hombro": {"flexion": (0, 180), "extension": (0, 60), "abduccion": (0, 180), "aduccion": (0, 30), "rot_interna": (0, 70), "rot_externa": (0, 90)},
    "codo": {"flexion": (0, 150), "extension": (0, 10)},
    "muneca": {"flexion": (0, 80), "extension": (0, 70), "desviacion_radial": (0, 20), "desviacion_ulnar": (0, 30)},
    "cadera": {"flexion": (0, 120), "extension": (0, 30), "abduccion": (0, 45), "aduccion": (0, 30), "rot_interna": (0, 45), "rot_externa": (0, 45)},
    "rodilla": {"flexion": (0, 135), "extension": (0, 10)},
    "tobillo": {"dorsiflexion": (0, 20), "plantiflexion": (0, 50)}
}

MOVEMENTS = {
    "hombro": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_0_rest", "lm": {"left": [23, 11, 13], "right": [24, 12, 14]}},
        {"id": "rot_interna", "label": "Rotación interna", "view": "frontal", "mode": "vertical", "lm": {"left": [13, 15], "right": [14, 16]}},
        {"id": "rot_externa", "label": "Rotación externa", "view": "frontal", "mode": "vertical", "lm": {"left": [13, 15], "right": [14, 16]}},
    ],
    "codo": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 13, 15], "right": [12, 14, 16]}},
    ],
    "muneca": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "desviacion_radial", "label": "Desviación Radial", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
        {"id": "desviacion_ulnar", "label": "Desviación Ulnar", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [13, 15, 19], "right": [14, 16, 20]}},
    ],
    "cadera": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "rot_interna", "label": "Rotación interna", "view": "frontal", "mode": "vertical", "lm": {"left": [25, 27], "right": [26, 28]}},
        {"id": "rot_externa", "label": "Rotación externa", "view": "frontal", "mode": "vertical", "lm": {"left": [25, 27], "right": [26, 28]}},
    ],
    "rodilla": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [23, 25, 27], "right": [24, 26, 28]}},
    ],
    "tobillo": [
        {"id": "dorsiflexion", "label": "Dorsiflexión", "view": "lateral", "mode": "angle_90_rest", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
        {"id": "plantiflexion", "label": "Plantiflexión", "view": "lateral", "mode": "angle_90_rest", "lm": {"left": [25, 27, 31], "right": [26, 28, 32]}},
    ],
}

VISIBILITY_MIN = 0.5
SMOOTH_WINDOW = 3

# ----------------------------------------------------------------------------
# 2. Geometría y Dibujo del Arco
# ----------------------------------------------------------------------------

def angle_between(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0: return None
    cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def angle_from_vertical(vertex, point):
    v = np.array([point[0] - vertex[0], point[1] - vertex[1]])
    mag = np.linalg.norm(v)
    if mag == 0: return None
    up = np.array([0, -1])
    cos_angle = np.clip(np.dot(v, up) / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def draw_angle_arc(frame, p1, p2, p3, color, radius=35, thickness=2):
    angle1 = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    angle2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
    if angle1 < 0: angle1 += 360
    if angle2 < 0: angle2 += 360
    
    a_min, a_max = min(angle1, angle2), max(angle1, angle2)
    if a_max - a_min > 180:
        start_angle, end_angle = a_max, a_min + 360
    else:
        start_angle, end_angle = a_min, a_max
        
    cv2.ellipse(frame, (int(p2[0]), int(p2[1])), (radius, radius), 0, start_angle, end_angle, color, thickness)

def pick_main_person(pose_landmarks_list):
    if len(pose_landmarks_list) <= 1: return pose_landmarks_list[0] if pose_landmarks_list else None
    best, best_area = None, -1
    for lm in pose_landmarks_list:
        xs, ys = [p.x for p in lm], [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area: best, best_area = lm, area
    return best

# ----------------------------------------------------------------------------
# 3. Procesamiento y MediaPipe
# ----------------------------------------------------------------------------

def download_model_if_needed():
    model_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")
    if not os.path.exists(model_path): urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path

def create_landmarker():
    model_path = download_model_if_needed()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6, min_tracking_confidence=0.6,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def analyze_frame_multijoint(frame, landmarker, configs, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks: return frame, [None]*len(configs), False

    h, w = frame.shape[:2]
    lm = pick_main_person(result.pose_landmarks)

    p_shoulder_l, p_shoulder_r = lm[11], lm[12]
    dx, dy = (p_shoulder_r.x - p_shoulder_l.x) * w, (p_shoulder_r.y - p_shoulder_l.y) * h
    tilt_angle = abs(math.degrees(math.atan2(dy, dx)))
    if 8 < tilt_angle < 172:
        cv2.putText(frame, "⚠️ Camara inclinada", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    for i1, i2 in POSE_CONNECTIONS:
        p1, p2 = lm[i1], lm[i2]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (150, 150, 150), 2)

    angles_out = []
    any_low_conf = False

    for idx, config in enumerate(configs):
        movement, side, smooth_buffer, base_color = config["mov"], config["side"], config["buffer"], config["color"]
        pts_norm = [lm[i] for i in movement["lm"][side]]
        pts_px = [(p.x * w, p.y * h) for p in pts_norm]
        low_conf = any((getattr(p, "visibility", 1.0) or 1.0) < VISIBILITY_MIN for p in pts_norm)
        if low_conf: any_low_conf = True

        if movement["mode"] == "vertical":
            vertex, point = pts_px
            angle = angle_from_vertical(vertex, point)
            if angle is not None:
                color = (60, 70, 226) if low_conf else base_color
                ref_pt = (vertex[0], vertex[1] + 50)
                draw_angle_arc(frame, point, vertex, ref_pt, color)
        else:
            a, b, c = pts_px
            vertex = b
            raw_angle = angle_between(a, b, c)
            angle = None
            if raw_angle is not None:
                if movement["mode"] == "angle_0_rest": angle = raw_angle
                elif movement["mode"] == "angle_180_rest": angle = abs(180.0 - raw_angle)
                elif movement["mode"] == "angle_90_rest": angle = abs(raw_angle - 90.0)
                color = (60, 70, 226) if low_conf else base_color
                draw_angle_arc(frame, a, b, c, color)

        color = (60, 70, 226) if low_conf else base_color
        for p in pts_px: cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1)

        smoothed = None
        if angle is not None:
            smooth_buffer.append(angle)
            if len(smooth_buffer) > SMOOTH_WINDOW: smooth_buffer.pop(0)
            smoothed = sum(smooth_buffer) / len(smooth_buffer)
            offset_y = -15 if idx == 0 else 25 
            text_org = (int(vertex[0]) + 15, int(vertex[1]) + offset_y)
            cv2.putText(frame, f"{smoothed:.0f}", text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
        angles_out.append(smoothed)

    return frame, angles_out, any_low_conf

def process_video_multijoint(video_path, configs, target_fps, preview_placeholder, start_sec, end_sec):
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, round(video_fps / target_fps))

    start_frame, end_frame = int(start_sec * video_fps), int(end_sec * video_fps)
    total_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # --- CONFIGURACIÓN PARA RECONSTRUIR EL VIDEO FLUIDO ---
    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_vid = min(640 / orig_w, 640 / orig_h)
    new_w, new_h = int(orig_w * scale_vid), int(orig_h * scale_vid)
    new_w, new_h = new_w - (new_w % 2), new_h - (new_h % 2)

    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = out_tfile.name

    container = av.open(output_video_path, mode='w')
    stream = container.add_stream('h264', rate=int(video_fps))
    stream.width, stream.height, stream.pix_fmt = new_w, new_h, 'yuv420p'
    stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '28'}

    history = []
    frames_processed = 0
    preview_placeholder.info("⏳ Procesando... ")

    while frames_processed <= total_frames:
        ret, frame = cap.read()
        if not ret: break

        current_ms = int((start_frame + frames_processed) * (1000 / video_fps))
        frame_analyzed, angles_out, low_conf = analyze_frame_multijoint(frame, landmarker, configs, current_ms)
        
        if frames_processed % frame_step == 0:
            t = (start_frame + frames_processed) / video_fps
            if all(a is not None for a in angles_out):
                history.append((t, angles_out, low_conf))

        # --- ESCRIBIR EL CUADRO EN EL NUEVO VIDEO ---
        frame_to_write = cv2.resize(frame_analyzed, (new_w, new_h))
        av_frame = av.VideoFrame.from_ndarray(frame_to_write, format='bgr24')
        for packet in stream.encode(av_frame):
            container.mux(packet)

        frames_processed += 1

    # --- CERRAR Y EXTRAER BYTES ---
    for packet in stream.encode(): container.mux(packet)
    container.close()
    cap.release()
    try: landmarker.close()
    except Exception: pass
        
    with open(output_video_path, 'rb') as f:
        video_bytes = f.read()
    try: os.remove(output_video_path)
    except Exception: pass

    preview_placeholder.empty()
    st.success("✅ Procesamiento completado.")

    return history, video_bytes


# ----------------------------------------------------------------------------
# 4. Gráficos y PDF
# ----------------------------------------------------------------------------

def make_multijoint_chart(history, titles, norm_ranges, colors):
    times = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for idx, title in enumerate(titles):
        angles = [h[1][idx] for h in history]
        ax.plot(times, angles, color=colors[idx], linewidth=2, label=title)
        
        if norm_ranges[idx]:
            ax.axhspan(norm_ranges[idx][0], norm_ranges[idx][1], color=colors[idx], alpha=0.1)

    low_conf_pts = [(h[0], h[1][0]) for h in history if h[2]]
    if low_conf_pts:
        lx, ly = zip(*low_conf_pts)
        ax.scatter(lx, ly, color="#e24b4a", s=15, zorder=3, label="Baja confianza")

    ax.set_ylim(0, 180)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Ángulo (°)") 
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def build_pdf_report(sessions):
    last = sessions[-1]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 110, 86)
    pdf.cell(0, 12, "Informe de movilidad articular", ln=True)
    pdf.set_draw_color(15, 110, 86)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 100, 105)
    pdf.multi_cell(0, 6, f"Paciente: {last['patient']}   |   RUN: {last['run']}   |   Generado: {date.today().isoformat()}")
    pdf.ln(4)

    for s in sessions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(28, 43, 48)
        titulo_limpio = s['joint'].replace("—", "-").replace("\u2014", "-")
        pdf.cell(0, 9, f"{titulo_limpio} - {s['date']}", ln=True)

        # -- CORRECCIÓN PDF (1): Usar tempfile para la imagen --
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(s["chart_png"])
            tmp_img_path = tmp_img.name

        pdf.image(tmp_img_path, w=150)
        pdf.ln(2)

        try:
            os.remove(tmp_img_path)
        except Exception:
            pass
        # --------------------------------------------------------

        # -- Resultados clínicos por articulación (igual que en la app) --
        for r in s.get("results", []):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 7, r["title"], ln=True)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(225, 245, 238)
            pdf.cell(60, 7, "Máximo Alcanzado", border=1, fill=True)
            pdf.cell(60, 7, "Rango AAOS Normal", border=1, fill=True)
            pdf.cell(60, 7, "Velocidad Máxima", border=1, fill=True, ln=True)

            pdf.set_font("Helvetica", "", 9)
            pdf.cell(60, 7, f"{r['max']}°", border=1)
            pdf.cell(60, 7, f"{r['norm_min']}° - {r['norm_max']}°", border=1)
            pdf.cell(60, 7, f"{r['vel']} °/s", border=1, ln=True)
            pdf.ln(3)
        pdf.ln(2)

        if s.get("diagnostico"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 6, f"Impresión Diagnóstica: {s['diagnostico']}", ln=True)
            
        if s.get("observaciones"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(28, 43, 48)
            pdf.cell(0, 6, "Observaciones Clínicas:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, s["observaciones"])
        
        pdf.ln(6)

    # -- CORRECCIÓN PDF (2): Detector inteligente de versión --
    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)
    # ---------------------------------------------------------


# ----------------------------------------------------------------------------
# 5. UI Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(page_title="M.M. - MotionMetrics", page_icon="📐", layout="centered")
st.title("M.M. - MotionMetrics 📐")
st.caption("Herramienta de análisis cinemático, goniometría 2D y evaluación de velocidad angular.")

if "sessions" not in st.session_state: st.session_state.sessions = []
if "history" not in st.session_state: st.session_state.history = []

st.header("1. Configuración Principal")
col1, col2 = st.columns(2)
with col1:
    bp1 = st.selectbox("Articulación Principal", list(BODY_PART_LABELS.keys()), format_func=lambda k: BODY_PART_LABELS[k], index=3)
with col2:
    mov_opts1 = MOVEMENTS.get(bp1, [])
    mov_id1 = st.selectbox("Movimiento 1", [m["id"] for m in mov_opts1], format_func=lambda mid: next(m["label"] for m in mov_opts1 if m["id"] == mid))

st.header("2. Configuración Secundaria")
col3, col4 = st.columns(2)
with col3:
    bp2 = st.selectbox("Articulación Secundaria", ["Ninguna"] + list(BODY_PART_LABELS.keys()), index=0)
with col4:
    if bp2 != "Ninguna":
        mov_opts2 = MOVEMENTS.get(bp2, [])
        mov_id2 = st.selectbox("Movimiento 2", [m["id"] for m in mov_opts2], format_func=lambda mid: next(m["label"] for m in mov_opts2 if m["id"] == mid))

col5, col6 = st.columns(2)
with col5:
    side = st.selectbox("Lado a evaluar", ["left", "right"], format_func=lambda s: "Izquierdo" if s == "left" else "Derecho")
with col6:
    camera_view = st.selectbox("Vista de cámara", ["lateral", "frontal"], index=0)

target_fps = st.select_slider("Resolución de captura", options=[5, 10, 15], value=10)

st.header("3. Video y Análisis")
uploaded_file = st.file_uploader("Sube el video del paciente", type=["mp4", "mov", "avi"])
preview_placeholder = st.empty()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap_temp = cv2.VideoCapture(tmp_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_temp = cap_temp.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps_temp
    cap_temp.release()

    st.subheader("Recortar Video")
    start_sec, end_sec = st.slider("Selecciona el segmento donde ocurre el movimiento", 0.0, float(duration), (0.0, float(duration)), step=0.1)

    if st.button("Procesar Análisis"):
        m1 = next(m for m in mov_opts1 if m["id"] == mov_id1)
        configs = [{"mov": m1, "side": side, "buffer": [], "color": (255, 120, 30)}] # Azul BGR
        titles = [f"{BODY_PART_LABELS[bp1]} - {m1['label']}"]
        norm_ranges = [NORMATIVE_RANGES[bp1][mov_id1]]
        colors_hex = ["#1e78ff"]

        if bp2 != "Ninguna":
            m2 = next(m for m in mov_opts2 if m["id"] == mov_id2)
            configs.append({"mov": m2, "side": side, "buffer": [], "color": (50, 205, 50)}) # Verde BGR
            titles.append(f"{BODY_PART_LABELS[bp2]} - {m2['label']}")
            norm_ranges.append(NORMATIVE_RANGES[bp2][mov_id2])
            colors_hex.append("#32cd32")

        with st.spinner("Procesando cinemática y velocidad angular..."):
            history, video_bytes = process_video_multijoint(tmp_path, configs, target_fps, preview_placeholder, start_sec, end_sec)
            st.session_state.history = history
            st.session_state.video_bytes = video_bytes # AQUÍ SE GUARDA EL VIDEO FINAL
            st.session_state.titles = titles
            st.session_state.norm_ranges = norm_ranges
            st.session_state.colors_hex = colors_hex

        try: os.remove(tmp_path)
        except Exception: pass

if st.session_state.history:
    history = st.session_state.history
    titles = st.session_state.titles
    norm_ranges = st.session_state.norm_ranges
    colors_hex = st.session_state.colors_hex

    # --- MOSTRAR EL VIDEO FLUIDO CON LOS ÁNGULOS ---
    if "video_bytes" in st.session_state:
        st.video(st.session_state.video_bytes)

    st.header("4. Evolución Temporal Coordinada")
    fig = make_multijoint_chart(history, titles, norm_ranges, colors_hex)
    st.pyplot(fig)
    
    chart_png = fig_to_png_bytes(fig)
    joint_name_combined = " + ".join(titles)
    
    max_angles_session = []
    max_vels_session = []

    st.header("5. Resultados Clínicos")
    for idx, title in enumerate(titles):
        st.write(f"**{title}**")
        angles = [h[1][idx] for h in history]
        vels = []
        for i in range(1, len(history)):
            dt = history[i][0] - history[i-1][0]
            if dt > 0.05: vels.append(abs(history[i][1][idx] - history[i-1][1][idx]) / dt)
        
        max_angle = max(angles)
        max_vel = max(vels) if vels else 0
        max_angles_session.append(max_angle)
        max_vels_session.append(max_vel)

        c1, c2, c3 = st.columns(3)
        c1.metric("Máximo Alcanzado", f"{max_angle:.1f}°")
        c2.metric("Rango AAOS Normal", f"{norm_ranges[idx][0]}° - {norm_ranges[idx][1]}°")
        c3.metric("Velocidad Máxima", f"{max_vel:.1f} °/s")
        st.divider()

    st.header("6. Registro de sesión y Observaciones")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Nombre del paciente")
    patient_run = p2.text_input("RUN", placeholder="12.345.678-9")
    test_date = p3.date_input("Fecha", value=date.today())
    
    diagnostico = st.text_input("Impresión Diagnóstica (Opcional)", placeholder="Ej: Limitación funcional leve por probable tendinopatía...")
    observaciones = st.text_area("Observaciones Clínicas", placeholder="Ej: Paciente refiere dolor en los últimos 15° de flexión. Se observan compensaciones musculares...")

    if st.button("Guardar temporalmente"):
        joint_results = []
        for idx, title in enumerate(titles):
            joint_results.append({
                "title": title,
                "max": round(max_angles_session[idx], 1),
                "norm_min": norm_ranges[idx][0],
                "norm_max": norm_ranges[idx][1],
                "vel": round(max_vels_session[idx], 1),
            })

        st.session_state.sessions.append({
            "run": patient_run or "-", 
            "patient": patient_name or "Sin nombre", 
            "date": str(test_date),
            "joint": joint_name_combined, 
            "results": joint_results,
            "diagnostico": diagnostico,
            "observaciones": observaciones,
            "chart_png": chart_png
        })
        st.success("Guardado en la sesión actual.")

if st.session_state.sessions:
    st.subheader("Historial (Se borrará al recargar la página)")
    sessions_df = pd.DataFrame([{
        "Paciente": s["patient"], 
        "Fecha": s["date"], 
        "Articulación": s["joint"],
        "Resultados": " | ".join(
            f"{r['title']}: {r['max']}° (máx {r['vel']} °/s)" for r in s.get("results", [])
        ),
        "Diagnóstico": s.get("diagnostico", ""),
        "Observaciones": s.get("observaciones", "")
    } for s in st.session_state.sessions])
    
    st.dataframe(sessions_df, use_container_width=True)

    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        if st.button("Generar informe (PDF)"):
            pdf_bytes = build_pdf_report(st.session_state.sessions)
            safe_filename = st.session_state.sessions[-1]["patient"].replace(' ', '_').replace('-', '_')
            st.download_button("Descargar informe (PDF)", data=pdf_bytes, file_name=f"informe_{safe_filename}.pdf", mime="application/pdf")
    with col_xlsx:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            sessions_df.to_excel(writer, index=False, sheet_name="Pruebas")
        st.download_button("Descargar historial (Excel)", data=excel_buffer.getvalue(), file_name="historial_pruebas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")