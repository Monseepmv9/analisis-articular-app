"""
M.M. - MotionMetrics
=====================================================================
Detecta la pose de una persona en un video, calcula el ángulo clínico (AAOS),
permite recortar el video, alerta sobre inclinación de cámara y calcula velocidad angular.
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

# La grabación en vivo desde el navegador es opcional
try:
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Modelo "Full" para máxima precisión
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# ----------------------------------------------------------------------------
# 1. Definición de Articulaciones, Movimientos y Rangos Normales (AAOS)
# ----------------------------------------------------------------------------

BODY_PART_LABELS = {
    "hombro": "Hombro",
    "codo": "Codo",
    "cadera": "Cadera",
    "rodilla": "Rodilla",
    "tobillo": "Tobillo",
}

NORMATIVE_RANGES = {
    "hombro": {"flexion": (0, 180), "extension": (0, 60), "abduccion": (0, 180), "aduccion": (0, 30), "rot_interna": (0, 70), "rot_externa": (0, 90)},
    "codo": {"flexion": (0, 150), "extension": (0, 10)},
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
    "cadera": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
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
# 2. Geometría
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
# 3. Procesamiento de video con MediaPipe Pose
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
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def analyze_frame(frame, landmarker, movement, side, timestamp_ms, smooth_buffer):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks: return frame, None, False

    h, w = frame.shape[:2]
    lm = pick_main_person(result.pose_landmarks)

    # ALERTA DE INCLINACIÓN DE CÁMARA (Validación de hombros paralelos al piso)
    p_shoulder_l, p_shoulder_r = lm[11], lm[12]
    dx = (p_shoulder_r.x - p_shoulder_l.x) * w
    dy = (p_shoulder_r.y - p_shoulder_l.y) * h
    tilt_angle = abs(math.degrees(math.atan2(dy, dx)))
    
    # Si la inclinación se desvía más de 8 grados de la horizontal (0° o 180°)
    if 8 < tilt_angle < 172:
        cv2.putText(frame, "⚠️ Camara inclinada", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    for i1, i2 in POSE_CONNECTIONS:
        p1, p2 = lm[i1], lm[i2]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (150, 150, 150), 2)

    idx = movement["lm"][side]
    pts_norm = [lm[i] for i in idx]
    pts_px = [(p.x * w, p.y * h) for p in pts_norm]
    low_conf = any((getattr(p, "visibility", 1.0) or 1.0) < VISIBILITY_MIN for p in pts_norm)

    if movement["mode"] == "vertical":
        vertex, point = pts_px
        angle = angle_from_vertical(vertex, point)
    else:
        a, b, c = pts_px
        vertex = b
        raw_angle = angle_between(a, b, c)
        
        if raw_angle is None: angle = None
        elif movement["mode"] == "angle_0_rest": angle = raw_angle
        elif movement["mode"] == "angle_180_rest": angle = abs(180.0 - raw_angle)
        elif movement["mode"] == "angle_90_rest": angle = abs(raw_angle - 90.0) 

    color = (60, 70, 226) if low_conf else (35, 93, 242)
    
    if angle is not None:
        if movement["mode"] == "vertical":
            ref_pt = (vertex[0], vertex[1] + 50)
            draw_angle_arc(frame, point, vertex, ref_pt, color)
        else:
            draw_angle_arc(frame, a, b, c, color)

    for p in pts_px:
        cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1)

    smoothed = None
    if angle is not None:
        smooth_buffer.append(angle)
        if len(smooth_buffer) > SMOOTH_WINDOW: smooth_buffer.pop(0)
        smoothed = sum(smooth_buffer) / len(smooth_buffer)
        
        label_num = f"{smoothed:.0f}"
        text_org = (int(vertex[0]) + 15, int(vertex[1]) - 15)
        cv2.putText(frame, label_num, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label_num, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.circle(frame, (text_org[0] + tw + 6, text_org[1] - th + 4), 4, color, 2, cv2.LINE_AA)

    return frame, smoothed, low_conf

def process_video(video_path, movement, side, target_fps, preview_placeholder, progress_bar, progress_text, start_sec, end_sec):
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, round(video_fps / target_fps))

    # RECORTAR VIDEO SEGÚN EL SLIDER
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    total_frames_to_process = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    orig_width, orig_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_dim = 640
    scale = min(max_dim / orig_width, max_dim / orig_height)
    new_width, new_height = int(orig_width * scale), int(orig_height * scale)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = out_tfile.name
    
    container = av.open(output_video_path, mode='w')
    stream = container.add_stream('h264', rate=int(video_fps))
    stream.width, stream.height = new_width, new_height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '28'}

    history = []
    smooth_buffer = []
    frames_processed = 0

    preview_placeholder.info("⏳ Procesando análisis cinemático...")

    while frames_processed <= total_frames_to_process:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (new_width, new_height))
        current_ms = int((start_frame + frames_processed) * (1000 / video_fps))

        frame_analyzed, angle, low_conf = analyze_frame(frame, landmarker, movement, side, current_ms, smooth_buffer)
        
        if frames_processed % frame_step == 0:
            t = (start_frame + frames_processed) / video_fps
            if angle is not None:
                history.append((t, angle, low_conf))

        av_frame = av.VideoFrame.from_ndarray(frame_analyzed, format='bgr24')
        for packet in stream.encode(av_frame):
            container.mux(packet)

        if total_frames_to_process > 0 and frames_processed % 5 == 0:
            progress_bar.progress(min(1.0, frames_processed / total_frames_to_process))
            progress_text.caption(f"Analizando segundo {frames_processed/video_fps:.1f} de {(total_frames_to_process)/video_fps:.1f}...")
                
        frames_processed += 1

    for packet in stream.encode(): container.mux(packet)
    container.close()
    cap.release()
    
    try: landmarker.close()
    except Exception: pass
        
    progress_bar.progress(1.0)
    progress_text.empty()
    preview_placeholder.empty()
    
    st.success("✅ Procesamiento completado:")
    with open(output_video_path, 'rb') as f: st.video(f.read())
        
    try: os.remove(output_video_path)
    except Exception: pass

    return history


# ----------------------------------------------------------------------------
# 4. Interfaz y Reportes
# ----------------------------------------------------------------------------

def make_chart(history, title, norm_range):
    times = [h[0] for h in history]
    angles = [h[1] for h in history]
    low_conf_pts = [(h[0], h[1]) for h in history if h[2]]

    fig, ax = plt.subplots(figsize=(7, 3))
    if norm_range:
        ax.axhspan(norm_range[0], norm_range[1], color='green', alpha=0.1, label='Rango Normal Esperado')

    ax.plot(times, angles, color="#0f6e56", linewidth=2)
    if low_conf_pts:
        lx, ly = zip(*low_conf_pts)
        ax.scatter(lx, ly, color="#e24b4a", s=12, zorder=3, label="Baja confianza")
        ax.legend(loc="upper right", fontsize=8)
    
    max_y = max(180, (norm_range[1] + 20) if norm_range else 180)
    ax.set_ylim(0, max_y)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Ángulo (°)") 
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# Configuración Base UI
st.set_page_config(page_title="M.M. - MotionMetrics", page_icon="📐", layout="centered")
st.title("M.M. - MotionMetrics 📐")
st.caption("Herramienta de análisis cinemático y goniometría 2D.")

if "sessions" not in st.session_state: st.session_state.sessions = []
if "history" not in st.session_state: st.session_state.history = []

st.header("1. Configuracion de la prueba")
col1, col2 = st.columns(2)
with col1:
    body_part = st.selectbox("Articulación", list(BODY_PART_LABELS.keys()), format_func=lambda k: BODY_PART_LABELS[k], index=4)
with col2:
    movement_options = MOVEMENTS[body_part]
    movement_id = st.selectbox("Movimiento", [m["id"] for m in movement_options], format_func=lambda mid: next(m["label"] for m in movement_options if m["id"] == mid))

movement = next(m for m in movement_options if m["id"] == movement_id)
current_norm_range = NORMATIVE_RANGES[body_part][movement_id]

col3, col4 = st.columns(2)
with col3:
    side = st.selectbox("Lado", ["left", "right"], format_func=lambda s: "Izquierdo" if s == "left" else "Derecho")
with col4:
    default_view_idx = 0 if movement["view"] == "lateral" else 1
    camera_view = st.selectbox("Vista de cámara", ["lateral", "frontal"], format_func=lambda v: "Lateral (de perfil)" if v == "lateral" else "Frontal (de frente)", index=default_view_idx)

if camera_view != movement["view"]:
    st.warning(f'Para "{movement["label"]}" la cámara debe estar en vista {"frontal" if movement["view"] == "frontal" else "lateral"}.')

st.info(f"📊 **Rango esperado (AAOS):** {current_norm_range[0]}° - {current_norm_range[1]}°")
target_fps = st.select_slider("Resolución de captura", options=[5, 10, 15], value=10, format_func=lambda f: f"{f} cuadros/seg")

st.header("2. Video")
uploaded_file = st.file_uploader("Sube el video del paciente", type=["mp4", "mov", "avi", "mkv", "webm"])

preview_placeholder = st.empty()
progress_bar = st.progress(0.0)
progress_text = st.empty()

if uploaded_file is not None:
    # Leer archivo temporal para el slider
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap_temp = cv2.VideoCapture(tmp_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_temp = cap_temp.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps_temp
    cap_temp.release()

    st.subheader("3. Recortar Video")
    st.caption("Usa los controles para procesar únicamente el segmento donde ocurre el movimiento (ahorra tiempo).")
    start_sec, end_sec = st.slider("Selecciona inicio y fin (segundos)", 0.0, float(duration), (0.0, float(duration)), step=0.1)

    analyze_disabled = camera_view != movement["view"]
    run_analysis = st.button("Analizar segmento", disabled=analyze_disabled)

    if run_analysis:
        with st.spinner("Procesando cinemática..."):
            history = process_video(tmp_path, movement, side, target_fps, preview_placeholder, progress_bar, progress_text, start_sec, end_sec)
        st.session_state.history = history
        
        # Eliminar archivo de manera robusta y segura
        try: os.remove(tmp_path)
        except Exception: pass

        if not history: st.error("No se detectó a la persona. Revisa la iluminación.")

history = st.session_state.history

if history:
    st.header("4. Evolución temporal y Velocidad")
    chart_title = f'{BODY_PART_LABELS[body_part]} - {movement["label"]} ({"izq." if side == "left" else "der."})'
    
    fig = make_chart(history, chart_title, current_norm_range)
    st.pyplot(fig)

    angles = [h[1] for h in history]
    max_angle = max(angles)

    # CÁLCULO DE VELOCIDAD ANGULAR MÁXIMA
    velocities = []
    for i in range(1, len(history)):
        dt = history[i][0] - history[i-1][0]
        if dt > 0.05: # Evitar división por cero o ruidos extremos
            d_theta = abs(history[i][1] - history[i-1][1])
            velocities.append(d_theta / dt)
    max_vel = max(velocities) if velocities else 0
    
    st.header("5. Resultados Clínicos")
    r1, r2, r3 = st.columns(3)
    r1.metric("Máximo Alcanzado", f"{max_angle:.1f}°") 
    
    if current_norm_range[0] <= max_angle <= current_norm_range[1] + 5: 
        r2.metric("Evaluación R.O.M.", "Dentro de rango", delta_color="normal")
    else:
        diff = max_angle - current_norm_range[1]
        r2.metric("Evaluación R.O.M.", f"Limitado por: {abs(diff):.1f}°", delta_color="inverse")
        
    r3.metric("Velocidad Máxima", f"{max_vel:.1f} °/s", help="Útil para evaluar control motor o espasticidad.")

    st.header("6. Registro de sesión")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Nombre del paciente")
    patient_run = p2.text_input("RUN", placeholder="12.345.678-9")
    test_date = p3.date_input("Fecha", value=date.today())

    if st.button("Guardar temporalmente"):
        st.session_state.sessions.append({
            "run": patient_run or "-", "patient": patient_name or "Sin nombre", "date": str(test_date),
            "joint": chart_title, "min": min(angles), "max": max(angles), "vel": max_vel,
            "mean": sum(angles) / len(angles), "norm_range_str": f"{current_norm_range[0]}°- {current_norm_range[1]}°"
        })
        st.success("Guardado en la sesión actual.")

if st.session_state.sessions:
    st.subheader("Historial (Se borrará al recargar la página)")
    sessions_df = pd.DataFrame([{
        "Paciente": s["patient"], "Fecha": s["date"], "Articulación": s["joint"],
        "Máximo": f"{s['max']:.1f}°", "Velocidad": f"{s['vel']:.1f} °/s"
    } for s in st.session_state.sessions])
    st.dataframe(sessions_df, use_container_width=True)