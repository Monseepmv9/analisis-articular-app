"""
M.M. - MotionMetrics
=====================================================================
Análisis cinemático 2D (MediaPipe). Soporta análisis simultáneo de
múltiples articulaciones, recorte de video y cálculo de velocidad.
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
    "hombro": "Hombro", "codo": "Codo", "cadera": "Cadera", 
    "rodilla": "Rodilla", "tobillo": "Tobillo",
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
    ],
    "cadera": [
        {"id": "flexion", "label": "Flexión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
        {"id": "extension", "label": "Extensión", "view": "lateral", "mode": "angle_180_rest", "lm": {"left": [11, 23, 25], "right": [12, 24, 26]}},
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

        a, b, c = pts_px
        vertex = b
        raw_angle = angle_between(a, b, c)
        angle = None
        if raw_angle is not None:
            if movement["mode"] == "angle_0_rest": angle = raw_angle
            elif movement["mode"] == "angle_180_rest": angle = abs(180.0 - raw_angle)
            elif movement["mode"] == "angle_90_rest": angle = abs(raw_angle - 90.0) 

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

def process_video_multijoint(video_path, configs, target_fps, preview_placeholder, progress_bar, start_sec, end_sec):
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_step = max(1, round(video_fps / target_fps))

    start_frame, end_frame = int(start_sec * video_fps), int(end_sec * video_fps)
    total_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(640 / orig_w, 640 / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    new_w, new_h = new_w - (new_w % 2), new_h - (new_h % 2)
    
    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = out_tfile.name
    
    container = av.open(output_video_path, mode='w')
    stream = container.add_stream('h264', rate=int(video_fps))
    stream.width, stream.height, stream.pix_fmt = new_w, new_h, 'yuv420p'

    history = []
    frames_processed = 0
    preview_placeholder.info("⏳ Procesando análisis cinemático...")

    while frames_processed <= total_frames:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (new_w, new_h))
        current_ms = int((start_frame + frames_processed) * (1000 / video_fps))

        frame_analyzed, angles_out, low_conf = analyze_frame_multijoint(frame, landmarker, configs, current_ms)
        
        if frames_processed % frame_step == 0:
            t = (start_frame + frames_processed) / video_fps
            if all(a is not None for a in angles_out):
                history.append((t, angles_out, low_conf))

        av_frame = av.VideoFrame.from_ndarray(frame_analyzed, format='bgr24')
        for packet in stream.encode(av_frame): container.mux(packet)
        if total_frames > 0 and frames_processed % 5 == 0:
            progress_bar.progress(min(1.0, frames_processed / total_frames))
        frames_processed += 1

    for packet in stream.encode(): container.mux(packet)
    container.close()
    cap.release()
    try: landmarker.close()
    except Exception: pass
        
    progress_bar.empty()
    preview_placeholder.empty()
    with open(output_video_path, 'rb') as f: st.video(f.read())
    try: os.remove(output_video_path)
    except Exception: pass

    return history

def make_multijoint_chart(history, titles, norm_ranges, colors):
    times = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for idx, title in enumerate(titles):
        angles = [h[1][idx] for h in history]
        ax.plot(times, angles, color=colors[idx], linewidth=2, label=title)
        if norm_ranges[idx]:
            ax.axhspan(norm_ranges[idx][0], norm_ranges[idx][1], color=colors[idx], alpha=0.1)

    ax.set_ylim(0, 180)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Ángulo (°)") 
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig

# ----------------------------------------------------------------------------
# 5. UI Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(page_title="M.M. - MotionMetrics", page_icon="📐", layout="centered")
st.title("M.M. - MotionMetrics 📐")

st.header("1. Configuración Principal")
col1, col2 = st.columns(2)
with col1:
    bp1 = st.selectbox("Articulación Principal", list(BODY_PART_LABELS.keys()), format_func=lambda k: BODY_PART_LABELS[k], index=2)
with col2:
    mov_opts1 = MOVEMENTS.get(bp1, [])
    mov_id1 = st.selectbox("Movimiento 1", [m["id"] for m in mov_opts1], format_func=lambda mid: next(m["label"] for m in mov_opts1 if m["id"] == mid))

st.header("2. Configuración Secundaria (Simultánea)")
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
progress_bar = st.progress(0.0)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap_temp = cv2.VideoCapture(tmp_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_temp = cap_temp.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps_temp
    cap_temp.release()

    start_sec, end_sec = st.slider("Recortar segmento", 0.0, float(duration), (0.0, float(duration)), step=0.1)

    if st.button("Procesar Análisis Combinado"):
        m1 = next(m for m in mov_opts1 if m["id"] == mov_id1)
        configs = [{"mov": m1, "side": side, "buffer": [], "color": (255, 120, 30)}] # Azul
        titles = [f"{BODY_PART_LABELS[bp1]} - {m1['label']}"]
        norm_ranges = [NORMATIVE_RANGES[bp1][mov_id1]]
        colors_hex = ["#1e78ff"]

        if bp2 != "Ninguna":
            m2 = next(m for m in mov_opts2 if m["id"] == mov_id2)
            configs.append({"mov": m2, "side": side, "buffer": [], "color": (50, 205, 50)}) # Verde
            titles.append(f"{BODY_PART_LABELS[bp2]} - {m2['label']}")
            norm_ranges.append(NORMATIVE_RANGES[bp2][mov_id2])
            colors_hex.append("#32cd32")

        with st.spinner("Procesando cinemática..."):
            history = process_video_multijoint(tmp_path, configs, target_fps, preview_placeholder, progress_bar, start_sec, end_sec)
            
            if history:
                st.subheader("Evolución Temporal Coordinada")
                fig = make_multijoint_chart(history, titles, norm_ranges, colors_hex)
                st.pyplot(fig)
                
                for idx, title in enumerate(titles):
                    st.write(f"**Resultados: {title}**")
                    angles = [h[1][idx] for h in history]
                    vels = []
                    for i in range(1, len(history)):
                        dt = history[i][0] - history[i-1][0]
                        if dt > 0.05: vels.append(abs(history[i][1][idx] - history[i-1][1][idx]) / dt)
                    max_vel = max(vels) if vels else 0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Máximo Alcanzado", f"{max(angles):.1f}°")
                    c2.metric("Rango Clínico", f"{norm_ranges[idx][0]}° - {norm_ranges[idx][1]}°")
                    c3.metric("Velocidad Máxima", f"{max_vel:.1f} °/s")
                    st.divider()

        try: os.remove(tmp_path)
        except Exception: pass