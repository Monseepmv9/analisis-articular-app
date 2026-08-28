"""
M.M. - MotionMetrics
=====================================================================
Detecta la pose de una persona en un video (usando MediaPipe) y calcula
el ángulo de la articulación y movimiento elegidos, con estandarización clínica AAOS.
"""

import io
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

# Modelo "Full" para máxima precisión al cruzar o superponer extremidades
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
    "hombro": {
        "flexion": (0, 180),
        "extension": (0, 60), 
        "abduccion": (0, 180),
        "aduccion": (0, 30),
        "rot_interna": (0, 70),
        "rot_externa": (0, 90)
    },
    "codo": {
        "flexion": (0, 150),
        "extension": (0, 10) 
    },
    "cadera": {
        "flexion": (0, 120), 
        "extension": (0, 30), 
        "abduccion": (0, 45), 
        "aduccion": (0, 30), 
        "rot_interna": (0, 45),
        "rot_externa": (0, 45)
    },
    "rodilla": {
        "flexion": (0, 135), 
        "extension": (0, 10) 
    },
    "tobillo": {
        "dorsiflexion": (0, 20),
        "plantiflexion": (0, 50)
    }
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
        {"id": "abduccion", "label": "Abducción", "view": "frontal", "mode": "angle_90_rest", "lm": {"left": [24, 23, 25], "right": [23, 24, 26]}},
        {"id": "aduccion", "label": "Aducción", "view": "frontal", "mode": "angle_90_rest", "lm": {"left": [24, 23, 25], "right": [23, 24, 26]}},
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
# 2. Geometría: cálculo de ángulos
# ----------------------------------------------------------------------------

def angle_between(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def angle_from_vertical(vertex, point):
    v = np.array([point[0] - vertex[0], point[1] - vertex[1]])
    mag = np.linalg.norm(v)
    if mag == 0:
        return None
    up = np.array([0, -1])
    cos_angle = np.clip(np.dot(v, up) / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def pick_main_person(pose_landmarks_list):
    if len(pose_landmarks_list) <= 1:
        return pose_landmarks_list[0] if pose_landmarks_list else None
    best, best_area = None, -1
    for lm in pose_landmarks_list:
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best, best_area = lm, area
    return best

# ----------------------------------------------------------------------------
# 3. Procesamiento de video con MediaPipe Pose
# ----------------------------------------------------------------------------

def download_model_if_needed():
    model_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
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

    if not result.pose_landmarks:
        return frame, None, False

    h, w = frame.shape[:2]
    lm = pick_main_person(result.pose_landmarks)

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
        
        if raw_angle is None:
            angle = None
        elif movement["mode"] == "angle_0_rest":
            angle = raw_angle
        elif movement["mode"] == "angle_180_rest":
            angle = abs(180.0 - raw_angle)
        elif movement["mode"] == "angle_90_rest":
            angle = abs(raw_angle - 90.0) 

    color = (60, 70, 226) if low_conf else (35, 93, 242)
    for p in pts_px:
        cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1)

    smoothed = None
    if angle is not None:
        smooth_buffer.append(angle)
        if len(smooth_buffer) > SMOOTH_WINDOW:
            smooth_buffer.pop(0)
        smoothed = sum(smooth_buffer) / len(smooth_buffer)
        
        label_num = f"{smoothed:.0f}"
        text_org = (int(vertex[0]) + 12, int(vertex[1]) - 12)
        
        cv2.putText(frame, label_num, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        (tw, th), _ = cv2.getTextSize(label_num, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        circle_org = (text_org[0] + tw + 6, text_org[1] - th + 4)
        cv2.circle(frame, circle_org, 4, color, 2, cv2.LINE_AA)

    return frame, smoothed, low_conf


if WEBRTC_AVAILABLE:
    class PoseVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.movement = None
            self.side = "left"
            self.history = []
            self.smooth_buffer = []
            self.last_ts = int(time.time() * 1000)
            self.start_time = None
            self.landmarker = create_landmarker()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.movement is not None:
                if self.start_time is None:
                    self.start_time = time.time()
                current_ms = int(time.time() * 1000)
                if current_ms <= self.last_ts:
                    current_ms = self.last_ts + 1
                self.last_ts = current_ms

                img, angle, low_conf = analyze_frame(
                    img, self.landmarker, self.movement, self.side, current_ms, self.smooth_buffer
                )
                if angle is not None:
                    t = time.time() - self.start_time
                    self.history.append((t, angle, low_conf))
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def process_video(video_path, movement, side, target_fps, preview_placeholder, progress_bar, progress_text):
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, round(video_fps / target_fps))

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_dim = 640
    scale = min(max_dim / orig_width, max_dim / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = out_tfile.name
    
    container = av.open(output_video_path, mode='w')
    stream = container.add_stream('h264', rate=int(video_fps))
    stream.width = new_width
    stream.height = new_height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '28'}

    history = []
    smooth_buffer = []
    frame_idx = 0

    preview_placeholder.info("⏳ Procesando análisis cinemático...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (new_width, new_height))
        current_ms = frame_idx * int(1000 / video_fps)

        frame_analyzed, angle, low_conf = analyze_frame(
            frame, landmarker, movement, side, current_ms, smooth_buffer
        )
        
        if frame_idx % frame_step == 0:
            t = frame_idx / video_fps
            if angle is not None:
                history.append((t, angle, low_conf))

        av_frame = av.VideoFrame.from_ndarray(frame_analyzed, format='bgr24')
        for packet in stream.encode(av_frame):
            container.mux(packet)

        if total_frames > 0 and frame_idx % 5 == 0:
            progress_bar.progress(min(1.0, frame_idx / total_frames))
            progress_text.caption(f"Analizando cuadro {frame_idx} de {total_frames}...")
                
        frame_idx += 1

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()
    
    try:
        landmarker.close()
    except Exception:
        pass
        
    progress_bar.progress(1.0)
    progress_text.empty()
    preview_placeholder.empty()
    
    st.success("✅ Procesamiento completado:")
    with open(output_video_path, 'rb') as f:
        st.video(f.read())
        
    try:
        os.remove(output_video_path)
    except Exception:
        pass

    return history


# ----------------------------------------------------------------------------
# 4. Gráficos y reportes PDF
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
    pdf.multi_cell(
        0, 6,
        f"Paciente: {last['patient']}   |   RUN: {last['run']}   |   "
        f"Generado: {date.today().isoformat()}   |   Nro. de pruebas: {len(sessions)}"
    )
    pdf.ln(4)

    for s in sessions:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(28, 43, 48)
        
        titulo_limpio = s['joint'].replace("—", "-").replace("\u2014", "-")
        pdf.cell(0, 9, f"{titulo_limpio} - {s['date']}", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(s["chart_png"])
            tmp_img_path = tmp_img.name

        pdf.image(tmp_img_path, w=150)
        pdf.ln(2)
        
        try:
            os.remove(tmp_img_path)
        except Exception:
            pass

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(225, 245, 238)
        pdf.cell(45, 8, "Mínimo", border=1, fill=True)
        pdf.cell(45, 8, "Máximo", border=1, fill=True)
        pdf.cell(45, 8, "Promedio", border=1, fill=True)
        pdf.cell(45, 8, "Rango Normal", border=1, fill=True, ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(45, 8, f"{s['min']:.1f}{chr(176)}", border=1)  
        pdf.cell(45, 8, f"{s['max']:.1f}{chr(176)}", border=1)
        pdf.cell(45, 8, f"{s['mean']:.1f}{chr(176)}", border=1)
        pdf.cell(45, 8, f"{s['norm_range_str']}", border=1, ln=True)
        pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 130, 135)
    
    pdf.multi_cell(
        0, 4.5,
        "Informe generado por estimacion de video 2D (MediaPipe Pose). "
        "No equivale a la precision de un sistema de sensores inerciales; "
        "corresponde a un screening funcional, no a una medicion clinica de referencia. "
        "Rangos de referencia basados en goniometria estandar (AAOS)."
    )

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)

# ----------------------------------------------------------------------------
# 5. Interfaz Streamlit
# ----------------------------------------------------------------------------

st.set_page_config(page_title="M.M. - MotionMetrics", page_icon="📐", layout="centered")

st.title("M.M. - MotionMetrics 📐")
st.caption("Sube un video, elige la articulacion, el movimiento y la vista de camara.")

if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "history" not in st.session_state:
    st.session_state.history = []

st.header("1. Configuracion de la prueba")
col1, col2 = st.columns(2)
with col1:
    body_part = st.selectbox(
        "Parte del cuerpo / articulacion",
        list(BODY_PART_LABELS.keys()),
        format_func=lambda k: BODY_PART_LABELS[k],
        index=4, 
    )
with col2:
    movement_options = MOVEMENTS[body_part]
    movement_id = st.selectbox(
        "Movimiento a evaluar",
        [m["id"] for m in movement_options],
        format_func=lambda mid: next(m["label"] for m in movement_options if m["id"] == mid),
    )
movement = next(m for m in movement_options if m["id"] == movement_id)
current_norm_range = NORMATIVE_RANGES[body_part][movement_id]

col3, col4 = st.columns(2)
with col3:
    side = st.selectbox("Lado", ["left", "right"], format_func=lambda s: "Izquierdo" if s == "left" else "Derecho")
with col4:
    default_view_idx = 0 if movement["view"] == "lateral" else 1
    camera_view = st.selectbox(
        "Vista de camara del video",
        ["lateral", "frontal"],
        format_func=lambda v: "Lateral (de perfil)" if v == "lateral" else "Frontal (de frente)",
        index=default_view_idx,
    )

if camera_view != movement["view"]:
    st.warning(
        f'"{movement["label"]}" necesita vista '
        f'{"frontal (de frente)" if movement["view"] == "frontal" else "lateral (de perfil)"} '
        "para que el angulo tenga sentido - ajusta la vista de camara arriba."
    )
elif movement["mode"] == "vertical":
    st.info(
        "Rotacion: graba de frente con el codo pegado al cuerpo y flectado a 90 grados. "
        "El angulo mide el antebrazo respecto a la vertical - es un proxy clinico."
    )

if camera_view == "lateral" and body_part in ["rodilla", "cadera", "tobillo"]:
    st.info("💡 **Tip Clínico:** Ubica al paciente de modo que la pierna a evaluar esté **más cerca de la cámara**. Esto evita que la pierna de apoyo confunda el seguimiento de la IA.")

st.info(f"📊 **Rango normal esperado (AAOS):** {current_norm_range[0]}° - {current_norm_range[1]}°")

target_fps = st.select_slider("Densidad de analisis", options=[5, 10, 15], value=10, format_func=lambda f: f"{f} cuadros/seg")

st.header("2. Fuente del video")
source_options = ["Subir un video", "Grabar en vivo desde la camara"] if WEBRTC_AVAILABLE else ["Subir un video"]
source = st.radio("Como quieres registrar la prueba?", source_options, horizontal=True)

preview_placeholder = st.empty()
progress_bar = st.progress(0.0)
progress_text = st.empty()

if source == "Subir un video":
    st.info("📱 **Si estás en un celular o tablet:** Al presionar el botón de abajo, tu dispositivo te dará la opción de abrir la cámara, grabar al paciente y subir el video automáticamente.")
    
    uploaded_file = st.file_uploader("Sube un video o graba directamente con tu cámara", type=["mp4", "mov", "avi", "mkv", "webm"])

    st.header("3. Analisis")
    analyze_disabled = uploaded_file is None or camera_view != movement["view"]
    run_analysis = st.button("Analizar video completo", disabled=analyze_disabled)

    if run_analysis and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Analizando video cuadro a cuadro..."):
            history = process_video(tmp_path, movement, side, target_fps, preview_placeholder, progress_bar, progress_text)
        st.session_state.history = history

        if not history:
            st.error("No se detecto a la persona en el video. Revisa encuadre, luz e iluminacion.")

elif WEBRTC_AVAILABLE:
    st.header("3. Analisis")
    if camera_view != movement["view"]:
        st.warning("Ajusta la vista de camara arriba antes de grabar, o el angulo no va a tener sentido.")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="pose-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.movement = movement
        ctx.video_processor.side = side

    st.caption('Dale permiso de camara al navegador, ubicate segun la vista requerida, haz el movimiento, y presiona "Stop" arriba cuando termines.')

    if ctx.video_processor and not ctx.state.playing and ctx.video_processor.history:
        if st.button("Usar esta grabacion"):
            st.session_state.history = ctx.video_processor.history
            st.success(f"Grabacion cargada - {len(st.session_state.history)} cuadros registrados.")

history = st.session_state.history

if history:
    st.header("4. Evolucion del angulo en el tiempo")
    
    chart_title = f'{BODY_PART_LABELS[body_part]} - {movement["label"]} ({"izq." if side == "left" else "der."})'
    
    fig = make_chart(history, chart_title, current_norm_range)
    st.pyplot(fig)
    st.caption("Puntos rojos = cuadros con baja confianza en la deteccion (posible oclusion).")

    angles = [h[1] for h in history]
    max_angle = max(angles)
    
    st.header("5. Resultados de la prueba")
    r1, r2, r3 = st.columns(3)
    r1.metric("Mínimo", f"{min(angles):.1f}°") 
    
    if current_norm_range[0] <= max_angle <= current_norm_range[1] + 5: 
        r2.metric("Máximo", f"{max_angle:.1f}°", "Dentro de rango", delta_color="normal")
    else:
        diff = max_angle - current_norm_range[1]
        r2.metric("Máximo", f"{max_angle:.1f}°", f"Diferencia: {diff:+.1f}°", delta_color="inverse")
        
    r3.metric("Promedio", f"{sum(angles) / len(angles):.1f}°")

    st.header("6. Paciente y registro de sesiones")
    p1, p2, p3 = st.columns(3)
    patient_name = p1.text_input("Nombre del paciente")
    patient_run = p2.text_input("RUN del paciente", placeholder="Ej: 12.345.678-9")
    test_date = p3.date_input("Fecha de la prueba", value=date.today())

    if st.button("Guardar esta prueba en el historial"):
        st.session_state.sessions.append({
            "patient": patient_name or "Sin nombre",
            "run": patient_run or "-",
            "date": str(test_date),
            "joint": chart_title,
            "min": min(angles),
            "max": max(angles),
            "mean": sum(angles) / len(angles),
            "norm_range_str": f"{current_norm_range[0]}°- {current_norm_range[1]}°",
            "chart_png": fig_to_png_bytes(make_chart(history, chart_title, current_norm_range)),
        })
        st.success("Prueba guardada en el historial de esta sesion.")

if st.session_state.sessions:
    st.subheader("Historial guardado (esta sesion)")
    sessions_df = pd.DataFrame([{
        "RUN": s["run"], "Paciente": s["patient"], "Fecha": s["date"], "Articulacion": s["joint"],
        "Mínimo °": round(s["min"], 1), "Máximo °": round(s["max"], 1), "Rango Esperado": s["norm_range_str"]
    } for s in st.session_state.sessions])
    st.dataframe(sessions_df, use_container_width=True)

    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        if st.button("Generar informe (PDF)"):
            pdf_bytes = build_pdf_report(st.session_state.sessions)
            last_patient = st.session_state.sessions[-1]["patient"]
            safe_filename = last_patient.replace(' ', '_').replace('-', '_')
            st.download_button("Descargar informe (PDF)", data=pdf_bytes, file_name=f"informe_{safe_filename}.pdf", mime="application/pdf")
    with col_xlsx:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            sessions_df.to_excel(writer, index=False, sheet_name="Pruebas")
        st.download_button("Descargar historial (Excel)", data=excel_buffer.getvalue(), file_name="historial_pruebas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")