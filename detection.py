import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import threading
import uuid
import ollama
import requests   

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Smart Surveillance",
    layout="wide"
)

st.title("AI Smart Surveillance System  ")

# =========================================================
# DIRECTORY
# =========================================================

SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# TELEGRAM ALERT SETUP (ADDED)
# =========================================================

BOT_TOKEN = "8794579001:AAFJvoUNKGM7A6IAx0gRWBHPRQ5ue7Y8tX0"
CHAT_ID = "5185724799"

def send_telegram_alert(label):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        message = f"🚨 ALERT: {label} detected in AI Surveillance System"

        payload = {
            "chat_id": CHAT_ID,
            "text": message
        }

        requests.post(url, data=payload)

    except Exception as e:
        print("Telegram Error:", e)

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Control Panel")

mode = st.sidebar.selectbox("Select Mode", ["Image", "Video", "Webcam"])

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

target = st.sidebar.text_input("Alert Object", "person")

class_filter = st.sidebar.text_input("Filter Classes", "")

language = st.sidebar.selectbox("Voice Language", ["English", "Tamil", "Hindi"])

cooldown = st.sidebar.slider("Alert Cooldown", 1, 10, 3)

save_snap = st.sidebar.checkbox("Save Snapshot", True)

# =========================================================
# SESSION STATE
# =========================================================

if "log" not in st.session_state:
    st.session_state.log = []

if "last_alert" not in st.session_state:
    st.session_state.last_alert = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "run_video" not in st.session_state:
    st.session_state.run_video = False

if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

# =========================================================
# LANGUAGE MAP
# =========================================================

lang_map = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi"
}

# =========================================================
# SPEAK
# =========================================================

def speak(text, lang="en"):
    try:
        filename = f"voice_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filename)
        playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print("Voice Error:", e)

# =========================================================
# ALERT (UPDATED WITH TELEGRAM)
# =========================================================

def alert(frame, label):

    now = time.time()

    if now - st.session_state.last_alert < cooldown:
        return

    st.session_state.last_alert = now

    st.error(f"ALERT: {label}")

    if save_snap:
        path = f"{SAVE_DIR}/{label}_{int(now)}.jpg"
        cv2.imwrite(path, frame)

    text_map = {
        "English": f"{label} detected",
        "Tamil": f"{label} கண்டறியப்பட்டது",
        "Hindi": f"{label} पाया गया"
    }

    threading.Thread(
        target=speak,
        args=(text_map[language], lang_map[language]),
        daemon=True
    ).start()

    #  TELEGRAM ALERT ADDED
    threading.Thread(
        target=send_telegram_alert,
        args=(label,),
        daemon=True
    ).start()

    st.session_state.log.append({
        "time": datetime.now(),
        "object": label
    })

# =========================================================
# PROCESS FRAME
# =========================================================

def process(frame):

    results = model(frame, conf=confidence)

    boxes = results[0].boxes
    names = results[0].names

    counts = {}

    allowed = None
    if class_filter:
        allowed = [c.strip().lower() for c in class_filter.split(",")]

    for box in boxes:

        cls = int(box.cls[0])
        label = names[cls]

        if allowed and label.lower() not in allowed:
            continue

        counts[label] = counts.get(label, 0) + 1

        if label == target:
            alert(frame, label)

    annotated = results[0].plot()

    return annotated, counts

# =========================================================
# DASHBOARD
# =========================================================

col1, col2, col3 = st.columns(3)
col1.metric("Mode", mode)
col2.metric("Target", target)
col3.metric("Confidence", confidence)

st.markdown("---")

# =========================================================
# IMAGE MODE
# =========================================================

if mode == "Image":

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        annotated, counts = process(img)

        colA, colB = st.columns([3, 1])

        colA.image(annotated, channels="BGR", use_container_width=True)

        with colB:
            st.subheader("Detected Objects")
            st.bar_chart(counts)

# =========================================================
# VIDEO MODE
# =========================================================

elif mode == "Video":

    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:

        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶ Start"):
                st.session_state.run_video = True

        with col2:
            if st.button("⛔ Stop"):
                st.session_state.run_video = False

        if st.session_state.run_video:

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())

            cap = cv2.VideoCapture(tfile.name)

            frame_box = st.empty()

            prev = time.time()

            while cap.isOpened() and st.session_state.run_video:

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))

                annotated, _ = process(frame)

                fps = 1 / (time.time() - prev + 1e-5)
                prev = time.time()

                cv2.putText(
                    annotated,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                frame_box.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )

                time.sleep(0.03)

            cap.release()
            st.session_state.run_video = False

# =========================================================
# WEBCAM MODE
# =========================================================

elif mode == "Webcam":

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.run_webcam = True

    with col2:
        if st.button("⛔ Stop Camera"):
            st.session_state.run_webcam = False

    frame_box = st.empty()

    cap = cv2.VideoCapture(0)

    prev = time.time()

    while cap.isOpened() and st.session_state.run_webcam:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not detected")
            break

        frame = cv2.resize(frame, (640, 480))

        annotated, _ = process(frame)

        fps = 1 / (time.time() - prev + 1e-5)
        prev = time.time()

        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        frame_box.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

        time.sleep(0.03)

    cap.release()

# =========================================================
# ANALYTICS
# =========================================================

st.markdown("---")
st.subheader("Analytics Dashboard")

if st.session_state.log:

    df = pd.DataFrame(st.session_state.log)

    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Object Frequency")
        st.bar_chart(df["object"].value_counts())

    with col2:
        st.subheader("Time Trend")
        df["time"] = pd.to_datetime(df["time"])
        st.line_chart(df.groupby(pd.Grouper(key="time", freq="1min")).size())

    st.download_button("Download CSV", df.to_csv(index=False), "data.csv")

else:
    st.info("No detections yet")

# =========================================================
# AI ASSISTANT
# =========================================================

st.markdown("---")
st.subheader("AI Assistant")

user_input = st.text_input("Ask AI Assistant")

if st.button("Ask AI") and user_input:

    detection_data = ""

    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        detection_data = f"""
Total detections: {len(df)}
Object counts: {df['object'].value_counts().to_dict()}
"""

    prompt = f"""
You are a surveillance AI assistant.

Detection Data:
{detection_data}

User Question:
{user_input}
"""

    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

        st.success(answer)

        threading.Thread(
            target=speak,
            args=(answer, lang_map[language]),
            daemon=True
        ).start()

    except Exception as e:
        st.error(f"AI Error: {e}")