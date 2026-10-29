import os
import sys

# Ensure repository root is in sys.path for cloud deployment
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import mediapipe as mp
try:
    from src.feature_extraction import extract_landmark_features
    from src.exercise_tracker import ExerciseTracker
except ImportError:
    from feature_extraction import extract_landmark_features
    from exercise_tracker import ExerciseTracker

# Setup Streamlit page configuration
st.set_page_config(
    page_title="AI Fitness Trainer | Real-Time Pose Correction",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .feedback-box {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .feedback-good {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .feedback-bad {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

# Cache ML model loading
@st.cache_resource
def load_models():
    squat_model_path = os.path.join(ROOT_DIR, "models", "squat_form_classifier.joblib")
    pushup_model_path = os.path.join(ROOT_DIR, "models", "pushup_form_classifier.joblib")
    
    models = {}
    if os.path.exists(squat_model_path):
        models['squat'] = joblib.load(squat_model_path)
    else:
        models['squat'] = None

    if os.path.exists(pushup_model_path):
        models['push_up'] = joblib.load(pushup_model_path)
    else:
        models['push_up'] = None
        
    return models

models = load_models()

# Sidebar Settings
st.sidebar.title("⚙️ Trainer Settings")
exercise_type = st.sidebar.selectbox(
    "Choose Exercise",
    ["Squat", "Push-up"],
    index=0
)
exercise_key = "squat" if exercise_type == "Squat" else "push_up"

input_mode = st.sidebar.radio(
    "Input Source",
    ["Webcam", "IP / Phone Camera (Wi-Fi)", "Sample Dataset Video", "Upload Video"],
    index=0
)

# Configuration for IP / Phone Camera
ip_camera_url = None
if input_mode == "IP / Phone Camera (Wi-Fi)":
    st.sidebar.markdown("""
    **📱 Phone Camera Setup:**
    1. Install **IP Webcam** (Android) or **Iriun / DroidCam / EpocCam** (iOS/Android).
    2. Ensure phone & PC are on the **same Wi-Fi**.
    3. Start server in the app and paste the streaming URL below (e.g., `http://192.168.1.10:8080/video`).
    """)
    ip_camera_url = st.sidebar.text_input(
        "Phone Stream URL",
        value="http://10.182.190.189:8080/video",
        help="Use /video or /video.mjpg endpoint from your IP Webcam app"
    )

# Select dataset sample if selected
sample_video_path = None
if input_mode == "Sample Dataset Video":
    dataset_folder = os.path.join(
        r"C:\Users\muthu\Downloads\Compressed\A Multi-View Raw Video Dataset of Seven Fitness Ex\Dataset Exercise-wise",
        exercise_key
    )
    form_choice = st.sidebar.selectbox("Exercise Form Quality", ["good", "bad"])
    view_choice = st.sidebar.selectbox("Camera View", ["front", "side", "diagonal"])
    
    v_dir = os.path.join(dataset_folder, form_choice, view_choice)
    if os.path.exists(v_dir):
        available_vids = [f for f in os.listdir(v_dir) if f.endswith('.mp4')]
        if available_vids:
            selected_file = st.sidebar.selectbox("Select Video", available_vids)
            sample_video_path = os.path.join(v_dir, selected_file)

confidence_thresh = st.sidebar.slider("Pose Detection Confidence", 0.3, 0.9, 0.5, 0.05)
enable_ml = st.sidebar.checkbox("Enable ML Classifier Assessment", value=True)

# Main layout header
st.markdown('<div class="main-title">🏋️ AI Fitness Trainer & Real-Time Pose Correction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated repetition counting and form quality assessment powered by MediaPipe, OpenCV & Scikit-Learn.</div>', unsafe_allow_html=True)

# Initialize Session State
if "tracker" not in st.session_state or st.session_state.get("current_exercise") != exercise_key:
    st.session_state.tracker = ExerciseTracker(exercise_type=exercise_key)
    st.session_state.current_exercise = exercise_key

tracker = st.session_state.tracker

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    rep_placeholder = st.empty()
    rep_placeholder.metric(label="Total Reps", value=tracker.counter)
with col2:
    good_placeholder = st.empty()
    good_placeholder.metric(label="✅ Good Reps", value=tracker.good_reps)
with col3:
    bad_placeholder = st.empty()
    bad_placeholder.metric(label="⚠️ Bad Reps", value=tracker.bad_reps)
with col4:
    stage_placeholder = st.empty()
    stage_placeholder.metric(label="Current Stage", value=str(tracker.stage).upper() if tracker.stage else "STANDBY")

# Feedback & Angle display
feed_col, angle_col = st.columns([3, 1])
with feed_col:
    feedback_placeholder = st.empty()
with angle_col:
    angle_placeholder = st.empty()

# Video feed column and Session Summary
video_col, summary_col = st.columns([3, 2])

with summary_col:
    st.subheader("📊 Session Rep History")
    history_placeholder = st.empty()
    if st.button("🔄 Reset Tracker"):
        tracker.reset()
        st.rerun()

with video_col:
    start_session = st.toggle("Start Camera / Video Processing", value=False)
    frame_placeholder = st.empty()

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

active_model = models.get(exercise_key) if enable_ml else None

if start_session:
    # Determine video source
    if input_mode == "Webcam":
        cap = cv2.VideoCapture(0)
    elif input_mode == "IP / Phone Camera (Wi-Fi)":
        if ip_camera_url:
            clean_url = ip_camera_url.strip()
            if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
                clean_url = "http://" + clean_url
            if not clean_url.endswith("/video") and not clean_url.endswith("/video.mjpg") and not clean_url.endswith(".mjpg"):
                clean_url = clean_url.rstrip("/") + "/video"
            cap = cv2.VideoCapture(clean_url)
        else:
            st.error("Please enter a valid Phone Camera stream URL.")
            cap = None
    elif input_mode == "Sample Dataset Video":
        if sample_video_path and os.path.exists(sample_video_path):
            cap = cv2.VideoCapture(sample_video_path)
        else:
            st.error("Sample video path not found.")
            cap = None
    elif input_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload an MP4 or MOV video", type=['mp4', 'mov', 'avi'])
        if uploaded_file is not None:
            temp_path = os.path.join(r"data", f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_path)
        else:
            cap = None
    else:
        cap = None

    if cap is not None and cap.isOpened():
        with mp_pose.Pose(
            min_detection_confidence=confidence_thresh,
            min_tracking_confidence=confidence_thresh,
            model_complexity=1
        ) as pose:
            while cap.isOpened() and start_session:
                ret, frame = cap.read()
                if not ret:
                    if input_mode != "Webcam":
                        # Loop video for continuous playback
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        break

                # Flip horizontally for natural mirror feel if webcam
                if input_mode == "Webcam":
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                feature_dict = None
                feature_vec = None

                if results.pose_landmarks:
                    # Draw Pose Skeleton
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    feature_dict, feature_vec = extract_landmark_features(results.pose_landmarks.landmark)

                # Process exercise tracker logic
                metrics = tracker.process_frame(
                    results.pose_landmarks.landmark if results.pose_landmarks else None,
                    ml_model=active_model,
                    feature_vector=feature_vec
                )

                # Update Streamlit Real-Time Dashboard UI
                rep_placeholder.metric(label="Total Reps", value=metrics['rep_count'])
                good_placeholder.metric(label="✅ Good Reps", value=metrics['good_reps'])
                bad_placeholder.metric(label="⚠️ Bad Reps", value=metrics['bad_reps'])
                stage_placeholder.metric(label="Current Stage", value=str(metrics['stage']).upper() if metrics['stage'] else "READY")

                # Feedback render
                is_good = (metrics['form'] == 'Good')
                box_class = "feedback-good" if is_good else "feedback-bad"
                fb_text = " | ".join(metrics['feedback'])
                feedback_placeholder.markdown(
                    f'<div class="feedback-box {box_class}"><b>Form:</b> {metrics["form"]} — {fb_text}</div>',
                    unsafe_allow_html=True
                )

                primary_ang = metrics.get('primary_angle', '-')
                angle_label = "Knee Angle" if exercise_key == "squat" else "Elbow Angle"
                angle_placeholder.metric(label=f"🎯 {angle_label}", value=f"{primary_ang}°")

                # Rep History Table
                if tracker.history:
                    history_df = pd.DataFrame(tracker.history)[['rep', 'verdict', 'min_angle', 'issues']]
                    history_placeholder.dataframe(history_df.tail(6), use_container_width=True)

                # Overlay HUD directly on the video frame
                cv2.rectangle(rgb_frame, (0, 0), (320, 80), (15, 23, 42), -1)
                hud_text = f"{exercise_type.upper()} | REPS: {metrics['rep_count']} | STAGE: {str(metrics['stage']).upper()}"
                cv2.putText(rgb_frame, hud_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                verdict_color = (34, 197, 94) if is_good else (239, 68, 68)
                cv2.putText(rgb_frame, f"STATUS: {metrics['form']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
