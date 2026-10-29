# AI Fitness Trainer with Real-Time Pose Correction

An AI-powered computer vision application that monitors squats and push-ups via webcam or video feeds, automatically counts repetitions, and provides real-time form assessment and corrective feedback.

## Features
- **Real-Time Pose Estimation**: Uses MediaPipe Pose (33 body landmarks) to accurately track joints and full-body posture.
- **Biomechanical Repetition Tracking**: Calculates joint angles (knee, hip, elbow, and torso alignment) using vector trigonometry to detect exercise inflection phases (UP / DOWN) and track rep completions.
- **Hybrid Form Assessment**:
  - **Rule-Based Engine**: Real-time geometric checks for knee valgus/cave, shallow depth, torso rounding/excessive lean, and hip sagging/piking.
  - **Machine Learning Classification**: Scikit-Learn `RandomForestClassifier` pipelines trained on the Multi-View Fitness Video Dataset (`good` vs `bad` execution) achieving >80% accuracy.
- **Interactive Streamlit Dashboard**:
  - Live webcam feed with pose skeleton overlay and HUD status.
  - Test mode with sample dataset videos across front, side, and diagonal angles.
  - Repetition counter, Good vs Bad rep breakdown, current inflection stage, and live feedback banners.
  - Historical table of completed repetitions with verdicts and issue breakdowns.

## Project Structure
```
├── data/
│   ├── squat_features.csv            # Extracted pose landmark feature dataset for squats
│   └── pushup_features.csv           # Extracted pose landmark feature dataset for push-ups
├── models/
│   ├── squat_form_classifier.joblib  # Trained scikit-learn classifier pipeline for squats
│   └── pushup_form_classifier.joblib # Trained scikit-learn classifier pipeline for push-ups
├── src/
│   ├── feature_extraction.py         # Joint angle calculations and pose feature vector extraction
│   ├── exercise_tracker.py           # Rep counting state machine and rule-based form evaluation
│   ├── train_dataset_builder.py      # Video dataset processor and feature extractor
│   ├── train_models.py               # ML training and evaluation script
│   └── app.py                        # Streamlit real-time interactive UI dashboard
├── run_app.bat                       # One-click Windows launch script
└── requirements.txt                  # Python dependencies
```

## How to Run

1. **Activate Environment & Run Streamlit**:
   Double-click `run_app.bat` or run the following in terminal:
   ```powershell
   .\.venv\Scripts\streamlit.exe run src\app.py
   ```
2. **In the Web App**:
   - Choose your exercise (**Squat** or **Push-up**).
   - Select your input source: **Webcam**, **Sample Dataset Video** (to test against dataset angles), or **Upload Video**.
   - Toggle **Start Camera / Video Processing** to begin tracking reps in real time.
