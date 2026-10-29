import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from src.feature_extraction import extract_landmark_features

mp_pose = mp.solutions.pose

def extract_inflection_features_from_video(video_path, exercise='squat'):
    """
    Simulates exercise tracking to find the deepest/inflection frame of each repetition,
    extracting features precisely at the moment form matters (bottom of squat, bottom of push-up).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    rep_features = []
    stage = 'up'
    min_angle = 180.0
    best_feat = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)

            if not res.pose_landmarks:
                continue

            fdict, fvec = extract_landmark_features(res.pose_landmarks.landmark)

            if exercise == 'squat':
                ang = fdict['knee_angle']
                if stage == 'down':
                    if ang < min_angle:
                        min_angle = ang
                        best_feat = fvec
                if ang < 115:
                    stage = 'down'
                    if ang < min_angle:
                        min_angle = ang
                        best_feat = fvec
                elif ang > 150:
                    if stage == 'down' and best_feat is not None:
                        rep_features.append(best_feat)
                        min_angle = 180.0
                        best_feat = None
                    stage = 'up'

            elif exercise == 'push_up':
                ang = fdict['elbow_angle']
                if stage == 'down':
                    if ang < min_angle:
                        min_angle = ang
                        best_feat = fvec
                if ang < 110:
                    stage = 'down'
                    if ang < min_angle:
                        min_angle = ang
                        best_feat = fvec
                elif ang > 145:
                    if stage == 'down' and best_feat is not None:
                        rep_features.append(best_feat)
                        min_angle = 180.0
                        best_feat = None
                    stage = 'up'

    cap.release()
    return rep_features

def build_rep_dataset(base_dir, exercise='squat', max_videos_per_class=65):
    ex_path = os.path.join(base_dir, "Dataset Exercise-wise", exercise)
    good_videos = glob.glob(os.path.join(ex_path, "good", "**", "*.mp4"), recursive=True)[:max_videos_per_class]
    bad_videos = glob.glob(os.path.join(ex_path, "bad", "**", "*.mp4"), recursive=True)[:max_videos_per_class]

    print(f"Building inflection-point dataset for {exercise} ({len(good_videos)} good, {len(bad_videos)} bad)...")

    records = []
    for idx, v in enumerate(good_videos):
        feats = extract_inflection_features_from_video(v, exercise=exercise)
        for f in feats:
            records.append(np.append(f, 1))

    for idx, v in enumerate(bad_videos):
        feats = extract_inflection_features_from_video(v, exercise=exercise)
        for f in feats:
            records.append(np.append(f, 0))

    feature_cols = [
        'knee_angle', 'hip_angle', 'elbow_angle', 'shoulder_angle', 'back_angle',
        'l_knee_angle', 'r_knee_angle', 'l_hip_angle', 'r_hip_angle',
        'l_elbow_angle', 'r_elbow_angle',
        'knee_ankle_diff_x', 'hip_knee_diff_y', 'label'
    ]

    df = pd.DataFrame(records, columns=feature_cols)
    print(f"{exercise.upper()} Rep Dataset: {len(df)} reps (Good: {sum(df['label']==1)}, Bad: {sum(df['label']==0)})")
    return df

if __name__ == '__main__':
    base_dir = r"C:\Users\muthu\Downloads\Compressed\A Multi-View Raw Video Dataset of Seven Fitness Ex"
    
    squat_df = build_rep_dataset(base_dir, 'squat', max_videos_per_class=70)
    squat_df.to_csv(r"c:\Users\muthu\Downloads\Compressed\data\squat_rep_features.csv", index=False)

    pushup_df = build_rep_dataset(base_dir, 'push_up', max_videos_per_class=70)
    pushup_df.to_csv(r"c:\Users\muthu\Downloads\Compressed\data\pushup_rep_features.csv", index=False)
    print("Rep feature extraction complete!")
