import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from src.feature_extraction import extract_landmark_features

mp_pose = mp.solutions.pose

def extract_features_from_video(video_path, pose_detector, sample_rate=3):
    """
    Extracts landmark feature vectors from sample frames of an exercise video.
    Returns a list of feature vectors for frames where pose was detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video {video_path}")
        return []

    frame_count = 0
    extracted_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            _, feature_vector = extract_landmark_features(results.pose_landmarks.landmark)
            extracted_features.append(feature_vector)

    cap.release()
    return extracted_features

def build_dataset_for_exercise(dataset_base_path, exercise='squat', max_videos_per_class=40, sample_rate=4):
    """
    Traverses the exercise-wise dataset for good and bad videos.
    Extracts features and builds a tabular dataset.
    Label: Good = 1, Bad = 0.
    """
    ex_path = os.path.join(dataset_base_path, "Dataset Exercise-wise", exercise)
    if not os.path.exists(ex_path):
        raise FileNotFoundError(f"Exercise path not found: {ex_path}")

    good_videos = glob.glob(os.path.join(ex_path, "good", "**", "*.mp4"), recursive=True)
    bad_videos = glob.glob(os.path.join(ex_path, "bad", "**", "*.mp4"), recursive=True)

    print(f"Found {len(good_videos)} good and {len(bad_videos)} bad videos for '{exercise}'.")

    # Sample to keep training rapid and balanced
    good_videos = good_videos[:max_videos_per_class]
    bad_videos = bad_videos[:max_videos_per_class]

    print(f"Processing {len(good_videos)} good and {len(bad_videos)} bad videos (sample rate {sample_rate})...")

    records = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
        # Process Good videos
        for idx, vpath in enumerate(good_videos):
            feats = extract_features_from_video(vpath, pose, sample_rate=sample_rate)
            for f in feats:
                records.append(np.append(f, 1))  # 1 = Good
            if (idx + 1) % 10 == 0 or idx == len(good_videos) - 1:
                print(f"  [Good] Processed {idx + 1}/{len(good_videos)} videos")

        # Process Bad videos
        for idx, vpath in enumerate(bad_videos):
            feats = extract_features_from_video(vpath, pose, sample_rate=sample_rate)
            for f in feats:
                records.append(np.append(f, 0))  # 0 = Bad
            if (idx + 1) % 10 == 0 or idx == len(bad_videos) - 1:
                print(f"  [Bad] Processed {idx + 1}/{len(bad_videos)} videos")

    feature_cols = [
        'knee_angle', 'hip_angle', 'elbow_angle', 'shoulder_angle', 'back_angle',
        'l_knee_angle', 'r_knee_angle', 'l_hip_angle', 'r_hip_angle',
        'l_elbow_angle', 'r_elbow_angle',
        'knee_ankle_diff_x', 'hip_knee_diff_y', 'label'
    ]

    df = pd.DataFrame(records, columns=feature_cols)
    print(f"Built dataset for '{exercise}': {len(df)} frames (Good: {sum(df['label'] == 1)}, Bad: {sum(df['label'] == 0)})")
    return df

if __name__ == '__main__':
    base_dir = r"C:\Users\muthu\Downloads\Compressed\A Multi-View Raw Video Dataset of Seven Fitness Ex"
    
    print("--- Extracting Squat Data ---")
    squat_df = build_dataset_for_exercise(base_dir, exercise='squat', max_videos_per_class=35, sample_rate=4)
    squat_df.to_csv(r"c:\Users\muthu\Downloads\Compressed\data\squat_features.csv", index=False)

    print("\n--- Extracting Push-up Data ---")
    pushup_df = build_dataset_for_exercise(base_dir, exercise='push_up', max_videos_per_class=35, sample_rate=4)
    pushup_df.to_csv(r"c:\Users\muthu\Downloads\Compressed\data\pushup_features.csv", index=False)

    print("\nFeature extraction completed successfully!")
