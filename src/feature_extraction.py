import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the 2D angle between three points (a, b, c) with b as the vertex.
    a, b, c are each [x, y] coordinates.
    Returns angle in degrees (0 to 180).
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))

def extract_landmark_features(landmarks):
    """
    Given a list of 33 MediaPipe pose landmarks (with x, y, z, visibility),
    extract key joint angles and normalized posture metrics useful for squats and push-ups.
    Returns a dict with key features and a flat 1D numpy feature vector.
    """
    # Helper to get (x, y)
    def pt(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    # Helper to get (x, y, z)
    def pt3(idx):
        return [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    # Left side
    l_shoulder = pt(11)
    l_elbow = pt(13)
    l_wrist = pt(15)
    l_hip = pt(23)
    l_knee = pt(25)
    l_ankle = pt(27)

    # Right side
    r_shoulder = pt(12)
    r_elbow = pt(14)
    r_wrist = pt(16)
    r_hip = pt(24)
    r_knee = pt(26)
    r_ankle = pt(28)

    # Angles
    l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    knee_angle = (l_knee_angle + r_knee_angle) / 2.0

    l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    hip_angle = (l_hip_angle + r_hip_angle) / 2.0

    l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0

    l_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)
    r_shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
    shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0

    # Back / trunk alignment: angle between shoulder, hip, and ankle
    l_back_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
    r_back_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
    back_angle = (l_back_angle + r_back_angle) / 2.0

    # Knee vs ankle horizontal alignment (for knee cave or over-forward knee tracking)
    knee_ankle_diff_x = abs((landmarks[25].x + landmarks[26].x)/2.0 - (landmarks[27].x + landmarks[28].x)/2.0)
    hip_knee_diff_y = ((landmarks[25].y + landmarks[26].y)/2.0) - ((landmarks[23].y + landmarks[24].y)/2.0)

    # Feature vector: standard set of angles + relative geometric offsets
    feature_dict = {
        'knee_angle': knee_angle,
        'hip_angle': hip_angle,
        'elbow_angle': elbow_angle,
        'shoulder_angle': shoulder_angle,
        'back_angle': back_angle,
        'l_knee_angle': l_knee_angle,
        'r_knee_angle': r_knee_angle,
        'l_hip_angle': l_hip_angle,
        'r_hip_angle': r_hip_angle,
        'l_elbow_angle': l_elbow_angle,
        'r_elbow_angle': r_elbow_angle,
        'knee_ankle_diff_x': knee_ankle_diff_x,
        'hip_knee_diff_y': hip_knee_diff_y
    }

    feature_vector = np.array([
        knee_angle, hip_angle, elbow_angle, shoulder_angle, back_angle,
        l_knee_angle, r_knee_angle, l_hip_angle, r_hip_angle,
        l_elbow_angle, r_elbow_angle,
        knee_ankle_diff_x, hip_knee_diff_y
    ], dtype=np.float32)

    return feature_dict, feature_vector
