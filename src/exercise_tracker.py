import cv2
import numpy as np
from src.feature_extraction import calculate_angle

class ExerciseTracker:
    def __init__(self, exercise_type='squat'):
        """
        exercise_type: 'squat' or 'push_up'
        """
        self.exercise_type = exercise_type
        self.counter = 0
        self.stage = None  # 'up' or 'down'
        self.good_reps = 0
        self.bad_reps = 0
        self.current_feedback = []
        self.history = []  # records per rep: {rep_id, form, issues}
        self.active_issues_in_rep = set()
        self.min_angle_in_rep = 180.0
        self.max_angle_in_rep = 0.0
        self.best_inflection_feat = None

    def reset(self):
        self.counter = 0
        self.stage = None
        self.good_reps = 0
        self.bad_reps = 0
        self.current_feedback = []
        self.history = []
        self.active_issues_in_rep.clear()
        self.min_angle_in_rep = 180.0
        self.max_angle_in_rep = 0.0
        self.best_inflection_feat = None

    def process_frame(self, landmarks, ml_model=None, feature_vector=None):
        """
        Processes pose landmarks for rep counting and form assessment.
        landmarks: MediaPipe pose landmarks list (33 points)
        """
        if not landmarks:
            return {
                'rep_count': self.counter,
                'stage': self.stage,
                'form': 'No pose detected',
                'feedback': ['Please stay in full view of the camera'],
                'good_reps': self.good_reps,
                'bad_reps': self.bad_reps
            }

        def p(idx):
            return [landmarks[idx].x, landmarks[idx].y]

        l_sh = p(11)
        r_sh = p(12)
        l_el = p(13)
        r_el = p(14)
        l_wr = p(15)
        r_wr = p(16)
        l_hip = p(23)
        r_hip = p(24)
        l_kn = p(25)
        r_kn = p(26)
        l_an = p(27)
        r_an = p(28)

        # Average angles across both body sides
        l_knee_angle = calculate_angle(l_hip, l_kn, l_an)
        r_knee_angle = calculate_angle(r_hip, r_kn, r_an)
        knee_angle = (l_knee_angle + r_knee_angle) / 2.0

        l_hip_angle = calculate_angle(l_sh, l_hip, l_kn)
        r_hip_angle = calculate_angle(r_sh, r_hip, r_kn)
        hip_angle = (l_hip_angle + r_hip_angle) / 2.0

        l_elbow_angle = calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = calculate_angle(r_sh, r_el, r_wr)
        elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0

        # Back alignment (shoulder - hip - ankle)
        l_body_align = calculate_angle(l_sh, l_hip, l_an)
        r_body_align = calculate_angle(r_sh, r_hip, r_an)
        body_alignment = (l_body_align + r_body_align) / 2.0

        live_issues = []
        is_rep_completed = False
        rep_verdict = None

        if self.exercise_type == 'squat':
            # Save the feature vector at the deepest inflection point of the rep
            if self.stage == 'down':
                if knee_angle < self.min_angle_in_rep:
                    self.min_angle_in_rep = knee_angle
                    self.best_inflection_feat = feature_vector

            # Natural biomechanics checks (relaxed thresholds to avoid false positives)
            # 1. Extreme back hunch or bent almost completely in half
            if hip_angle < 40 and knee_angle > 110:
                live_issues.append("Chest dropped too low; keep spine neutral")
                self.active_issues_in_rep.add("Excessive forward hunch")

            # 2. Extreme knee collapse inward (valgus)
            knee_spread = abs(landmarks[25].x - landmarks[26].x)
            ankle_spread = abs(landmarks[27].x - landmarks[28].x)
            if ankle_spread > 0.12 and knee_spread < ankle_spread * 0.50:
                live_issues.append("Watch knee alignment; avoid knees caving inward")
                self.active_issues_in_rep.add("Knees caving inward (valgus)")

            # Rep State Machine:
            # Stand: knee > 155 deg
            # Down inflection: knee < 110 deg
            if knee_angle > 150:
                if self.stage == 'down':
                    self.counter += 1
                    is_rep_completed = True

                    # Check depth: squat depth should achieve around ~105 deg or lower
                    if self.min_angle_in_rep > 115:
                        self.active_issues_in_rep.add("Shallow squat depth (aim to descend below 105°)")

                    # Query ML Classifier on the inflection point features
                    ml_pred = None
                    target_feat = self.best_inflection_feat if self.best_inflection_feat is not None else feature_vector
                    if ml_model is not None and target_feat is not None:
                        try:
                            # 1 = Good, 0 = Bad
                            pred = ml_model.predict([target_feat])[0]
                            ml_pred = "Good" if pred == 1 else "Bad"
                        except Exception:
                            ml_pred = None

                    # If no obvious biomechanical defect occurred, trust Good form
                    if len(self.active_issues_in_rep) == 0:
                        rep_verdict = "Good"
                        self.good_reps += 1
                    elif ml_pred == "Good" and len(self.active_issues_in_rep) <= 1:
                        # ML recognizes good execution despite slight camera view divergence
                        rep_verdict = "Good"
                        self.good_reps += 1
                    else:
                        rep_verdict = "Bad"
                        self.bad_reps += 1

                    self.history.append({
                        'rep': self.counter,
                        'verdict': rep_verdict,
                        'min_angle': round(self.min_angle_in_rep, 1),
                        'issues': list(self.active_issues_in_rep)
                    })
                    self.active_issues_in_rep.clear()
                    self.min_angle_in_rep = 180.0
                    self.best_inflection_feat = None

                self.stage = 'up'

            elif knee_angle < 110:
                self.stage = 'down'

        elif self.exercise_type == 'push_up':
            # Save the feature vector at the deepest push-up inflection point
            if self.stage == 'down':
                if elbow_angle < self.min_angle_in_rep:
                    self.min_angle_in_rep = elbow_angle
                    self.best_inflection_feat = feature_vector

            # Push-up form checks:
            # Body alignment (sagging hips or excessive pike)
            if body_alignment < 135:
                live_issues.append("Keep torso straight; hips sagging or piking")
                self.active_issues_in_rep.add("Hips sagging or piking")

            # Rep State Machine:
            # UP: elbow > 145 deg
            # DOWN: elbow < 105 deg
            if elbow_angle > 145:
                if self.stage == 'down':
                    self.counter += 1
                    is_rep_completed = True

                    if self.min_angle_in_rep > 115:
                        self.active_issues_in_rep.add("Push-up depth shallow (descend to ~90° elbow)")

                    ml_pred = None
                    target_feat = self.best_inflection_feat if self.best_inflection_feat is not None else feature_vector
                    if ml_model is not None and target_feat is not None:
                        try:
                            pred = ml_model.predict([target_feat])[0]
                            ml_pred = "Good" if pred == 1 else "Bad"
                        except Exception:
                            ml_pred = None

                    if len(self.active_issues_in_rep) == 0:
                        rep_verdict = "Good"
                        self.good_reps += 1
                    elif ml_pred == "Good" and len(self.active_issues_in_rep) <= 1:
                        rep_verdict = "Good"
                        self.good_reps += 1
                    else:
                        rep_verdict = "Bad"
                        self.bad_reps += 1

                    self.history.append({
                        'rep': self.counter,
                        'verdict': rep_verdict,
                        'min_angle': round(self.min_angle_in_rep, 1),
                        'issues': list(self.active_issues_in_rep)
                    })
                    self.active_issues_in_rep.clear()
                    self.min_angle_in_rep = 180.0
                    self.best_inflection_feat = None

                self.stage = 'up'

            elif elbow_angle < 105:
                self.stage = 'down'

        # Current live feedback
        self.current_feedback = live_issues if live_issues else ["Good form! Keep going."]

        # Real-time state verdict
        form_status = "Good" if len(live_issues) == 0 else "Needs Adjustment"
        primary_angle = knee_angle if self.exercise_type == 'squat' else elbow_angle

        return {
            'rep_count': self.counter,
            'stage': self.stage,
            'form': form_status,
            'feedback': self.current_feedback,
            'primary_angle': round(primary_angle, 1),
            'good_reps': self.good_reps,
            'bad_reps': self.bad_reps,
            'is_rep_completed': is_rep_completed,
            'rep_verdict': rep_verdict,
            'history': self.history
        }
