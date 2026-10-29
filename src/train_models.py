import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_rep_classifier(csv_path, output_model_path, exercise_name="Exercise"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"==================================================")
    print(f"Training Inflection-Point Classifier for {exercise_name.upper()}")
    print(f"Total Rep Samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"==================================================")

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bad Form (0)", "Good Form (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, output_model_path)
    print(f"\nSaved trained model to: {output_model_path}\n")

if __name__ == '__main__':
    squat_csv = r"c:\Users\muthu\Downloads\Compressed\data\squat_rep_features.csv"
    squat_model_path = r"c:\Users\muthu\Downloads\Compressed\models\squat_form_classifier.joblib"
    if os.path.exists(squat_csv):
        train_rep_classifier(squat_csv, squat_model_path, "Squat")

    pushup_csv = r"c:\Users\muthu\Downloads\Compressed\data\pushup_rep_features.csv"
    pushup_model_path = r"c:\Users\muthu\Downloads\Compressed\models\pushup_form_classifier.joblib"
    if os.path.exists(pushup_csv):
        train_rep_classifier(pushup_csv, pushup_model_path, "Push-up")
