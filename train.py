import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


DATA = "diabetes.csv"
TARGET = "diabetes"


# ==========================================
# Load Data
# ==========================================

def load_data(path=DATA):

    df = pd.read_csv(path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


# ==========================================
# Preprocessing
# ==========================================

def build_preprocessor():

    categorical_features = [
        "gender",
        "smoking_history"
    ]

    numerical_features = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "blood_glucose_level",
        "HbA1c_level"
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                categorical_features
            ),
            (
                "num",
                numerical_transformer,
                numerical_features
            )
        ]
    )

    return preprocessor


# ==========================================
# Model Evaluation
# ==========================================

def model_evaluation(model, X_test, y_test):

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        prediction
    )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    X, y = load_data()

    print("Dataset shape:", X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # ======================================
    # Train/Test Split
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ======================================
    # Preprocessor
    # ======================================

    preprocessor = build_preprocessor()

    # ======================================
    # Models
    # ======================================

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    # ======================================
    # Training
    # ======================================

    best_model = None
    best_score = 0
    best_model_name = ""

    results = []

    print("\n" + "=" * 65)
    print("Training Models...")
    print("=" * 65)

    for name, algorithm in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", algorithm)
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            prediction
        ) = model_evaluation(
            pipeline,
            X_test,
            y_test
        )

        print("\n" + name)
        print("-" * 40)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                prediction
            )
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

        # Choose based on F1
        if f1 > best_score:

            best_score = f1
            best_model = pipeline
            best_model_name = name

    # ======================================
    # Results
    # ======================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.sort_values(
            "F1",
            ascending=False
        ).to_string(index=False)
    )

    # ======================================
    # Save Complete Pipeline
    # ======================================

    joblib.dump(
        best_model,
        "diabetes_model.pkl"
    )

    print("\n" + "=" * 65)
    print("BEST MODEL")
    print("=" * 65)

    print("Model:", best_model_name)

    print("\nModel saved as:")
    print("diabetes_model.pkl")