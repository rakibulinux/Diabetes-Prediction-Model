import logging
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
MODEL_PATH = "diabetes_model.pkl"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv(DATA_URL)
logger.info("Columns: %s", df.columns.tolist())
logger.info("Shape: %s", df.shape)
logger.info("Class distribution:\n%s", df["Outcome"].value_counts())

df[["Glucose", "BloodPressure", "BMI"]] = df[
    ["Glucose", "BloodPressure", "BMI"]
].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
target_col = "Outcome"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [5, 10, None],
    "clf__min_samples_split": [2, 5],
}

mlflow.set_experiment("diabetes-prediction")

with mlflow.start_run() as run:
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    logger.info("Best params: %s", grid.best_params_)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1")

    logger.info("Accuracy:  %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall:    %.4f", rec)
    logger.info("F1 Score:  %.4f", f1)
    logger.info("ROC-AUC:   %.4f", roc_auc)
    logger.info("CV F1:     %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cv_f1_mean": cv_scores.mean(),
    })
    input_example = X_test.iloc[:1]
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        best_model, "model", signature=signature, input_example=input_example
    )

    joblib.dump(best_model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("MLflow run ID: %s", run.info.run_id)
