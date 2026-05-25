import argparse
import hashlib
import logging
import warnings
from copy import deepcopy

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
MODEL_PATH = "diabetes_model.pkl"
RANDOM_STATE = 42

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COL = "Outcome"


def compute_dataset_hash(url: str) -> str:
    df = pd.read_csv(url)
    raw = df.to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    df["Glucose_Risk"] = pd.cut(
        df["Glucose"], bins=[0, 100, 126, 200, 300], labels=[0, 1, 2, 3]
    ).astype(int)

    df["Insulin_Glucose"] = df["Insulin"] * df["Glucose"] / 1000.0
    df["BMI_Age"] = df["BMI"] * df["Age"] / 100.0
    df["Insulin_Resistance"] = df["Glucose"] * df["Insulin"] / 405.0

    return df


def find_best_threshold(model, X_val, y_val):
    probas = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.2, 0.8, 61)
    best_f1, best_th = 0, 0.5
    for th in thresholds:
        preds = (probas >= th).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def build_models():
    return {
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            "grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6, 10],
                "model__learning_rate": [0.01, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__scale_pos_weight": [1, 2],
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                max_iter=1000,
            ),
            "grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        },
    }


def main(args):
    logger.info("=== Diabetes Prediction — Production Training Pipeline ===")
    np.random.seed(args.seed)

    # Data ingestion
    logger.info("Loading data from %s", DATA_URL)
    df = pd.read_csv(DATA_URL)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Class distribution:\n%s", df[TARGET_COL].value_counts())

    data_hash = compute_dataset_hash(DATA_URL)
    logger.info("Dataset hash: %s", data_hash)

    # Missing value imputation
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)
    logger.info("Missing values:\n%s", df.isnull().sum())

    imputer = KNNImputer(n_neighbors=5)
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    logger.info("Missing after imputation:\n%s", df.isnull().sum())

    # Feature engineering
    df = engineer_features(df)

    eng_features = FEATURE_COLS + [
        "BMI_Category", "Age_Group", "Glucose_Risk",
        "Insulin_Glucose", "BMI_Age", "Insulin_Resistance",
    ]
    logger.info("Engineered features: %s", eng_features)

    X = df[eng_features]
    y = df[TARGET_COL]

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        random_state=args.seed, stratify=y_train,
    )
    logger.info(
        "Splits — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # SMOTE oversampling
    smote = SMOTE(random_state=args.seed)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info("After SMOTE — train: %d", len(X_train_res))
    logger.info("Class dist: %s", y_train_res.value_counts().to_dict())

    # Multi-model comparison
    scaler = StandardScaler()
    models = build_models()

    best_overall_score = 0
    best_overall_model = None
    best_overall_name = ""
    best_overall_threshold = 0.5
    results = []

    mlflow.set_experiment(args.experiment)

    for name, cfg in models.items():
        logger.info("=" * 50)
        logger.info("Training: %s", name)

        pipeline = Pipeline([("scaler", scaler), ("model", cfg["model"])])

        gs = GridSearchCV(
            pipeline, cfg["grid"], cv=5, scoring="f1", n_jobs=-1, verbose=0,
        )
        gs.fit(X_train_res, y_train_res)

        best_pipeline = gs.best_estimator_
        logger.info("Best params: %s", gs.best_params_)

        # Threshold tuning on validation
        base_preds = best_pipeline.predict(X_val)
        base_f1 = f1_score(y_val, base_preds)

        best_th, tuned_f1 = find_best_threshold(best_pipeline, X_val, y_val)
        logger.info(
            "Val F1 — base (0.5): %.4f, tuned (%.2f): %.4f",
            base_f1, best_th, tuned_f1,
        )

        # Test set evaluation
        test_probas = best_pipeline.predict_proba(X_test)[:, 1]
        test_preds = (test_probas >= best_th).astype(int)

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds)
        rec = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        roc_auc = roc_auc_score(y_test, test_probas)

        logger.info(
            "Test — Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %.4f",
            acc, prec, rec, f1, roc_auc,
        )
        logger.info(
            "Confusion Matrix:\n%s", confusion_matrix(y_test, test_preds)
        )

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "threshold": best_th,
            "val_f1_tuned": tuned_f1,
            "params": gs.best_params_,
        })

        # MLflow logging
        with mlflow.start_run(run_name=name):
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics({
                "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "roc_auc": roc_auc, "threshold": best_th,
                "val_f1_tuned": tuned_f1,
            })
            mlflow.log_param("model_name", name)
            mlflow.log_param("data_hash", data_hash)
            mlflow.log_param("feature_count", len(eng_features))
            mlflow.log_param("seed", args.seed)
            mlflow.sklearn.log_model(
                best_pipeline, "model",
                input_example=X_test.iloc[:1].to_dict(orient="records"),
            )

        if f1 > best_overall_score:
            best_overall_score = f1
            best_overall_model = deepcopy(best_pipeline)
            best_overall_name = name
            best_overall_threshold = best_th

    logger.info("=" * 50)
    logger.info(
        "Best single model: %s (F1: %.4f, threshold: %.2f)",
        best_overall_name, best_overall_score, best_overall_threshold,
    )

    # Ensemble voting classifier
    logger.info("Training ensemble...")
    estimators = []
    for name, cfg in models.items():
        pipe = Pipeline([("scaler", scaler), ("model", cfg["model"])])
        gs = GridSearchCV(
            pipe, cfg["grid"], cv=3, scoring="f1", n_jobs=-1,
        )
        gs.fit(X_train_res, y_train_res)
        estimators.append((name, gs.best_estimator_))

    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train_res, y_train_res)

    ens_th, _ = find_best_threshold(ensemble, X_val, y_val)
    ens_probas = ensemble.predict_proba(X_test)[:, 1]
    ens_preds = (ens_probas >= ens_th).astype(int)

    ens_acc = accuracy_score(y_test, ens_preds)
    ens_f1 = f1_score(y_test, ens_preds)
    ens_auc = roc_auc_score(y_test, ens_probas)

    logger.info(
        "Ensemble — Acc: %.4f, F1: %.4f, AUC: %.4f (th: %.2f)",
        ens_acc, ens_f1, ens_auc, ens_th,
    )

    if ens_f1 > best_overall_score:
        best_overall_score = ens_f1
        best_overall_model = ensemble
        best_overall_name = "Ensemble"
        best_overall_threshold = ens_th
        logger.info("Ensemble is the best model!")

    # SHAP analysis
    logger.info("Computing SHAP values...")
    shap_expected = None
    try:
        import shap as shap_lib

        clf = None
        if hasattr(best_overall_model, "named_steps"):
            clf = best_overall_model.named_steps.get("model")
        elif hasattr(best_overall_model, "estimators_"):
            clf = best_overall_model.estimators_[0]

        if clf is not None:
            explainer = shap_lib.TreeExplainer(clf)
            shap_expected = explainer.expected_value
            logger.info("SHAP expected_value: %s", shap_expected)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)

    # Look up best model results
    best_result = None
    for r in results:
        if r["model"] == best_overall_name:
            best_result = r
            break
    if best_result is None and best_overall_name == "Ensemble":
        best_result = {
            "accuracy": ens_acc,
            "f1": ens_f1,
            "roc_auc": ens_auc,
        }

    # --- Package model with metadata ---
    def coerce_shap(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, (list, np.ndarray)):
            return float(val[1]) if len(val) > 1 else float(val[0])
        return None

    metadata = {
        "model_name": best_overall_name,
        "classification_threshold": best_overall_threshold,
        "features": eng_features,
        "base_features": FEATURE_COLS,
        "data_hash": data_hash,
        "seed": args.seed,
        "dataset_url": DATA_URL,
        "test_metrics": {
            "accuracy": best_result["accuracy"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"],
        },
        "shap_expected": coerce_shap(shap_expected),
        "all_results": results,
    }

    model_package = {"model": best_overall_model, "metadata": metadata}
    joblib.dump(model_package, MODEL_PATH)
    logger.info("Model + metadata saved to %s", MODEL_PATH)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  %-20s  Acc: %.4f  F1: %.4f  AUC: %.4f  Th: %.2f",
            r["model"], r["accuracy"], r["f1"], r["roc_auc"], r["threshold"],
        )
    logger.info(
        "  %-20s  Acc: %.4f  F1: %.4f  AUC: %.4f  Th: %.2f",
        "Ensemble", ens_acc, ens_f1, ens_auc, ens_th,
    )
    logger.info("  ---")
    logger.info(
        "  Best: %s (F1: %.4f, threshold: %.2f)",
        best_overall_name, best_overall_score, best_overall_threshold,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--experiment", type=str, default="diabetes-prediction")
    args = parser.parse_args()
    main(args)
