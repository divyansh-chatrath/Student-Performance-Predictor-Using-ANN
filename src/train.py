"""
EduPredict – Student Performance Predictor
==========================================
Training pipeline using scikit-learn MLPClassifier (Artificial Neural Network).
Exports model weights as JSON for TensorFlow.js reconstruction in the web app.

Compatible with Python 3.13+ (no TensorFlow required for training).

Run:
    python src/train.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
NUMERICAL_FEATURES    = ["Age", "Attendance", "Study_Hours", "Previous_Score",
                          "Assignment_Score", "Participation", "Sleep_Hours"]
CATEGORICAL_FEATURES  = ["Gender", "Internet_Access", "Extracurricular"]
TARGET                = "Performance"
PASSING_SCORE         = 40.0
RANDOM_SEED           = 42

# ANN Architecture
HIDDEN_LAYERS         = (64, 32)   # Two hidden layers
ACTIVATION            = "relu"
MAX_ITER              = 300
LEARNING_RATE_INIT    = 0.001
BATCH_SIZE            = 32


# ══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic Dataset Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(n: int = 2000, seed: int = RANDOM_SEED,
                     passing_score: float = PASSING_SCORE) -> pd.DataFrame:
    """
    Generate a realistic synthetic student performance dataset.
    Final_Score is computed from weighted academic + lifestyle factors
    plus Gaussian noise — NOT random alone.
    """
    rng = np.random.default_rng(seed)

    student_id       = [f"STU{str(i+1).zfill(4)}" for i in range(n)]
    age              = rng.integers(16, 26, n)
    gender           = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    attendance       = rng.uniform(40, 100, n).round(2)
    study_hours      = rng.uniform(0, 10,  n).round(2)
    previous_score   = rng.uniform(20, 100, n).round(2)
    assignment_score = rng.uniform(20, 100, n).round(2)
    participation    = rng.uniform(0, 100,  n).round(2)
    sleep_hours      = rng.uniform(3, 10,   n).round(2)
    internet_access  = rng.choice(["Yes", "No"], n, p=[0.75, 0.25])
    extracurricular  = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])

    # Compute final score from meaningful relationships
    internet_boost = np.where(internet_access == "Yes", 5.0, 0.0)
    extra_effect   = np.where(extracurricular  == "Yes", 3.0, -2.0)
    sleep_effect   = np.clip((sleep_hours - 6.0) * 2.0, -6.0, 6.0)

    raw_score = (
        0.30 * attendance
        + 0.25 * study_hours * 10       # study_hours (0-10) → (0-100) scale
        + 0.25 * previous_score
        + 0.10 * assignment_score
        + 0.05 * participation
        + internet_boost
        + extra_effect
        + sleep_effect
        + rng.normal(0, 6, n)           # Gaussian noise
    )
    final_score = np.clip(raw_score, 0, 100).round(2)
    performance = np.where(final_score >= passing_score, "Pass", "Fail")

    return pd.DataFrame({
        "Student_ID":       student_id,
        "Age":              age,
        "Gender":           gender,
        "Attendance":       attendance,
        "Study_Hours":      study_hours,
        "Previous_Score":   previous_score,
        "Assignment_Score": assignment_score,
        "Participation":    participation,
        "Sleep_Hours":      sleep_hours,
        "Internet_Access":  internet_access,
        "Extracurricular":  extracurricular,
        "Final_Score":      final_score,
        "Performance":      performance,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 2. Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def _feature_names(cat_categories: dict) -> list:
    names = list(NUMERICAL_FEATURES)
    for col, cats in cat_categories.items():
        for cat in cats:
            names.append(f"{col}_{cat}")
    return names


def preprocess(df: pd.DataFrame, fit: bool = True,
               scaler_params: dict = None,
               cat_categories: dict = None):
    """
    Standardise numerical features and one-hot encode categorical features.

    When fit=True: learns parameters from df and returns (X, y, params_dict).
    When fit=False: applies saved params and returns (X, y).
    """
    df = df.copy()

    # Target
    y = (df[TARGET] == "Pass").astype(np.float32).values

    # ── Numerical ───────────────────────────────────────────────
    num_data = df[NUMERICAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    fill_means = (pd.Series(scaler_params["means"], index=NUMERICAL_FEATURES)
                  if not fit else num_data.mean())
    num_data = num_data.fillna(fill_means)

    if fit:
        means = num_data.mean().values
        stds  = num_data.std().values
        stds  = np.where(stds == 0, 1.0, stds)
        scaler_params = {"means": means.tolist(), "stds": stds.tolist()}
    num_scaled = ((num_data.values - scaler_params["means"])
                  / scaler_params["stds"]).astype(np.float32)

    # ── Categorical (one-hot) ────────────────────────────────────
    cat_arrays = []
    if fit:
        cat_categories = {}
    for col in CATEGORICAL_FEATURES:
        col_data = df[col].fillna("Unknown")
        if fit:
            cats = sorted(col_data.unique().tolist())
            cat_categories[col] = cats
        else:
            cats = cat_categories[col]
        ohe = np.zeros((len(df), len(cats)), dtype=np.float32)
        for j, cat in enumerate(cats):
            ohe[:, j] = (col_data == cat).astype(float)
        cat_arrays.append(ohe)

    X = np.hstack([num_scaled] + cat_arrays)

    if fit:
        params = {
            "scaler":               scaler_params,
            "categories":           cat_categories,
            "numerical_features":   NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_names":        _feature_names(cat_categories),
            "n_features":           int(X.shape[1]),
            "passing_score":        PASSING_SCORE,
        }
        return X, y, params

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANN Model (scikit-learn MLPClassifier)
# ══════════════════════════════════════════════════════════════════════════════
def build_and_train_model(X_train, y_train):
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,   # (64, 32)
        activation=ACTIVATION,              # relu
        solver="adam",
        alpha=1e-4,                         # L2 regularisation
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        verbose=True,
    )
    model.fit(X_train, y_train.astype(int))
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Export model weights for TF.js reconstruction
# ══════════════════════════════════════════════════════════════════════════════
def export_model_weights(model, output_dir: Path, feature_names: list,
                          training_history: dict):
    """
    Export all layer weights and biases as JSON so the web app can
    reconstruct the model using TensorFlow.js tf.tensor().
    """
    layers = []
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        activation = "sigmoid" if i == len(model.coefs_) - 1 else ACTIVATION
        layers.append({
            "layer_index": i,
            "name":        f"layer_{i}",
            "activation":  activation,
            "weights":     W.T.tolist(),   # shape: [out, in] for TF.js Dense
            "biases":      b.tolist(),
            "in_features": int(W.shape[0]),
            "out_features": int(W.shape[1]),
        })

    architecture = {
        "type":           "MLPClassifier",
        "framework":      "scikit-learn",
        "hidden_layers":  list(HIDDEN_LAYERS),
        "activation":     ACTIVATION,
        "output_activation": "sigmoid",
        "n_features_in":  int(model.n_features_in_),
        "n_outputs":      1,
        "feature_names":  feature_names,
        "n_iter":         int(model.n_iter_),
        "best_loss":      float(model.best_loss_) if (hasattr(model, "best_loss_") and model.best_loss_ is not None) else None,
        "training_history": training_history,
    }

    model_json = {"architecture": architecture, "layers": layers}
    out_path = output_dir / "model_weights.json"
    with open(out_path, "w") as f:
        json.dump(model_json, f)          # no indent → smaller file
    print(f"  ✓ Model weights saved → {out_path} "
          f"({out_path.stat().st_size / 1024:.1f} KB)")
    return model_json


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   threshold: float = 0.5) -> dict:
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, roc_auc_score, accuracy_score)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    y_int = y_test.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_int, preds).ravel()

    return {
        "accuracy":         float(accuracy_score(y_int, preds)),
        "precision":        float(precision_score(y_int, preds, zero_division=0)),
        "recall":           float(recall_score(y_int, preds, zero_division=0)),
        "f1":               float(f1_score(y_int, preds, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_int, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold":        threshold,
        "n_test":           int(len(y_test)),
        "n_pass_actual":    int(y_int.sum()),
        "n_fail_actual":    int((1 - y_int).sum()),
        "n_pass_pred":      int(preds.sum()),
        "n_fail_pred":      int((1 - preds).sum()),
        "n_train":          int(len(y_test) * 4),  # approx from 80/20 split
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def main():
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  EduPredict – ANN Training Pipeline (scikit-learn)")
    print("=" * 60)

    # Step 1 – Generate data ──────────────────────────────────────
    print("\n[1/5] Generating synthetic dataset (2,000 records)…")
    df = generate_dataset(n=2000, seed=RANDOM_SEED)
    df.to_csv(DATA_DIR / "student_data.csv", index=False)
    df.head(20).to_csv(DATA_DIR / "student_data_sample.csv", index=False)
    print(f"  ✓ Saved  → {DATA_DIR / 'student_data.csv'}")
    vc = df["Performance"].value_counts()
    print(f"  Pass: {vc.get('Pass', 0)}  |  Fail: {vc.get('Fail', 0)}")

    # Step 2 – Preprocess ─────────────────────────────────────────
    print("\n[2/5] Preprocessing (StandardScaler + OneHot)…")
    X, y, params = preprocess(df, fit=True)

    idx = list(range(len(df)))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

    print(f"  ✓ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  ✓ Features ({X.shape[1]}): {params['feature_names']}")

    with open(MODEL_DIR / "preprocessor.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  ✓ Preprocessor → {MODEL_DIR / 'preprocessor.json'}")

    # Step 3 – Train ──────────────────────────────────────────────
    print("\n[3/5] Training ANN (MLPClassifier)…")
    print(f"  Architecture: Input({X_train.shape[1]}) → Dense(64,ReLU) "
          f"→ Dense(32,ReLU) → Dense(1,Sigmoid)")
    model = build_and_train_model(X_train, y_train)
    print(f"  ✓ Converged in {model.n_iter_} iterations")

    # Reconstruct training history from loss_curve_
    history = {}
    if hasattr(model, "loss_curve_"):
        history["loss"] = [float(v) for v in model.loss_curve_]
    if hasattr(model, "validation_scores_"):
        history["val_accuracy"] = [float(v) for v in model.validation_scores_]
    if hasattr(model, "best_validation_score_"):
        history["best_val_accuracy"] = float(model.best_validation_score_)

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Training history → {MODEL_DIR / 'training_history.json'}")

    # Step 4 – Export weights ──────────────────────────────────────
    print("\n[4/5] Exporting model weights for TF.js…")
    export_model_weights(model, MODEL_DIR, params["feature_names"], history)

    # Step 5 – Evaluate ────────────────────────────────────────────
    print("\n[5/5] Evaluating on hold-out test set…")
    metrics = evaluate_model(model, X_test, y_test)
    metrics["n_train"] = int(X_train.shape[0])

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion  : TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics → {MODEL_DIR / 'metrics.json'}")

    # Save test predictions for export
    proba_test = model.predict_proba(X_test)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)
    test_df = df.iloc[idx_test].copy().reset_index(drop=True)
    test_df["Predicted_Performance"] = np.where(preds_test == 1, "Pass", "Fail")
    test_df["Pass_Probability"]      = proba_test.round(4)
    test_df["Fail_Probability"]      = (1 - proba_test).round(4)
    test_df["Correct"] = (test_df["Performance"] == test_df["Predicted_Performance"])
    test_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    print(f"  ✓ Test predictions → {DATA_DIR / 'test_predictions.csv'}")

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete!")
    print(f"     Model  → {MODEL_DIR}/model_weights.json")
    print(f"     Data   → {DATA_DIR}/student_data.csv")
    print(f"     Open   app/index.html in a browser to use the dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
