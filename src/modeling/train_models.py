import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "modeling"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports" / "model_evaluation"

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

TARGET_COLUMN = "target_value"

TIME_COLUMN = "target_datetime_utc"

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "lag_1h",
    "lag_3h",
    "lag_6h",
    "lag_24h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "hour",
    "day_of_week",
    "month"
]

CATEGORICAL_FEATURES = [
    "city",
    "sensor_id"
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def create_models():

    return {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42
        )
    }


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def create_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "mae": mean_absolute_error(y_true, y_pred),

        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),

        "r2": r2_score(y_true, y_pred)
    }


# --------------------------------------------------
# TRAIN ONE POLLUTANT
# --------------------------------------------------

def train_pollutant(pollutant):

    print("\n" + "=" * 60)
    print(f"TRAINING POLLUTANT: {pollutant.upper()}")
    print("=" * 60)

    pollutant_dir = DATA_DIR / pollutant

    train_file = pollutant_dir / "train.csv"
    test_file = pollutant_dir / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Train/test files missing for {pollutant}"
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Validate required columns.
    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, TIME_COLUMN]
    )

    for column in required_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Missing column {column} in {train_file}"
            )

        if column not in test_df.columns:
            raise ValueError(
                f"Missing column {column} in {test_file}"
            )

    # Parse timestamps.
    train_df[TIME_COLUMN] = pd.to_datetime(
        train_df[TIME_COLUMN],
        utc=True
    )

    test_df[TIME_COLUMN] = pd.to_datetime(
        test_df[TIME_COLUMN],
        utc=True
    )

    # Verify chronological split.
    if train_df[TIME_COLUMN].max() >= test_df[TIME_COLUMN].min():
        raise ValueError(
            f"Temporal leakage detected for {pollutant}"
        )

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

    y_train = train_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Categorical identifiers must be strings.
    for column in CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].astype(str)
        X_test[column] = X_test[column].astype(str)

    results = []
    
    predictions = pd.DataFrame({
        TIME_COLUMN: test_df[TIME_COLUMN].values,
        "city": test_df["city"].values,
        "sensor_id": test_df["sensor_id"].values,
        "actual_value": y_test.values
    })


    # --------------------------------------------------
    # PERSISTENCE BASELINE
    # --------------------------------------------------

    print("\nEvaluating persistence baseline...")

    baseline_predictions = X_test["lag_1h"].astype(float)

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    results.append({
        "pollutant": pollutant,
        "model": "persistence_baseline",
        **baseline_metrics
    })

    predictions["persistence_baseline"] = (
        baseline_predictions.to_numpy()
    )

    print(
        f"Persistence MAE: {baseline_metrics['mae']:.4f}"
    )

    # --------------------------------------------------
    # TRAIN MACHINE LEARNING MODELS
    # --------------------------------------------------

    models = create_models()

    for model_name, estimator in models.items():

        print(f"\nTraining {model_name}...")

        # Linear Regression benefits from scaling.
        scale_numeric = (
            model_name == "linear_regression"
        )

        preprocessor = create_preprocessor(
            scale_numeric=scale_numeric
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        # Fit ONLY on training data.
        pipeline.fit(X_train, y_train)

        # Predict on unseen future data.
        y_pred = pipeline.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        results.append({
            "pollutant": pollutant,
            "model": model_name,
            **metrics
        })

        predictions[model_name] = y_pred

        # Save trained pipeline.
        save_dir = MODEL_DIR / pollutant
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{model_name}.joblib"

        joblib.dump(pipeline, model_path)

        print("MAE:", round(metrics["mae"], 4))
        print("RMSE:", round(metrics["rmse"], 4))
        print("R2:", round(metrics["r2"], 4))
        print("Saved:", model_path)

    # Save predictions.
    prediction_dir = REPORT_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        prediction_dir / f"{pollutant}_predictions.csv",
        index=False
    )

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nAIRSIGHT AI — MACHINE LEARNING TRAINING")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pollutant in POLLUTANTS:

        results = train_pollutant(pollutant)

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    # Sort by pollutant and RMSE.
    results_df = results_df.sort_values(
        ["pollutant", "rmse"]
    )

    metrics_file = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(metrics_file, index=False)

    # Select best model per pollutant by RMSE.
    best_models = (
        results_df
        .sort_values("rmse")
        .groupby("pollutant")
        .first()
        .reset_index()
    )

    best_file = REPORT_DIR / "best_models.csv"

    best_models.to_csv(best_file, index=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\nBEST MODEL PER POLLUTANT")
    print(
        best_models[
            ["pollutant", "model", "mae", "rmse", "r2"]
        ].to_string(index=False)
    )

    print("\nSaved comparison:", metrics_file)
    print("Saved best models:", best_file)

    print("\nAll model training experiments completed!")


if __name__ == "__main__":
    main()
