from pathlib import Path

import numpy as np
import pandas as pd

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
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "modeling"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "model_evaluation"
)

VALIDATION_PREDICTIONS_DIR = (
    REPORT_DIR / "validation_predictions"
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3"
]

TARGET_COLUMN = "target_value"
TIME_COLUMN = "target_datetime_utc"

FIT_RATIO = 0.80

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
# 2. MODEL DEFINITIONS
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
# 3. PREPROCESSING
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
# 4. CHRONOLOGICAL VALIDATION SPLIT
# --------------------------------------------------

def split_validation_data(df, pollutant):

    df = df.sort_values(TIME_COLUMN).copy()

    unique_times = (
        df[TIME_COLUMN]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 5:
        raise ValueError(
            f"Insufficient timestamps for {pollutant}"
        )

    split_index = int(
        len(unique_times) * FIT_RATIO
    )

    if split_index <= 0 or split_index >= len(unique_times):
        raise ValueError(
            f"Invalid validation split for {pollutant}"
        )

    cutoff_time = unique_times.iloc[split_index]

    fit_df = df[
        df[TIME_COLUMN] < cutoff_time
    ].copy()

    validation_df = df[
        df[TIME_COLUMN] >= cutoff_time
    ].copy()

    if fit_df.empty or validation_df.empty:
        raise ValueError(
            f"Empty partition for {pollutant}"
        )

    if fit_df[TIME_COLUMN].max() >= validation_df[TIME_COLUMN].min():
        raise ValueError(
            f"Temporal leakage detected for {pollutant}"
        )

    return fit_df, validation_df, cutoff_time


# --------------------------------------------------
# 5. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        raise ValueError("No valid predictions available.")

    return {
        "samples": len(y_true),

        "mae": mean_absolute_error(
            y_true, y_pred
        ),

        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),

        "r2": (
            r2_score(y_true, y_pred)
            if len(y_true) >= 2 and np.var(y_true) > 0
            else np.nan
        )
    }


# --------------------------------------------------
# 6. VALIDATE ONE POLLUTANT
# --------------------------------------------------

def validate_pollutant(pollutant):

    print("\n" + "=" * 65)
    print(f"VALIDATING: {pollutant.upper()}")
    print("=" * 65)

    pollutant_dir = DATA_DIR / pollutant
    train_file = pollutant_dir / "train.csv"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    df = pd.read_csv(train_file)

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, TIME_COLUMN]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{pollutant}: Missing columns: {missing}"
        )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        utc=True,
        errors="coerce"
    )

    if df[TIME_COLUMN].isna().any():
        raise ValueError(
            f"Invalid timestamps in {pollutant}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing target values in {pollutant}"
        )

    # Split using unique timestamps.
    fit_df, validation_df, cutoff_time = (
        split_validation_data(df, pollutant)
    )

    print("Total rows:", len(df))
    print("Model fitting rows:", len(fit_df))
    print("Validation rows:", len(validation_df))
    print("Fit period ends:", fit_df[TIME_COLUMN].max())
    print("Validation starts:", validation_df[TIME_COLUMN].min())
    print("Cutoff:", cutoff_time)

    X_fit = fit_df[FEATURE_COLUMNS].copy()
    X_validation = validation_df[FEATURE_COLUMNS].copy()

    y_fit = fit_df[TARGET_COLUMN].astype(float)
    y_validation = validation_df[TARGET_COLUMN].astype(float)

    # Keep categorical IDs consistent.
    for column in CATEGORICAL_FEATURES:

        X_fit[column] = X_fit[column].astype(str)

        X_validation[column] = (
            X_validation[column].astype(str)
        )

    results = []

    predictions = pd.DataFrame({
        TIME_COLUMN: validation_df[TIME_COLUMN].values,
        "city": validation_df["city"].values,
        "sensor_id": validation_df["sensor_id"].values,
        "actual_value": y_validation.values
    })

    # ----------------------------------------------
    # PERSISTENCE BASELINE
    # ----------------------------------------------

    print("\nEvaluating persistence baseline...")

    baseline_predictions = (
        X_validation["lag_1h"].astype(float)
    )

    baseline_metrics = calculate_metrics(
        y_validation,
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

    # ----------------------------------------------
    # MACHINE LEARNING MODELS
    # ----------------------------------------------

    for model_name, estimator in create_models().items():

        print(f"\nTraining {model_name}...")

        preprocessor = create_preprocessor(
            scale_numeric=(
                model_name == "linear_regression"
            )
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        # Fit only on the earlier period.
        pipeline.fit(X_fit, y_fit)

        # Predict on the later validation period.
        y_pred = pipeline.predict(X_validation)

        metrics = calculate_metrics(
            y_validation,
            y_pred
        )

        results.append({
            "pollutant": pollutant,
            "model": model_name,
            **metrics
        })

        predictions[model_name] = y_pred

        print("MAE:", round(metrics["mae"], 4))
        print("RMSE:", round(metrics["rmse"], 4))
        print("R2:", round(metrics["r2"], 4))

    # Save validation predictions.
    VALIDATION_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        VALIDATION_PREDICTIONS_DIR
        / f"{pollutant}_validation_predictions.csv",
        index=False
    )

    return results


# --------------------------------------------------
# 7. SELECT HYBRID MODEL
# --------------------------------------------------


def select_hybrid_models(results_df):

    selected_models = []

    for pollutant in POLLUTANTS:

        pollutant_results = results_df[
            results_df["pollutant"] == pollutant
        ].copy()

        baseline = pollutant_results[
            pollutant_results["model"]
            == "persistence_baseline"
        ]

        if baseline.empty:
            raise ValueError(
                f"Persistence baseline missing for {pollutant}"
            )

        baseline = baseline.iloc[0]

        # An ML candidate must improve both metrics.
        ml_candidates = pollutant_results[
            pollutant_results["model"]
            != "persistence_baseline"
        ].copy()

        qualified = ml_candidates[
            (ml_candidates["mae"] < baseline["mae"])
            & (ml_candidates["rmse"] < baseline["rmse"])
        ]

        if qualified.empty:

            chosen = baseline.copy()

            reason = (
                "Persistence selected: no ML model "
                "improved both MAE and RMSE"
            )

        else:

            chosen = qualified.sort_values(
                ["rmse", "mae"]
            ).iloc[0].copy()

            reason = (
                "ML selected: improved both MAE and RMSE "
                "against persistence"
            )

        selected_models.append({
            "pollutant": pollutant,
            "model": chosen["model"],
            "samples": chosen["samples"],
            "mae": chosen["mae"],
            "rmse": chosen["rmse"],
            "r2": chosen["r2"],
            "baseline_mae": baseline["mae"],
            "baseline_rmse": baseline["rmse"],
            "selection_reason": reason
        })

    return pd.DataFrame(selected_models)


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    print("\nAIRSIGHT AI — VALIDATION PIPELINE")
    print("=" * 65)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for pollutant in POLLUTANTS:

        results = validate_pollutant(pollutant)

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    # Save all validation metrics.
    comparison_file = (
        REPORT_DIR / "validation_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )

    # Select one candidate per pollutant.
    selected = select_hybrid_models(results_df)

    selection_file = (
        REPORT_DIR / "hybrid_model_selection.csv"
    )

    selected.to_csv(
        selection_file,
        index=False
    )

    print("\n" + "=" * 65)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.sort_values(
            ["pollutant", "rmse"]
        ).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("PROVISIONAL HYBRID MODEL SELECTION")
    print("=" * 65)

    print(
        selected[
            [
                "pollutant",
                "model",
                "mae",
                "rmse",
                "r2",
                "selection_reason"
            ]
        ].to_string(index=False)
    )

    print("\nSaved validation comparison:")
    print(comparison_file)

    print("\nSaved hybrid selection:")
    print(selection_file)

    print("\nValidation predictions:")
    print(VALIDATION_PREDICTIONS_DIR)

    print("\nValidation pipeline completed successfully!")


if __name__ == "__main__":
    main()
