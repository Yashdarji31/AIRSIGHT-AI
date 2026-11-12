import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "modeling"

SELECTION_FILE = (
    BASE_DIR
    / "reports"
    / "model_evaluation"
    / "hybrid_model_selection.csv"
)

MODEL_DIR = BASE_DIR / "models" / "hybrid"

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "model_evaluation"
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

PERSISTENCE_MODEL = "persistence_baseline"

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

def create_model(model_name):

    models = {
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

    if model_name not in models:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return models[model_name]


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
# 4. TRAIN SELECTED MODEL
# --------------------------------------------------

def train_selected_model(pollutant, model_name):

    print("\n" + "=" * 60)
    print(f"HYBRID TRAINING: {pollutant.upper()}")
    print("=" * 60)

    train_file = (
        DATA_DIR
        / pollutant
        / "train.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    train_df = pd.read_csv(train_file)

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, TIME_COLUMN]
    )

    missing = [
        col for col in required_columns
        if col not in train_df.columns
    ]

    if missing:
        raise ValueError(
            f"{pollutant}: Missing columns: {missing}"
        )

    train_df[TIME_COLUMN] = pd.to_datetime(
        train_df[TIME_COLUMN],
        utc=True,
        errors="coerce"
    )

    if train_df[TIME_COLUMN].isna().any():
        raise ValueError(
            f"Invalid timestamps in {pollutant}"
        )

    X_train = train_df[FEATURE_COLUMNS].copy()

    y_train = train_df[TARGET_COLUMN].astype(float)

    for column in CATEGORICAL_FEATURES:
        X_train[column] = X_train[column].astype(str)

    if model_name == PERSISTENCE_MODEL:

        print("Selected approach: Persistence baseline")
        print("No ML pipeline needs to be trained.")

        return {
            "pollutant": pollutant,
            "selected_model": model_name,
            "training_rows": len(train_df),
            "status": "persistence_no_training_required",
            "model_path": ""
        }

    # Create selected ML pipeline.
    estimator = create_model(model_name)

    preprocessor = create_preprocessor(
        scale_numeric=(
            model_name == "linear_regression"
        )
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    # Fit using the complete original training partition.
    pipeline.fit(X_train, y_train)

    # Save separately from the original model artifacts.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / f"{pollutant}_{model_name}.joblib"
    )

    joblib.dump(pipeline, model_path)

    print("Selected model:", model_name)
    print("Training observations:", len(train_df))
    print("Saved model:", model_path)

    return {
        "pollutant": pollutant,
        "selected_model": model_name,
        "training_rows": len(train_df),
        "status": "trained_successfully",
        "model_path": str(model_path)
    }


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    print("\nAIRSIGHT AI — HYBRID MODEL TRAINING")
    print("=" * 60)

    if not SELECTION_FILE.exists():
        raise FileNotFoundError(SELECTION_FILE)

    selection_df = pd.read_csv(SELECTION_FILE)

    required = ["pollutant", "model"]

    if not set(required).issubset(selection_df.columns):
        raise ValueError(
            "Hybrid selection file has missing columns."
        )

    if selection_df["pollutant"].duplicated().any():
        raise ValueError(
            "Duplicate pollutant selections detected."
        )

    selection = selection_df.set_index("pollutant")

    if set(selection.index) != set(POLLUTANTS):
        raise ValueError(
            "Selection file must contain exactly "
            "the six configured pollutants."
        )

    results = []

    for pollutant in POLLUTANTS:

        model_name = selection.loc[pollutant, "model"]

        result = train_selected_model(
            pollutant,
            model_name
        )

        results.append(result)

    manifest = {
        "architecture": "hybrid",
        "selection_policy": (
            "ML selected only when validation MAE "
            "and RMSE both improve over persistence"
        ),
        "models": results
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_file = (
        MODEL_DIR / "hybrid_model_registry.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    manifest_df = pd.DataFrame(results)

    manifest_df.to_csv(
        REPORT_DIR / "hybrid_training_manifest.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("HYBRID TRAINING SUMMARY")
    print("=" * 60)

    print(
        manifest_df.to_string(index=False)
    )

    print("\nSaved model registry:", manifest_file)

    print("\nHybrid model training completed!")


if __name__ == "__main__":
    main()
