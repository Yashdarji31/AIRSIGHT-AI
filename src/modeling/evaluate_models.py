from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3"
]

MODEL_COLUMNS = [
    "persistence_baseline",
    "linear_regression",
    "random_forest",
    "hist_gradient_boosting"
]


# --------------------------------------------------
# 2. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan
        }

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2 and np.var(actual) > 0
        else np.nan
    )

    return {
        "samples": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# 3. LOAD PREDICTIONS
# --------------------------------------------------

def load_predictions():

    all_predictions = []

    for pollutant in POLLUTANTS:

        file_path = (
            PREDICTIONS_DIR
            / f"{pollutant}_predictions.csv"
        )

        if not file_path.exists():
            print(f"WARNING: Missing file: {file_path}")
            continue

        df = pd.read_csv(file_path)

        required_columns = [
            "target_datetime_utc",
            "city",
            "sensor_id",
            "actual_value"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{pollutant}: Missing columns: {missing}"
            )

        df["pollutant"] = pollutant

        available_models = [
            model for model in MODEL_COLUMNS
            if model in df.columns
        ]

        if not available_models:
            raise ValueError(
                f"No prediction columns found for {pollutant}"
            )

        all_predictions.append(df)

        print(
            f"Loaded {pollutant.upper()}: "
            f"{len(df)} prediction rows"
        )

    if not all_predictions:
        raise ValueError("No prediction files were loaded.")

    return pd.concat(
        all_predictions,
        ignore_index=True
    )


# --------------------------------------------------
# 4. OVERALL MODEL EVALUATION
# --------------------------------------------------

def evaluate_overall(df):

    results = []

    for pollutant in POLLUTANTS:

        pollutant_df = df[
            df["pollutant"] == pollutant
        ]

        for model in MODEL_COLUMNS:

            if model not in pollutant_df.columns:
                continue

            metrics = calculate_metrics(
                pollutant_df["actual_value"],
                pollutant_df[model]
            )

            results.append({
                "pollutant": pollutant,
                "model": model,
                **metrics
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "overall_evaluation.csv",
        index=False
    )

    return results_df


# --------------------------------------------------
# 5. CITY-WISE AND SENSOR-WISE EVALUATION
# --------------------------------------------------

def evaluate_groups(df, group_columns, output_name):

    results = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        group_info = dict(
            zip(group_columns, keys)
        )

        for model in MODEL_COLUMNS:

            if model not in group.columns:
                continue

            metrics = calculate_metrics(
                group["actual_value"],
                group[model]
            )

            results.append({
                **group_info,
                "model": model,
                **metrics
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / output_name,
        index=False
    )

    return results_df


# --------------------------------------------------
# 6. RESIDUAL ANALYSIS
# --------------------------------------------------

def evaluate_residuals(df):

    results = []

    for pollutant in POLLUTANTS:

        pollutant_df = df[
            df["pollutant"] == pollutant
        ]

        for model in MODEL_COLUMNS:

            if model not in pollutant_df.columns:
                continue

            actual = pollutant_df["actual_value"]
            predicted = pollutant_df[model]

            valid = (
                np.isfinite(actual)
                & np.isfinite(predicted)
            )

            actual = actual[valid]
            predicted = predicted[valid]

            if len(actual) == 0:
                continue

            residuals = predicted - actual

            results.append({
                "pollutant": pollutant,
                "model": model,
                "samples": len(residuals),
                "mean_residual": residuals.mean(),
                "residual_std": residuals.std(),
                "mean_absolute_residual": np.abs(
                    residuals
                ).mean(),
                "maximum_absolute_error": np.abs(
                    residuals
                ).max(),
                "negative_predictions": int(
                    (predicted < 0).sum()
                )
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "residual_analysis.csv",
        index=False
    )

    return results_df


# --------------------------------------------------
# 7. SELECT BEST CANDIDATE BY RMSE
# --------------------------------------------------

def select_best_models(overall_df):

    valid = overall_df.dropna(subset=["RMSE"])

    best = (
        valid.sort_values("RMSE")
        .groupby("pollutant", as_index=False)
        .first()
    )

    best.to_csv(
        OUTPUT_DIR / "evaluation_best_models.csv",
        index=False
    )

    return best


# --------------------------------------------------
# 8. GENERATE ACTUAL VS PREDICTED PLOTS
# --------------------------------------------------

def generate_plots(df, best_models):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for _, row in best_models.iterrows():

        pollutant = row["pollutant"]
        model = row["model"]

        subset = df[
            df["pollutant"] == pollutant
        ].copy()

        valid = (
            np.isfinite(subset["actual_value"])
            & np.isfinite(subset[model])
        )

        subset = subset[valid]

        if subset.empty:
            continue

        actual = subset["actual_value"]
        predicted = subset[model]

        # Actual vs predicted

        plt.figure(figsize=(8, 6))

        plt.scatter(
            actual,
            predicted,
            alpha=0.6
        )

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())

        plt.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--"
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")

        plt.title(
            f"{pollutant.upper()} - "
            f"Actual vs Predicted ({model})"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{pollutant}_actual_vs_predicted.png"
        )

        plt.close()

        # Residual distribution

        residuals = predicted - actual

        plt.figure(figsize=(8, 6))

        plt.hist(
            residuals,
            bins=30,
            edgecolor="black"
        )

        plt.axvline(
            0,
            linestyle="--"
        )

        plt.xlabel("Residual (Predicted - Actual)")
        plt.ylabel("Frequency")

        plt.title(
            f"{pollutant.upper()} - Residual Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{pollutant}_residuals.png"
        )

        plt.close()


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("AIRSIGHT AI - MODEL EVALUATION")
    print("=" * 60)

    df = load_predictions()

    print("\nLoaded prediction dataset:")
    print(f"Total rows: {len(df)}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Sensors: {df['sensor_id'].nunique()}")

    overall = evaluate_overall(df)

    evaluate_groups(
        df,
        ["pollutant", "city"],
        "city_wise_evaluation.csv"
    )

    evaluate_groups(
        df,
        ["pollutant", "city", "sensor_id"],
        "sensor_wise_evaluation.csv"
    )

    residuals = evaluate_residuals(df)

    best = select_best_models(overall)

    generate_plots(df, best)

    print("\nOVERALL MODEL PERFORMANCE")
    print(
        overall.sort_values(
            ["pollutant", "RMSE"]
        ).to_string(index=False)
    )

    print("\nBEST CANDIDATE PER POLLUTANT (RMSE)")
    print(best.to_string(index=False))

    print("\nNEGATIVE PREDICTION COUNTS")
    print(
        residuals[
            [
                "pollutant",
                "model",
                "negative_predictions"
            ]
        ].to_string(index=False)
    )

    print("\nEvaluation completed successfully.")

    print(f"\nReports saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
