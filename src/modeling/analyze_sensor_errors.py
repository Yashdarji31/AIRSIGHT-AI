from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "reports"
    / "model_evaluation"
    / "hybrid_backtest"
    / "hybrid_backtest_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "model_evaluation"
    / "hybrid_backtest"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for (pollutant, city, sensor_id), group in df.groupby(
        ["pollutant", "city", "sensor_id"]
    ):
        actual = group["actual_value"]
        hybrid = group["hybrid_prediction"]
        persistence = group["persistence_prediction"]

        hybrid_mae = mean_absolute_error(actual, hybrid)
        persistence_mae = mean_absolute_error(actual, persistence)

        hybrid_rmse = np.sqrt(
            mean_squared_error(actual, hybrid)
        )

        persistence_rmse = np.sqrt(
            mean_squared_error(actual, persistence)
        )

        results.append({
            "pollutant": pollutant,
            "city": city,
            "sensor_id": sensor_id,
            "observations": len(group),

            "hybrid_MAE": hybrid_mae,
            "persistence_MAE": persistence_mae,

            "hybrid_RMSE": hybrid_rmse,
            "persistence_RMSE": persistence_rmse,

            "MAE_improvement_percent": (
                (persistence_mae - hybrid_mae)
                / persistence_mae * 100
                if persistence_mae != 0 else np.nan
            ),

            "RMSE_improvement_percent": (
                (persistence_rmse - hybrid_rmse)
                / persistence_rmse * 100
                if persistence_rmse != 0 else np.nan
            ),

            "mean_prediction_error": float(
                (hybrid - actual).mean()
            ),
        })

    result_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "sensor_level_backtest.csv"

    result_df.to_csv(output_file, index=False)

    print("\nAIRSIGHT AI — SENSOR-LEVEL ERROR ANALYSIS")
    print("=" * 65)

    for pollutant in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        subset = result_df[
            result_df["pollutant"] == pollutant
        ]

        print(f"\n{pollutant.upper()}")

        print(
            subset.sort_values(
                "RMSE_improvement_percent"
            ).to_string(index=False)
        )

    print("\nSaved:", output_file)
    print("\nSensor-level analysis completed.")


if __name__ == "__main__":
    main()
