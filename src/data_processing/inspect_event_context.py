import pandas as pd

from src.config import PROCESSED_DATA_DIR, BASE_DIR


INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "openaq_hourly_measurements_cleaned.csv"
)

EVENT_FILE = (
    BASE_DIR / "reports" / "eda" / "outlier_events.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "reports" / "eda" / "event_context.csv"
)


def main():

    print("Starting event context investigation...\n")

    df = pd.read_csv(INPUT_FILE)
    events = pd.read_csv(EVENT_FILE)

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce"
    )

    events["start_utc"] = pd.to_datetime(
        events["start_utc"],
        utc=True,
        errors="coerce"
    )

    events["end_utc"] = pd.to_datetime(
        events["end_utc"],
        utc=True,
        errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df["outlier_flag"] = (
        df["outlier_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    context_frames = []

    # Investigate the 10 longest events.
    selected_events = events.sort_values(
        ["observations", "peak_value"],
        ascending=False
    ).head(10)

    for _, event in selected_events.iterrows():

        start = event["start_utc"] - pd.Timedelta(hours=3)
        end = event["end_utc"] + pd.Timedelta(hours=3)

        mask = (
            (df["city"] == event["city"])
            & (df["sensor_id"] == event["sensor_id"])
            & (df["parameter"] == event["parameter"])
            & (df["datetime_from_utc"] >= start)
            & (df["datetime_from_utc"] <= end)
        )

        context = df.loc[mask].copy()

        context["event_city"] = event["city"]
        context["event_sensor_id"] = event["sensor_id"]
        context["event_parameter"] = event["parameter"]

        context["datetime_ist"] = (
            context["datetime_from_utc"]
            .dt.tz_convert("Asia/Kolkata")
        )

        context_frames.append(context)

    if not context_frames:
        print("No event context found.")
        return

    result = pd.concat(
        context_frames,
        ignore_index=True
    )

    columns = [
        "event_city",
        "event_sensor_id",
        "event_parameter",
        "datetime_ist",
        "value",
        "unit",
        "outlier_flag"
    ]

    result = result[columns].sort_values(
        [
            "event_city",
            "event_sensor_id",
            "event_parameter",
            "datetime_ist"
        ]
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("Events investigated:", len(selected_events))
    print("Context observations:", len(result))

    print("\n========== EVENT CONTEXT ==========")

    print(result.to_string(index=False))

    print("\nReport saved:")
    print(OUTPUT_FILE)

    print("\nEvent context investigation completed.")


if __name__ == "__main__":
    main()
