
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from map_view import render_sensor_map

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIRSIGHT AI | Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openaq_hourly_measurements_cleaned.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
    / "hybrid_predictions.csv"
)

REJECTED_PATH = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
    / "rejected_sensors.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "hybrid_backtest"
    / "hybrid_backtest_metrics.csv"
)


CITY_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "hybrid_backtest"
    / "hybrid_backtest_city_metrics.csv"
)

# ============================================================
# 3. DARK INTELLIGENCE THEME
# ============================================================

CSS_PATH = Path(__file__).parent / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )


# ============================================================
# 4. PLOTLY DARK THEME
# ============================================================

pio.templates["airsight_dark"] = pio.templates["plotly_dark"]

pio.templates["airsight_dark"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        family="Arial",
        color="#CBD5E1",
        size=12,
    ),

    title=dict(
        font=dict(
            color="#F1F5F9",
            size=18,
        )
    ),

    xaxis=dict(
        gridcolor="#263449",
        zerolinecolor="#263449",
        linecolor="#263449",
    ),

    yaxis=dict(
        gridcolor="#263449",
        zerolinecolor="#263449",
        linecolor="#263449",
    ),

    legend=dict(
        font=dict(color="#CBD5E1"),
    ),
)

pio.templates.default = "airsight_dark"


# ============================================================
# 5. POLLUTANT COLORS
# ============================================================

POLLUTANT_COLORS = {
    "pm25": "#22D3EE",
    "pm10": "#3B82F6",
    "no2": "#F59E0B",
    "so2": "#A78BFA",
    "co": "#10B981",
    "o3": "#F472B6",
}

PARTICULATE_POLLUTANTS = ["pm25", "pm10", "o3"]
GAS_POLLUTANTS = ["co", "no2", "so2"]


# ============================================================
# 6. DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_main_data(path_string):

    df = load_csv(path_string)

    if df.empty:
        return df

    required_columns = [
        "sensor_id",
        "city",
        "parameter",
        "value",
        "datetime_from_utc",
        "coverage_percent",
        "negative_value_flag",
        "outlier_flag",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df["coverage_percent"] = pd.to_numeric(
        df["coverage_percent"],
        errors="coerce",
    )

    df["negative_value_flag"] = (
        df["negative_value_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    df["outlier_flag"] = (
        df["outlier_flag"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return df.dropna(
        subset=[
            "datetime_from_utc",
            "value",
            "city",
            "parameter",
        ]
    )


try:
    data = load_main_data(str(DATA_PATH))
    predictions = load_csv(str(PREDICTIONS_PATH))
    rejected = load_csv(str(REJECTED_PATH))
    metrics = load_csv(str(METRICS_PATH))
    city_metrics = load_csv(str(CITY_METRICS_PATH))

except Exception as error:
    st.error(f"Data loading failed: {error}")
    st.stop()


# ============================================================
# 7. VALIDATE MAIN DATASET
# ============================================================

st.title("🌍 AIRSIGHT AI")

st.caption(
    "AI-powered multi-city air quality monitoring "
    "and next-hour pollutant prediction"
)

if data.empty:
    st.error(
        "Hourly measurement data could not be loaded. "
        "Check the cleaned CSV file."
    )
    st.stop()


latest_timestamp = data["datetime_from_utc"].max()

st.warning(
    "HISTORICAL DATA MODE — This dashboard displays "
    "stored OpenAQ measurements and historical model "
    "predictions. It is not a live monitoring system."
)

st.caption(
    "Latest measurement in dataset: "
    f"{latest_timestamp.strftime('%d %b %Y, %H:%M UTC')}"
)


# ============================================================
# 8. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("AIRSIGHT AI")
st.sidebar.caption("Dashboard Controls")

cities = sorted(data["city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "Select City",
    ["All Cities"] + cities,
)

pollutants = sorted(data["parameter"].dropna().unique())

selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["All Pollutants"] + pollutants,
)

filtered = data.copy()

if selected_city != "All Cities":
    filtered = filtered[
        filtered["city"] == selected_city
    ]

if selected_pollutant != "All Pollutants":
    filtered = filtered[
        filtered["parameter"] == selected_pollutant
    ]

if filtered.empty:
    st.warning("No measurements match the selected filters.")
    st.stop()


# ============================================================
# 9. OVERVIEW METRICS
# ============================================================

st.subheader("Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Cities",
    filtered["city"].nunique(),
)

col2.metric(
    "Sensors",
    filtered["sensor_id"].nunique(),
)

col3.metric(
    "Pollutants",
    filtered["parameter"].nunique(),
)

col4.metric(
    "Measurements",
    f"{len(filtered):,}",
)

st.divider()


# ============================================================
# 10. DASHBOARD TABS
# ============================================================

overview_tab, city_tab, prediction_tab, model_tab, quality_tab, sensor_map_tab = (
    st.tabs(
        [
            "Overview",
            "City Explorer",
            "AI Predictions",
            "Model Performance",
            "Data Quality",
            "Sensor Map",
        ]
    )
)


# ============================================================
# 11. OVERVIEW TAB
# ============================================================

with overview_tab:

    st.subheader("Pollutant Concentration Analysis")

    chart_data = filtered[
        ~filtered["negative_value_flag"]
    ].copy()

    particulate_data = chart_data[
        chart_data["parameter"].isin(
            PARTICULATE_POLLUTANTS
        )
    ]

    gas_data = chart_data[
        chart_data["parameter"].isin(
            GAS_POLLUTANTS
        )
    ]

    chart_col1, chart_col2 = st.columns(2)

    # --------------------------------------------------------
    # PARTICULATE AND OZONE CHART
    # --------------------------------------------------------

    with chart_col1:

        st.markdown("#### Particulate & Ozone")
        st.caption("Concentration in µg/m³")

        if not particulate_data.empty:

            particulate_summary = (
                particulate_data
                .groupby("parameter", as_index=False)
                .agg(
                    median_value=("value", "median")
                )
            )

            fig = px.bar(
                particulate_summary,
                x="parameter",
                y="median_value",
                color="parameter",
                color_discrete_map=POLLUTANT_COLORS,
                labels={
                    "parameter": "Pollutant",
                    "median_value": "Median concentration",
                },
            )

            fig.update_layout(
                showlegend=False,
                height=380,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        else:
            st.info(
                "No particulate or ozone data available."
            )

    # --------------------------------------------------------
    # GASEOUS POLLUTANT CHART
    # --------------------------------------------------------

    with chart_col2:

        st.markdown("#### Gaseous Pollutants")
        st.caption("Concentration in ppb")

        if not gas_data.empty:

            gas_summary = (
                gas_data
                .groupby("parameter", as_index=False)
                .agg(
                    median_value=("value", "median")
                )
            )

            fig = px.bar(
                gas_summary,
                x="parameter",
                y="median_value",
                color="parameter",
                color_discrete_map=POLLUTANT_COLORS,
                labels={
                    "parameter": "Pollutant",
                    "median_value": "Median concentration",
                },
            )

            fig.update_layout(
                showlegend=False,
                height=380,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        else:
            st.info(
                "No gaseous pollutant data available."
            )

    st.caption(
        "Values represent medians of available sensor "
        "measurements. Pollutants are separated by unit "
        "to avoid misleading cross-unit comparisons."
    )

    st.divider()

    # --------------------------------------------------------
    # CITY-WISE MEASUREMENT COVERAGE
    # --------------------------------------------------------

    st.subheader("City-wise Measurement Coverage")

    city_counts = (
        filtered.groupby("city")
        .size()
        .reset_index(name="measurements")
    )

    fig = px.bar(
        city_counts,
        x="city",
        y="measurements",
        color="city",
        title="Available measurements per city",
    )

    fig.update_layout(
        showlegend=False,
        height=420,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# 12. CITY EXPLORER TAB
# ============================================================

with city_tab:

    st.subheader("Historical Pollutant Trends")

    trend_data = filtered[
        ~filtered["negative_value_flag"]
    ].copy()

    trend_data["hour_timestamp"] = (
        trend_data["datetime_from_utc"]
        .dt.floor("h")
    )

    trend = (
        trend_data
        .groupby(
            [
                "hour_timestamp",
                "parameter",
                "unit",
            ],
            as_index=False,
        )
        .agg(
            median_value=("value", "median"),
            sensor_count=("sensor_id", "nunique"),
        )
    )

    # Separate trend charts by measurement unit

    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:

        st.markdown("#### Particulate & Ozone Trends")

        particulate_trend = trend[
            trend["parameter"].isin(
                PARTICULATE_POLLUTANTS
            )
        ]

        if not particulate_trend.empty:

            fig = px.line(
                particulate_trend,
                x="hour_timestamp",
                y="median_value",
                color="parameter",
                color_discrete_map=POLLUTANT_COLORS,
                labels={
                    "hour_timestamp": "Time (UTC)",
                    "median_value": "Median concentration",
                    "parameter": "Pollutant",
                },
                hover_data=[
                    "unit",
                    "sensor_count",
                ],
            )

            fig.update_layout(height=400)

            st.plotly_chart(
                fig,
                width="stretch",
            )

        else:
            st.info("No particulate trend data available.")

    with trend_col2:

        st.markdown("#### Gaseous Pollutant Trends")

        gas_trend = trend[
            trend["parameter"].isin(
                GAS_POLLUTANTS
            )
        ]

        if not gas_trend.empty:

            fig = px.line(
                gas_trend,
                x="hour_timestamp",
                y="median_value",
                color="parameter",
                color_discrete_map=POLLUTANT_COLORS,
                labels={
                    "hour_timestamp": "Time (UTC)",
                    "median_value": "Median concentration",
                    "parameter": "Pollutant",
                },
                hover_data=[
                    "unit",
                    "sensor_count",
                ],
            )

            fig.update_layout(height=400)

            st.plotly_chart(
                fig,
                width="stretch",
            )

        else:
            st.info("No gaseous pollutant trend data available.")

    st.caption(
        "Each point represents the median of available "
        "sensor measurements for the selected city and hour."
    )

    st.divider()

    st.subheader("Measurement Records")

    display_columns = [
        "city",
        "location_name",
        "sensor_id",
        "parameter",
        "unit",
        "value",
        "coverage_percent",
        "datetime_from_utc",
    ]

    available_columns = [
        col for col in display_columns
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[available_columns].sort_values(
            "datetime_from_utc",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 13. AI PREDICTIONS TAB
# ============================================================

with prediction_tab:

    st.subheader("Next-Hour Prediction Results")

    if predictions.empty:

        st.info("No prediction records are available.")

    else:

        st.warning(
            "HISTORICAL DEMONSTRATION — These predictions "
            "were generated from historical measurements "
            "and are not current live forecasts."
        )

        predictions_display = predictions.copy()



        # ------------------------------------------------------------
        # PREDICTION AVAILABILITY AND DATA QUALITY SUMMARY
        # ------------------------------------------------------------

        rejected_display = rejected.copy()

        # Apply the same city and pollutant filters
        if not rejected_display.empty:

            if selected_city != "All Cities":
                rejected_display = rejected_display[
                    rejected_display["city"] == selected_city
                ]

            if selected_pollutant != "All Pollutants":
                rejected_display = rejected_display[
                    rejected_display["pollutant"] == selected_pollutant
                ]

        available_count = len(predictions_display)
        rejected_count = len(rejected_display)
        assessed_count = available_count + rejected_count

        availability_rate = (
            (available_count / assessed_count) * 100
            if assessed_count > 0 else 0
        )

        st.markdown("### Prediction Availability")

        availability_col1, availability_col2, availability_col3 = (
            st.columns(3)
        )

        availability_col1.metric(
            "Available Predictions",
            available_count,
        )

        availability_col2.metric(
            "Rejected Records",
            rejected_count,
        )

        availability_col3.metric(
            "Availability Rate",
            f"{availability_rate:.1f}%",
        )

        st.caption(
            "Availability rate = available prediction records / "
            "all assessed records. Counts are based on the selected filters."
        )

        # Rejection reasons
        if not rejected_display.empty:

            st.markdown("#### Why were sensors rejected?")

            rejection_summary = (
                rejected_display["reason"]
                .value_counts()
                .rename_axis("reason")
                .reset_index(name="count")
            )

            rejection_fig = px.bar(
                rejection_summary,
                x="count",
                y="reason",
                orientation="h",
                text="count",
                color_discrete_sequence=["#22D3EE"],
            )

            rejection_fig.update_traces(
                textposition="outside",
            )

            rejection_fig.update_layout(
                height=350,
                showlegend=False,
                yaxis_title="",
            )

            st.plotly_chart(
                rejection_fig,
                width="stretch",
            )
        else:
            st.info(
                "No rejected sensor records for the selected filters."
            )



        numeric_columns = [
            "latest_observed_value",
            "predicted_value",
            "latest_coverage_percent",
            "data_age_hours",
        ]

        for column in numeric_columns:

            if column in predictions_display.columns:

                predictions_display[column] = pd.to_numeric(
                    predictions_display[column],
                    errors="coerce",
                )

        if selected_city != "All Cities":

            predictions_display = predictions_display[
                predictions_display["city"] == selected_city
            ]

        if selected_pollutant != "All Pollutants":

            predictions_display = predictions_display[
                predictions_display["pollutant"]
                == selected_pollutant
            ]

        if predictions_display.empty:

            st.info(
                "No predictions are available for "
                "the selected city and pollutant."
            )

        else:

            st.metric(
                "Available prediction records",
                len(predictions_display),
            )

            st.dataframe(
                predictions_display,
                width="stretch",
                hide_index=True,
            )

            # Latest observation versus predicted value

            prediction_col1, prediction_col2 = st.columns(2)

            with prediction_col1:

                st.markdown("#### Observed vs Predicted")

                fig = px.bar(
                    predictions_display,
                    x="city",
                    y=[
                        "latest_observed_value",
                        "predicted_value",
                    ],
                    barmode="group",
                    color_discrete_sequence=[
                        "#22D3EE",
                        "#F59E0B",
                    ],
                    labels={
                        "value": "Concentration",
                        "variable": "Measurement",
                        "city": "City",
                    },
                )

                fig.update_layout(height=400)

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            with prediction_col2:

                st.markdown("#### Prediction Details")

                for _, row in predictions_display.iterrows():

                    with st.container(border=True):

                        st.markdown(
                            f"**{row['city']} — "
                            f"{row['pollutant'].upper()}**"
                        )

                        st.write(
                            f"Observed: "
                            f"{row['latest_observed_value']} "
                            f"{row['unit']}"
                        )

                        st.write(
                            f"Predicted: "
                            f"{row['predicted_value']:.2f} "
                            f"{row['unit']}"
                        )

                        st.caption(
                            f"Model: {row['selected_model']}"
                        )

                        st.caption(
                            f"Target: "
                            f"{row['prediction_datetime_utc']}"
                        )



# ============================================================
# 14. MODEL PERFORMANCE TAB
# ============================================================

with model_tab:

    st.subheader("Hybrid Model Evaluation")

    # --------------------------------------------------------
    # OVERALL MODEL PERFORMANCE
    # --------------------------------------------------------

    if metrics.empty:

        st.info(
            "Model evaluation metrics are unavailable."
        )

    else:

        st.warning(
            "RETROSPECTIVE BACKTEST — The test data was "
            "previously examined during model development. "
            "These results are not an untouched independent "
            "benchmark."
        )

        numeric_columns = [
            "hybrid_MAE",
            "hybrid_RMSE",
            "hybrid_R2",
            "persistence_MAE",
            "persistence_RMSE",
            "persistence_R2",
        ]

        for column in numeric_columns:

            if column in metrics.columns:

                metrics[column] = pd.to_numeric(
                    metrics[column],
                    errors="coerce",
                )

        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # MAE COMPARISON
        # ----------------------------------------------------

        st.subheader("MAE Comparison")

        mae_comparison = metrics.melt(
            id_vars=["pollutant"],
            value_vars=[
                "hybrid_MAE",
                "persistence_MAE",
            ],
            var_name="model",
            value_name="MAE",
        )

        fig = px.bar(
            mae_comparison,
            x="pollutant",
            y="MAE",
            color="model",
            barmode="group",
            color_discrete_sequence=[
                "#22D3EE",
                "#F59E0B",
            ],
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # ----------------------------------------------------
        # RMSE COMPARISON
        # ----------------------------------------------------

        st.subheader("RMSE Comparison")

        rmse_comparison = metrics.melt(
            id_vars=["pollutant"],
            value_vars=[
                "hybrid_RMSE",
                "persistence_RMSE",
            ],
            var_name="model",
            value_name="RMSE",
        )

        fig = px.bar(
            rmse_comparison,
            x="pollutant",
            y="RMSE",
            color="model",
            barmode="group",
            color_discrete_sequence=[
                "#22D3EE",
                "#F59E0B",
            ],
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "Error metrics are reported in each pollutant's "
            "original unit. Compare hybrid and persistence "
            "within the same pollutant, not across pollutants."
        )

    # ========================================================
    # CITY-WISE HYBRID VS PERSISTENCE ANALYSIS
    # ========================================================

    st.divider()

    st.subheader("City-wise Model Intelligence")

    if city_metrics.empty:

        st.info(
            "City-wise evaluation data is unavailable."
        )

    else:

        city_analysis = city_metrics.copy()

        # ----------------------------------------------------
        # VALIDATE REQUIRED COLUMNS
        # ----------------------------------------------------

        required_city_columns = [
            "city",
            "pollutant",
            "hybrid_MAE",
            "persistence_MAE",
            "hybrid_RMSE",
            "persistence_RMSE",
            "evaluation_rows",
        ]

        missing_city_columns = [
            column
            for column in required_city_columns
            if column not in city_analysis.columns
        ]

        if missing_city_columns:

            st.error(
                "City-wise metrics file is missing required "
                f"columns: {missing_city_columns}"
            )

        else:

            # ------------------------------------------------
            # CONVERT NUMERIC COLUMNS
            # ------------------------------------------------

            numeric_city_columns = [
                "hybrid_MAE",
                "persistence_MAE",
                "hybrid_RMSE",
                "persistence_RMSE",
                "evaluation_rows",
            ]

            for column in numeric_city_columns:

                city_analysis[column] = pd.to_numeric(
                    city_analysis[column],
                    errors="coerce",
                )

            city_pollutants = sorted(
                city_analysis["pollutant"]
                .dropna()
                .unique()
            )

            if not city_pollutants:

                st.info(
                    "No pollutant-level city metrics are available."
                )

            else:

                # --------------------------------------------
                # POLLUTANT SELECTOR
                # --------------------------------------------

                selected_model_pollutant = st.selectbox(
                    "Select pollutant for city-wise comparison",
                    city_pollutants,
                    key="city_model_pollutant",
                )

                city_analysis = city_analysis[
                    city_analysis["pollutant"]
                    == selected_model_pollutant
                ].copy()

                # --------------------------------------------
                # CALCULATE IMPROVEMENT PERCENTAGES
                # --------------------------------------------

                # Positive improvement means hybrid has lower error.

                city_analysis["MAE_improvement_percent"] = (
                    (
                        city_analysis["persistence_MAE"]
                        - city_analysis["hybrid_MAE"]
                    )
                    / city_analysis["persistence_MAE"].replace(
                        0,
                        float("nan"),
                    )
                ) * 100

                city_analysis["RMSE_improvement_percent"] = (
                    (
                        city_analysis["persistence_RMSE"]
                        - city_analysis["hybrid_RMSE"]
                    )
                    / city_analysis["persistence_RMSE"].replace(
                        0,
                        float("nan"),
                    )
                ) * 100

                # --------------------------------------------
                # MODEL WIN COUNTS
                # --------------------------------------------

                valid_rmse_improvements = city_analysis[
                    "RMSE_improvement_percent"
                ].dropna()

                hybrid_wins = int(
                    (
                        valid_rmse_improvements > 0
                    ).sum()
                )

                persistence_wins = int(
                    (
                        valid_rmse_improvements < 0
                    ).sum()
                )

                tied = int(
                    (
                        valid_rmse_improvements == 0
                    ).sum()
                )

                win_col1, win_col2, win_col3 = st.columns(3)

                win_col1.metric(
                    "Hybrid performs better",
                    hybrid_wins,
                )

                win_col2.metric(
                    "Persistence performs better",
                    persistence_wins,
                )

                win_col3.metric(
                    "Equal RMSE",
                    tied,
                )

                st.caption(
                    "Model winners are determined by RMSE within "
                    "each city for the selected pollutant. Cities "
                    "with missing metric values are excluded "
                    "from the win counts."
                )

                # --------------------------------------------
                # CITY-WISE RMSE IMPROVEMENT CHART
                # --------------------------------------------

                st.subheader(
                    f"{selected_model_pollutant.upper()} — "
                    "City-wise RMSE Improvement"
                )

                chart_data = (
                    city_analysis
                    .dropna(
                        subset=["RMSE_improvement_percent"]
                    )
                    .sort_values(
                        "RMSE_improvement_percent",
                        ascending=True,
                    )
                )

                if not chart_data.empty and (
                    chart_data["RMSE_improvement_percent"].abs() < 1e-9
                ).all():

                    st.info(
                        "Both models have identical RMSE across all "
                        "eight cities for this pollutant."
                    )

                elif not chart_data.empty:

                    fig = px.bar(
                        chart_data,
                        x="RMSE_improvement_percent",
                        y="city",
                        orientation="h",
                        color="RMSE_improvement_percent",
                        color_continuous_scale=[
                            "#EF4444",
                            "#F59E0B",
                            "#10B981",
                        ],
                        labels={
                            "RMSE_improvement_percent": (
                                "RMSE improvement (%)"
                            ),
                            "city": "City",
                        },
                        hover_data={
                            "hybrid_RMSE": ":.3f",
                            "persistence_RMSE": ":.3f",
                            "evaluation_rows": True,
                        },
                    )

                    fig.add_vline(
                        x=0,
                        line_dash="dash",
                        line_color="white",
                    )

                    fig.update_layout(
                        height=430,
                        coloraxis_colorbar_title="Improvement (%)",
                        yaxis_title="",
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                else:

                    st.info(
                        "No valid RMSE comparison values are "
                        "available for this pollutant."
                    )

                st.caption(
                    "Positive values indicate lower hybrid RMSE. "
                    "Negative values indicate lower persistence "
                    "RMSE. The comparison is within the selected "
                    "pollutant."
                )

                # --------------------------------------------
                # DETAILED CITY-WISE METRICS TABLE
                # --------------------------------------------

                st.subheader(
                    "Detailed City-wise Metrics"
                )

                display_columns = [
                    "city",
                    "pollutant",
                    "hybrid_MAE",
                    "persistence_MAE",
                    "MAE_improvement_percent",
                    "hybrid_RMSE",
                    "persistence_RMSE",
                    "RMSE_improvement_percent",
                    "evaluation_rows",
                ]

                st.dataframe(
                    city_analysis[display_columns].round(3),
                    width="stretch",
                    hide_index=True,
                )



# ============================================================
# 15. DATA QUALITY TAB
# ============================================================

with quality_tab:

    st.subheader("Sensor Data Quality")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    negative_count = int(
        filtered["negative_value_flag"].sum()
    )

    outlier_count = int(
        filtered["outlier_flag"].sum()
    )

    low_coverage_count = int(
        (filtered["coverage_percent"] < 75).sum()
    )

    quality_col1.metric(
        "Negative-value flags",
        negative_count,
    )

    quality_col2.metric(
        "Outlier flags",
        outlier_count,
    )

    quality_col3.metric(
        "Low-coverage records",
        low_coverage_count,
    )

    st.divider()

    # Coverage distribution

    st.subheader("Sensor Coverage Distribution")

    coverage_data = filtered.dropna(
        subset=["coverage_percent"]
    ).copy()

    if not coverage_data.empty:

        coverage_data["coverage_category"] = pd.cut(
            coverage_data["coverage_percent"],
            bins=[-0.01, 50, 75, 100],
            labels=[
                "0–50%",
                "50–75%",
                "75–100%",
            ],
        )

        coverage_summary = (
            coverage_data["coverage_category"]
            .value_counts()
            .reindex(
                [
                    "0–50%",
                    "50–75%",
                    "75–100%",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        coverage_summary.columns = [
            "coverage_category",
            "records",
        ]

        fig = px.bar(
            coverage_summary,
            x="coverage_category",
            y="records",
            color="coverage_category",
            title="Measurement coverage categories",
        )

        fig.update_layout(
            showlegend=False,
            height=350,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    else:

        st.info("Coverage information is unavailable.")

    st.divider()

    # Rejected sensors

    st.subheader("Rejected Sensors")

    if rejected.empty:

        st.info(
            "No rejected sensor report is available."
        )

    else:

        rejected_display = rejected.copy()

        if selected_city != "All Cities":

            rejected_display = rejected_display[
                rejected_display["city"] == selected_city
            ]

        if selected_pollutant != "All Pollutants":

            rejected_display = rejected_display[
                rejected_display["pollutant"]
                == selected_pollutant
            ]

        st.metric(
            "Rejected sensor records",
            len(rejected_display),
        )

        st.dataframe(
            rejected_display,
            width="stretch",
            hide_index=True,
        )



# ============================================================
# 16. SENSOR MAP TAB
# ============================================================

with sensor_map_tab:

    render_sensor_map()


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <h4 style="color:#22D3EE;">🌍 AIRSIGHT AI</h4>
        <p style="color:#94A3B8;">
            Air Quality Intelligence Platform
        </p>
        <p style="color:#64748B; font-size:12px;">
            Powered by OpenAQ data and a hybrid ML prediction pipeline
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
