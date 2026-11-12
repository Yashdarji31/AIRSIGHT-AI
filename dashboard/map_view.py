import html
import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st

from branca.element import MacroElement, Template
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "openaq_hourly_measurements_cleaned.csv"
)

POLLUTANT_COLORS = {
    "pm25": "#00D4FF",
    "pm10": "#00E5A8",
    "no2": "#FFB020",
    "so2": "#FF6B6B",
    "co": "#B38BFF",
    "o3": "#FFE066",
}



class ResetMapControl(MacroElement):
    """Add a reliable reset-view button to a Folium map."""

    _template = Template("""
    {% macro script(this, kwargs) %}

        var resetControl = L.control({
            position: 'topleft'
        });

        resetControl.onAdd = function(map) {

            var container = L.DomUtil.create(
                'div',
                'leaflet-bar leaflet-control'
            );

            var button = L.DomUtil.create(
                'a',
                '',
                container
            );

            button.href = '#';
            button.title = 'Reset map view';
            button.innerHTML = '&#8634;';

            button.style.width = '85px';
            button.style.height = '32px';
            button.style.lineHeight = '32px';
            button.style.textAlign = 'center';
            button.style.fontSize = '12px';
            button.style.fontWeight = 'bold';
            button.style.color = '#22D3EE';
            button.style.background = '#111827';
            button.style.border = '1px solid #22D3EE';
            button.style.textDecoration = 'none';

            L.DomEvent.disableClickPropagation(container);

            L.DomEvent.on(button, 'click', function(event) {

                L.DomEvent.preventDefault(event);

                var bounds = {{ this.bounds_json | safe }};

                if (bounds[0][0] === bounds[1][0] &&
                    bounds[0][1] === bounds[1][1]) {

                    map.setView(
                        bounds[0],
                        11
                    );

                } else {

                    map.fitBounds(
                        bounds,
                        {padding: [40, 40]}
                    );

                }
            });

            return container;
        };

        resetControl.addTo({{ this._parent.get_name() }});

    {% endmacro %}
    """)

    def __init__(self, bounds):
        super().__init__()
        self._name = "ResetMapControl"
        self.bounds_json = json.dumps(bounds)


# ============================================================
# 2. LOAD SENSOR DATA
# ============================================================

def load_sensor_map_data():
    """Load the latest record for each sensor-pollutant pair."""

    required_columns = [
        "sensor_id",
        "city",
        "parameter",
        "unit",
        "value",
        "datetime_from_utc",
        "coverage_percent",
        "latitude",
        "longitude",
    ]

    if not DATA_PATH.exists():
        st.error(f"Sensor dataset not found: {DATA_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        st.error(f"Could not read sensor dataset: {exc}")
        return pd.DataFrame()

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        st.error(
            f"Missing required dataset columns: {', '.join(missing)}"
        )
        return pd.DataFrame()

    df = df.copy()

    df["datetime_from_utc"] = pd.to_datetime(
        df["datetime_from_utc"],
        utc=True,
        errors="coerce",
    )

    for col in [
        "latitude",
        "longitude",
        "value",
        "coverage_percent",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["parameter"] = (
        df["parameter"]
        .astype(str)
        .str.lower()
    )

    df = df.dropna(
        subset=[
            "sensor_id",
            "city",
            "parameter",
            "datetime_from_utc",
            "latitude",
            "longitude",
            "value",
        ]
    )

    # Keep the latest observation for each sensor-pollutant pair.
    latest = (
        df.sort_values("datetime_from_utc")
        .drop_duplicates(
            subset=["sensor_id", "parameter"],
            keep="last",
        )
    )

    return latest


# ============================================================
# 3. RENDER INTERACTIVE SENSOR MAP
# ============================================================

def render_sensor_map():

    st.subheader("India Sensor Network")

    st.caption(
        "Explore the latest available historical measurement "
        "for each sensor-pollutant pair."
    )

    data = load_sensor_map_data()

    if data.empty:
        st.warning(
            "No valid sensor records are available to display."
        )
        return

    st.info(
        "Historical dataset view — markers show the latest "
        "stored observations, not live air-quality readings. "
        "Values are pollutant concentrations, not calculated AQI."
    )

    # --------------------------------------------------------
    # CITY AND POLLUTANT FILTERS
    # --------------------------------------------------------

    cities = (
        ["All Cities"]
        + sorted(data["city"].dropna().unique().tolist())
    )

    pollutants = (
        ["All Pollutants"]
        + sorted(data["parameter"].dropna().unique().tolist())
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_city = st.selectbox(
            "Filter by city",
            cities,
            key="sensor_map_city",
        )

    with filter_col2:
        selected_pollutant = st.selectbox(
            "Filter by pollutant",
            pollutants,
            key="sensor_map_pollutant",
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
        st.warning(
            "No sensor records match the selected filters."
        )
        return

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Sensor-pollutant records",
        len(filtered),
    )

    metric2.metric(
        "Unique sensors",
        filtered["sensor_id"].nunique(),
    )

    metric3.metric(
        "Cities represented",
        filtered["city"].nunique(),
    )

    # --------------------------------------------------------
    # CALCULATE SENSOR BOUNDS
    # --------------------------------------------------------

    coordinates = (
        filtered[["latitude", "longitude"]]
        .drop_duplicates()
    )

    unique_coordinate_count = len(coordinates)

    min_lat = float(coordinates["latitude"].min())
    max_lat = float(coordinates["latitude"].max())
    min_lon = float(coordinates["longitude"].min())
    max_lon = float(coordinates["longitude"].max())

    sensor_bounds = [
        [min_lat, min_lon],
        [max_lat, max_lon],
    ]

    # --------------------------------------------------------
    # CREATE INDIA MAP
    # --------------------------------------------------------

    india_map = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Automatically position the map.
    if unique_coordinate_count == 1:

        india_map.location = [
            min_lat,
            min_lon,
        ]

        india_map.zoom_start = 11

    else:

        india_map.fit_bounds(
            sensor_bounds,
            padding=[40, 40],
        )

    # --------------------------------------------------------
    # DARK MAP STYLING
    # --------------------------------------------------------

    dark_map_css = """
    <style>

        .leaflet-tile-pane {
            filter:
                invert(100%)
                hue-rotate(180deg)
                brightness(85%)
                contrast(90%)
                saturate(70%);
        }

        .leaflet-control-zoom a {
            background-color: #111827 !important;
            color: #F1F5F9 !important;
            border-color: #374151 !important;
        }

        .leaflet-control-attribution {
            background: rgba(17, 24, 39, 0.85) !important;
            color: #CBD5E1 !important;
        }

        .leaflet-control-attribution a {
            color: #22D3EE !important;
        }

        .leaflet-control-layers {
            background: #111827 !important;
            color: #F1F5F9 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }

        .leaflet-control-layers label {
            color: #F1F5F9 !important;
        }

    </style>
    """

    india_map.get_root().header.add_child(
        folium.Element(dark_map_css)
    )

    # --------------------------------------------------------
    # CREATE SEPARATE POLLUTANT LAYERS
    # --------------------------------------------------------

    pollutant_clusters = {}

    for pollutant in sorted(
        filtered["parameter"].unique()
    ):

        pollutant_group = folium.FeatureGroup(
            name=f"{pollutant.upper()} Sensors",
            show=True,
        ).add_to(india_map)

        pollutant_clusters[pollutant] = MarkerCluster(
            show_coverage_on_hover=False,
        ).add_to(pollutant_group)

    # --------------------------------------------------------
    # ADD SENSOR MARKERS
    # --------------------------------------------------------

    for _, row in filtered.iterrows():

        pollutant = row["parameter"]

        color = POLLUTANT_COLORS.get(
            pollutant,
            "#FFFFFF",
        )

        city = html.escape(str(row["city"]))
        sensor_id = html.escape(str(row["sensor_id"]))

        pollutant_label = html.escape(
            pollutant.upper()
        )

        unit = html.escape(str(row["unit"]))

        timestamp = html.escape(
            row["datetime_from_utc"].strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        )

        coverage = row["coverage_percent"]

        coverage_text = (
            f"{coverage:.1f}%"
            if pd.notna(coverage)
            else "Not available"
        )

        popup_html = f"""
        <div style="font-family:Arial; min-width:190px;">

            <h4 style="margin-bottom:8px;">
                {city}
            </h4>

            <b>Sensor ID:</b> {sensor_id}<br>

            <b>Pollutant:</b> {pollutant_label}<br>

            <b>Latest value:</b>
            {row['value']:.3f} {unit}<br>

            <b>Coverage:</b> {coverage_text}<br>

            <b>Observed:</b> {timestamp}

        </div>
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"],
            ],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{city} | {pollutant_label}",
            popup=folium.Popup(
                popup_html,
                max_width=300,
            ),
        ).add_to(
            pollutant_clusters[pollutant]
        )

    # --------------------------------------------------------
    # POLLUTANT LAYER CONTROL
    # --------------------------------------------------------

    folium.LayerControl(
        position="topright",
        collapsed=False,
    ).add_to(india_map)

    
    # --------------------------------------------------------
    # RESET MAP CONTROL
    # --------------------------------------------------------

    ResetMapControl(sensor_bounds).add_to(india_map)

    # --------------------------------------------------------
    # POLLUTANT LEGEND
    # --------------------------------------------------------

    available_pollutants = sorted(
        filtered["parameter"].unique()
    )

    legend_items = "".join(
        f"""
        <div style="margin:4px 0;">

            <span style="
                display:inline-block;
                width:10px;
                height:10px;
                border-radius:50%;
                background:{POLLUTANT_COLORS.get(name, '#FFFFFF')};
                margin-right:7px;
            "></span>

            {name.upper()}

        </div>
        """
        for name in available_pollutants
    )

    legend_html = f"""
    <div style="
        position:fixed;
        bottom:25px;
        left:25px;
        z-index:9999;
        background:#111827;
        color:#F9FAFB;
        padding:12px 15px;
        border:1px solid #374151;
        border-radius:8px;
        font-size:12px;
    ">

        <b>Pollutant Legend</b>

        {legend_items}

    </div>
    """

    india_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # --------------------------------------------------------
    # DISPLAY MAP
    # --------------------------------------------------------

    st_folium(
        india_map,
        width="stretch",
        height=620,
        returned_objects=[],
        key="airsight_sensor_map",
    )

    latest_time = filtered[
        "datetime_from_utc"
    ].max()

    st.caption(
        "Latest observation in the filtered records: "
        f"{latest_time.strftime('%d %b %Y, %H:%M UTC')}"
    )

    # ========================================================
    # 4. SENSOR MEASUREMENT DETAILS TABLE
    # ========================================================

    st.divider()

    st.subheader("Sensor Measurement Details")

    display_columns = [
        "city",
        "sensor_id",
        "parameter",
        "value",
        "unit",
        "coverage_percent",
        "datetime_from_utc",
        "latitude",
        "longitude",
    ]

    sensor_table = filtered[
        display_columns
    ].copy()

    sensor_table = sensor_table.rename(
        columns={
            "city": "City",
            "sensor_id": "Sensor ID",
            "parameter": "Pollutant",
            "value": "Concentration",
            "unit": "Unit",
            "coverage_percent": "Coverage (%)",
            "datetime_from_utc": "Observation Time (UTC)",
            "latitude": "Latitude",
            "longitude": "Longitude",
        }
    )

    sensor_table["Pollutant"] = (
        sensor_table["Pollutant"].str.upper()
    )

    sensor_table["Observation Time (UTC)"] = (
        sensor_table["Observation Time (UTC)"]
        .dt.strftime("%Y-%m-%d %H:%M")
    )

    st.dataframe(
        sensor_table.sort_values(
            ["City", "Pollutant"]
        ),
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # 5. DOWNLOAD FILTERED SENSOR DATA
    # ========================================================

    csv_data = filtered.copy()

    csv_data["datetime_from_utc"] = (
        csv_data["datetime_from_utc"]
        .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    csv_data = csv_data.sort_values(
        ["city", "parameter"]
    )

    csv_bytes = csv_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Sensor Data (CSV)",
        data=csv_bytes,
        file_name="airsight_sensor_measurements.csv",
        mime="text/csv",
        width="stretch",
    )
