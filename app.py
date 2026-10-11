import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))
import live_monitor


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AirPulse",
    page_icon="🌫️",
    layout="wide"
)


# =========================
# LOAD CSS
# =========================

CSS_PATH = Path(__file__).parent / "css" / "style.css"

with open(CSS_PATH, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# =========================
# PLOTLY DARK THEME
# Matches css/style.css so charts aren't white boxes
# on the dark app background. Applied once, globally,
# to every px.* chart in the app.
# =========================

import plotly.graph_objects as go

AIRPULSE_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#151C23",
        plot_bgcolor="#151C23",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E7ECEF", size=12),
        title_font=dict(family="Space Grotesk, sans-serif", color="#E7ECEF", size=16),
        colorway=[
            "#3FD6C6", "#5B8CFF", "#F5C64B",
            "#F0924A", "#EF6A6A", "#4ADE80", "#B978E8"
        ],
        xaxis=dict(
            gridcolor="#28333C", zerolinecolor="#28333C",
            linecolor="#28333C", tickfont=dict(color="#8694A3")
        ),
        yaxis=dict(
            gridcolor="#28333C", zerolinecolor="#28333C",
            linecolor="#28333C", tickfont=dict(color="#8694A3")
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E7ECEF")),
        hoverlabel=dict(
            bgcolor="#1B242C", font_color="#E7ECEF", bordercolor="#28333C"
        ),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)

px.defaults.template = AIRPULSE_PLOTLY_TEMPLATE


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "katraj_airpulse_episodes_final.csv"
)

ML_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "katraj_airpulse_ml_final.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
)

MODEL_PATH = MODEL_DIR / "airpulse_rf_model.joblib"
IMPUTER_PATH = MODEL_DIR / "airpulse_imputer.joblib"
ENCODER_PATH = MODEL_DIR / "airpulse_encoder.joblib"
SEASONAL_THRESHOLDS_PATH = MODEL_DIR / "seasonal_thresholds.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

SHAP_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "shap_results"
)

GLOBAL_SHAP_PATH = SHAP_DIR / "global_shap_importance.csv"
SHAP_VALUES_PATH = SHAP_DIR / "test_shap_values.csv"
SHAP_METADATA_PATH = SHAP_DIR / "shap_test_metadata.csv"


# ==========================================================
# SEASON FUNCTION
# ==========================================================

def get_season(month):
    """
    Convert month number into the season categories
    used during AirPulse model training.
    """

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-Monsoon"


def add_season_column(dataframe):
    """
    Ensure that the dataframe contains the season column.
    """

    df_copy = dataframe.copy()

    if "timestamp" in df_copy.columns:

        df_copy["timestamp"] = pd.to_datetime(
            df_copy["timestamp"],
            errors="coerce"
        )

        if "season" not in df_copy.columns:
            df_copy["season"] = (
                df_copy["timestamp"]
                .dt.month
                .map(get_season)
            )

        else:
            # Recalculate only missing season values.
            missing_season = df_copy["season"].isna()

            df_copy.loc[missing_season, "season"] = (
                df_copy.loc[missing_season, "timestamp"]
                .dt.month
                .map(get_season)
            )

    return df_copy


# ==========================================================
# LOAD MAIN DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = add_season_column(df)

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Could not load AirPulse dataset: {e}"
    )

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌫️ AirPulse")

st.sidebar.markdown(
    '<div class="airpulse-live"><span class="dot"></span>'
    'Katraj Dairy, Pune · live feed</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Pollution Analysis",
        "Episode Detection",
        "24-Hour Early Warning",
        "Live Monitor",
        "Explainability",
        "Dataset Upload"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AirPulse\n"
    "Explainable ML for Air Pollution\n"
    "Episode Detection & Early Warning"
)


# ==========================================================
# OVERVIEW
# ==========================================================

if page == "Overview":

    st.title("🌫️ AirPulse")

    st.subheader(
        "Explainable Machine Learning for Air Pollution "
        "Episode Detection and 24-Hour Early Warning"
    )

    st.markdown(
        """
        AirPulse analyzes high-frequency air-quality data to detect
        sustained pollution episodes, predict the risk of an episode
        24 hours ahead, and explain model predictions using SHAP.
        """
    )

    st.markdown("---")

    # ======================================================
    # DATASET STATISTICS
    # ======================================================

    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()

    total_records = len(df)

    episode_count = (
        df["episode_label"] == 1
    ).sum()

    unknown_count = (
        df["episode_label"] == -1
    ).sum()

    normal_count = (
        df["episode_label"] == 0
    ).sum()

    pm25_missing = (
        df["PM2.5"].isna()
    ).sum()

    pm25_mean = df["PM2.5"].mean()
    pm25_median = df["PM2.5"].median()
    pm25_max = df["PM2.5"].max()

    # ======================================================
    # DATASET OVERVIEW
    # ======================================================

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Observations",
            f"{total_records:,}"
        )

    with col2:
        st.metric(
            "Detected Episodes",
            f"{episode_count:,}"
        )

    with col3:
        st.metric(
            "Normal Observations",
            f"{normal_count:,}"
        )

    with col4:
        st.metric(
            "Unknown Observations",
            f"{unknown_count:,}"
        )

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "PM2.5 Mean",
            f"{pm25_mean:.2f} µg/m³"
        )

    with col2:
        st.metric(
            "PM2.5 Median",
            f"{pm25_median:.2f} µg/m³"
        )

    with col3:
        st.metric(
            "PM2.5 Maximum",
            f"{pm25_max:.2f} µg/m³"
        )

    with col4:
        st.metric(
            "Missing PM2.5",
            f"{pm25_missing:,}"
        )

    st.markdown("---")

    # ======================================================
    # DATASET DETAILS
    # ======================================================

    st.subheader("Dataset Details")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📍 Monitoring Station")

        st.write(
            "**Station:** Katraj Dairy, Pune"
        )

        st.write(
            "**Monitoring Agency:** MPCB"
        )

        st.write(
            "**Data Frequency:** Hourly"
        )

        st.write(
            f"**Start:** {start_date.strftime('%d %b %Y %H:%M')}"
        )

        st.write(
            f"**End:** {end_date.strftime('%d %b %Y %H:%M')}"
        )

    with col2:

        st.markdown("### 📊 AirPulse Dataset")

        st.write(
            f"**Rows:** {df.shape[0]:,}"
        )

        st.write(
            f"**Columns:** {df.shape[1]}"
        )

        st.write(
            "**Episode threshold:** 71.25 µg/m³"
        )

        st.write(
            "**Minimum episode duration:** 2 hours"
        )

        st.write(
            "**Forecast horizon:** 24 hours"
        )

    st.markdown("---")

    # ======================================================
    # EPISODE LABEL DISTRIBUTION
    # ======================================================

    st.subheader("Episode Label Distribution")

    label_data = pd.DataFrame(
        {
            "Category": [
                "Normal",
                "Episode",
                "Unknown"
            ],
            "Count": [
                normal_count,
                episode_count,
                unknown_count
            ]
        }
    )

    st.bar_chart(
        label_data.set_index("Category")
    )

    st.markdown("---")

    # ======================================================
    # ABOUT AIRPULSE
    # ======================================================

    st.subheader("About AirPulse")

    st.markdown(
        """
        **AirPulse consists of four major stages:**

        1. **Episode Detection**  
           Identifies sustained PM2.5 pollution episodes.

        2. **Feature Engineering**  
           Uses pollutant history, meteorological conditions,
           lag features and rolling statistics.

        3. **24-Hour Early Warning**  
           Uses a Random Forest model to estimate the likelihood
           of an upcoming pollution episode.

        4. **Explainability**  
           Uses SHAP to understand which features contributed
           to individual model predictions.
        """
    )


# ==========================================================
# POLLUTION ANALYSIS
# ==========================================================

elif page == "Pollution Analysis":

    st.title("📊 Pollution Analysis")

    st.markdown(
        """
        Explore the temporal behaviour of air pollutants and their
        relationship with meteorological conditions in the
        Katraj Dairy monitoring dataset.
        """
    )

    st.markdown("---")

    # ======================================================
    # DATE FILTER
    # ======================================================

    st.subheader("Analysis Period")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if (
        isinstance(selected_dates, tuple)
        and len(selected_dates) == 2
    ):

        start_selected = pd.Timestamp(
            selected_dates[0]
        )

        end_selected = (
            pd.Timestamp(selected_dates[1])
            + pd.Timedelta(days=1)
        )

        analysis_df = df[
            (df["timestamp"] >= start_selected)
            &
            (df["timestamp"] < end_selected)
        ].copy()

    else:

        analysis_df = df.copy()

    # ======================================================
    # FILTERED DATA SUMMARY
    # ======================================================

    if not analysis_df.empty:

        st.info(
            f"Showing {len(analysis_df):,} observations "
            f"from {analysis_df['timestamp'].min().strftime('%d %b %Y')} "
            f"to {analysis_df['timestamp'].max().strftime('%d %b %Y')}."
        )

    # ======================================================
    # PM2.5 TIME SERIES
    # ======================================================

    st.subheader("PM2.5 Time Series")

    fig_pm25 = px.line(
        analysis_df,
        x="timestamp",
        y="PM2.5",
        labels={
            "timestamp": "Time",
            "PM2.5": "PM2.5 (µg/m³)"
        },
        title="PM2.5 Concentration Over Time"
    )

    fig_pm25.update_layout(
        hovermode="x unified",
        height=500
    )

    fig_pm25.add_hline(
        y=71.25,
        line_dash="dash",
        annotation_text="Episode threshold: 71.25 µg/m³"
    )

    st.plotly_chart(
        fig_pm25,
        use_container_width=True
    )

    # ======================================================
    # PM10 TIME SERIES
    # ======================================================

    st.subheader("PM10 Time Series")

    fig_pm10 = px.line(
        analysis_df,
        x="timestamp",
        y="PM10",
        labels={
            "timestamp": "Time",
            "PM10": "PM10 (µg/m³)"
        },
        title="PM10 Concentration Over Time"
    )

    fig_pm10.update_layout(
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(
        fig_pm10,
        use_container_width=True
    )

    # ======================================================
    # POLLUTANT COMPARISON
    # ======================================================

    st.markdown("---")

    st.subheader("Pollutant Comparison")

    pollutant_options = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "SO2",
        "CO",
        "OZONE"
    ]

    selected_pollutants = st.multiselect(
        "Select pollutants",
        pollutant_options,
        default=["PM2.5", "PM10", "NO2"]
    )

    if selected_pollutants:

        pollutant_df = analysis_df[
            ["timestamp"] + selected_pollutants
        ].copy()

        pollutant_long = pollutant_df.melt(
            id_vars="timestamp",
            value_vars=selected_pollutants,
            var_name="Pollutant",
            value_name="Concentration"
        )

        fig_pollutants = px.line(
            pollutant_long,
            x="timestamp",
            y="Concentration",
            color="Pollutant",
            labels={
                "timestamp": "Time",
                "Concentration": "Concentration"
            },
            title="Selected Pollutants Over Time"
        )

        fig_pollutants.update_layout(
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(
            fig_pollutants,
            use_container_width=True
        )

    # ======================================================
    # HOURLY PM2.5 PATTERN
    # ======================================================

    st.markdown("---")

    st.subheader("Hourly PM2.5 Pattern")

    hourly_df = (
        analysis_df
        .assign(
            hour=analysis_df["timestamp"].dt.hour
        )
        .groupby("hour")["PM2.5"]
        .mean()
        .reset_index()
    )

    fig_hour = px.line(
        hourly_df,
        x="hour",
        y="PM2.5",
        markers=True,
        labels={
            "hour": "Hour of Day",
            "PM2.5": "Mean PM2.5 (µg/m³)"
        },
        title="Average PM2.5 by Hour of Day"
    )

    fig_hour.update_xaxes(
        dtick=1
    )

    fig_hour.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # ======================================================
    # SEASONAL PM2.5
    # ======================================================

    st.subheader("Seasonal PM2.5 Pattern")

    season_order = [
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon"
    ]

    seasonal_df = (
        analysis_df
        .groupby(
            "season",
            observed=False
        )["PM2.5"]
        .mean()
        .reindex(season_order)
        .reset_index()
    )

    fig_season = px.bar(
        seasonal_df,
        x="season",
        y="PM2.5",
        labels={
            "season": "Season",
            "PM2.5": "Mean PM2.5 (µg/m³)"
        },
        title="Average PM2.5 by Season"
    )

    fig_season.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_season,
        use_container_width=True
    )

    # ======================================================
    # WEATHER RELATIONSHIPS
    # ======================================================

    st.markdown("---")

    st.subheader("PM2.5 and Meteorological Conditions")

    weather_variable = st.selectbox(
        "Select meteorological variable",
        [
            "temperature_C",
            "RH",
            "pressure_hPa",
            "wind_speed",
            "precipitation_mm",
            "solar_radiation_Wm2"
        ]
    )

    weather_df = analysis_df[
        ["PM2.5", weather_variable]
    ].dropna()

    fig_weather = px.scatter(
        weather_df,
        x=weather_variable,
        y="PM2.5",
        opacity=0.35,
        labels={
            weather_variable: weather_variable,
            "PM2.5": "PM2.5 (µg/m³)"
        },
        title=f"PM2.5 vs {weather_variable}"
    )

    fig_weather.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True
    )


# ==========================================================
# EPISODE DETECTION
# ==========================================================

elif page == "Episode Detection":

    st.title("🚨 Episode Detection")

    st.markdown(
        """
        AirPulse defines a pollution episode as a sustained period
        where PM2.5 remains above the episode threshold for at least
        two consecutive valid hourly observations.
        """
    )

    st.markdown("---")

    # ======================================================
    # EPISODE DATA
    # ======================================================

    episode_df = df[
        df["episode_label"] == 1
    ].copy()

    episode_df = episode_df.sort_values(
        "timestamp"
    )

    # ======================================================
    # EPISODE SUMMARY
    # ======================================================

    episode_summary = (
        episode_df
        .groupby("episode_id")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            duration_hours=("timestamp", "count"),
            max_pm25=("PM2.5", "max"),
            mean_pm25=("PM2.5", "mean")
        )
        .reset_index()
    )

    episode_summary = episode_summary.sort_values(
        "start_time"
    )

    # ======================================================
    # KEY METRICS
    # ======================================================

    total_episodes = len(
        episode_summary
    )

    total_episode_hours = (
        episode_summary["duration_hours"].sum()
    )

    average_duration = (
        episode_summary["duration_hours"].mean()
    )

    longest_duration = (
        episode_summary["duration_hours"].max()
    )

    st.subheader("Episode Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Detected Episodes",
            f"{total_episodes:,}"
        )

    with col2:
        st.metric(
            "Episode Hours",
            f"{total_episode_hours:,}"
        )

    with col3:
        st.metric(
            "Average Duration",
            f"{average_duration:.1f} h"
        )

    with col4:
        st.metric(
            "Longest Episode",
            f"{longest_duration:.0f} h"
        )

    st.markdown("---")

    # ======================================================
    # EPISODE TIMELINE
    # ======================================================

    st.subheader("Pollution Episode Timeline")

    timeline_df = df[
        [
            "timestamp",
            "PM2.5",
            "episode_label"
        ]
    ].copy()

    timeline_df["Status"] = timeline_df[
        "episode_label"
    ].map(
        {
            1: "Episode",
            0: "Normal",
            -1: "Unknown"
        }
    )

    fig_episode = px.scatter(
        timeline_df,
        x="timestamp",
        y="PM2.5",
        color="Status",
        labels={
            "timestamp": "Time",
            "PM2.5": "PM2.5 (µg/m³)"
        },
        title="Detected Pollution Episodes"
    )

    fig_episode.add_hline(
        y=71.25,
        line_dash="dash",
        annotation_text="Episode threshold: 71.25 µg/m³"
    )

    fig_episode.update_layout(
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_episode,
        use_container_width=True
    )

    # ======================================================
    # EPISODE DURATION DISTRIBUTION
    # ======================================================

    st.subheader("Episode Duration Distribution")

    fig_duration = px.histogram(
        episode_summary,
        x="duration_hours",
        nbins=20,
        labels={
            "duration_hours": "Episode Duration (hours)"
        },
        title="Distribution of Episode Durations"
    )

    fig_duration.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_duration,
        use_container_width=True
    )

    # ======================================================
    # LONGEST EPISODES
    # ======================================================

    st.markdown("---")

    st.subheader("Longest Pollution Episodes")

    longest_episodes = (
        episode_summary
        .sort_values(
            "duration_hours",
            ascending=False
        )
        .head(10)
        .copy()
    )

    longest_episodes["start_time"] = (
        longest_episodes["start_time"]
        .dt.strftime("%d %b %Y %H:%M")
    )

    longest_episodes["end_time"] = (
        longest_episodes["end_time"]
        .dt.strftime("%d %b %Y %H:%M")
    )

    longest_episodes["max_pm25"] = (
        longest_episodes["max_pm25"]
        .round(2)
    )

    longest_episodes["mean_pm25"] = (
        longest_episodes["mean_pm25"]
        .round(2)
    )

    longest_episodes = longest_episodes.rename(
        columns={
            "episode_id": "Episode ID",
            "start_time": "Start",
            "end_time": "End",
            "duration_hours": "Duration (hours)",
            "max_pm25": "Max PM2.5",
            "mean_pm25": "Mean PM2.5"
        }
    )

    st.dataframe(
        longest_episodes[
            [
                "Episode ID",
                "Start",
                "End",
                "Duration (hours)",
                "Max PM2.5",
                "Mean PM2.5"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # INDIVIDUAL EPISODE EXPLORER
    # ======================================================

    st.markdown("---")

    st.subheader(
        "🔎 Explore an Individual Episode"
    )

    episode_ids = episode_summary[
        "episode_id"
    ].tolist()

    selected_episode = st.selectbox(
        "Select Episode",
        episode_ids
    )

    selected_data = episode_df[
        episode_df["episode_id"] == selected_episode
    ].copy()

    if not selected_data.empty:

        selected_start = selected_data[
            "timestamp"
        ].min()

        selected_end = selected_data[
            "timestamp"
        ].max()

        selected_duration = len(
            selected_data
        )

        selected_max = selected_data[
            "PM2.5"
        ].max()

        selected_mean = selected_data[
            "PM2.5"
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Start",
                selected_start.strftime(
                    "%d %b %Y %H:%M"
                )
            )

        with col2:
            st.metric(
                "End",
                selected_end.strftime(
                    "%d %b %Y %H:%M"
                )
            )

        with col3:
            st.metric(
                "Duration",
                f"{selected_duration} hours"
            )

        with col4:
            st.metric(
                "Maximum PM2.5",
                f"{selected_max:.2f} µg/m³"
            )

        fig_selected = px.line(
            selected_data,
            x="timestamp",
            y="PM2.5",
            markers=True,
            labels={
                "timestamp": "Time",
                "PM2.5": "PM2.5 (µg/m³)"
            },
            title=f"Episode {selected_episode}"
        )

        fig_selected.add_hline(
            y=71.25,
            line_dash="dash",
            annotation_text="Episode threshold"
        )

        fig_selected.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_selected,
            use_container_width=True
        )


# ==========================================================
# 24-HOUR EARLY WARNING
# ==========================================================

elif page == "24-Hour Early Warning":

    st.title("⚠️ 24-Hour Early Warning")

    st.markdown(
        """
        AirPulse uses a Random Forest model to estimate whether the
        monitoring station is likely to enter a pollution episode
        approximately 24 hours ahead.
        """
    )

    st.markdown("---")

    # ======================================================
    # LOAD MODEL ARTIFACTS
    # ======================================================

    @st.cache_resource
    def load_model_artifacts():

        model = joblib.load(
            MODEL_PATH
        )

        imputer = joblib.load(
            IMPUTER_PATH
        )

        encoder = joblib.load(
            ENCODER_PATH
        )

        return model, imputer, encoder

    @st.cache_data
    def load_ml_data():

        ml_df = pd.read_csv(
            ML_DATA_PATH
        )

        ml_df["timestamp"] = pd.to_datetime(
            ml_df["timestamp"],
            errors="coerce"
        )

        # ==================================================
        # IMPORTANT SEASON FIX
        # ==================================================
        #
        # The saved ML dataset should already contain season.
        # If it does not, recreate it from timestamp.
        #

        ml_df = add_season_column(
            ml_df
        )

        return ml_df

    try:

        model, imputer, encoder = (
            load_model_artifacts()
        )

        ml_df = load_ml_data()

    except Exception as e:

        st.error(
            f"Could not load the AirPulse model or ML dataset: {e}"
        )

        st.stop()

    # ======================================================
    # CHECK SEASON
    # ======================================================

    if "season" not in ml_df.columns:

        st.error(
            "The ML dataset does not contain the required "
            "'season' feature."
        )

        st.stop()

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    st.subheader("Model Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Forecast Horizon",
            "24 Hours"
        )

    with col3:
        st.metric(
            "Decision Threshold",
            f"{threshold:.3f}"
        )

    with col4:
        st.metric(
            "Model Features",
            "86"
        )

    st.caption(
        f"Threshold source: {threshold_source}. Episode prevalence "
        "varies sharply by season (Monsoon ~0.2% of hours, "
        "Post-Monsoon ~26%), so AirPulse selects a threshold tuned "
        "for the current observation's season rather than one fixed "
        "value for the whole year -- this improved back-tested test "
        "F1 from 0.55 to 0.70. Each season's threshold was selected "
        "using the validation dataset only."
    )

    st.markdown("---")

    # ======================================================
    # SELECT TIMESTAMP
    # ======================================================

    st.subheader("Select Observation")

    st.write(
        "Select a historical observation to generate a "
        "24-hour early-warning prediction."
    )

    available_timestamps = (
        ml_df["timestamp"]
        .sort_values(ascending=False)
        .tolist()
    )

    selected_timestamp = st.selectbox(
        "Observation timestamp",
        available_timestamps,
        format_func=lambda x: x.strftime(
            "%d %b %Y %H:%M"
        )
    )

    selected_row = ml_df[
        ml_df["timestamp"] == selected_timestamp
    ].iloc[0]

    # ======================================================
    # PREPARE INPUT FEATURES
    # ======================================================

    DROP_COLUMNS = [
        "timestamp",
        "episode_label",
        "episode_id",
        "target_24h"
    ]

    X_input = pd.DataFrame(
        [selected_row]
    ).drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    # ======================================================
    # IMPORTANT FEATURE SEPARATION
    # ======================================================

    categorical_features = [
        "season"
    ]

    # Check season explicitly.
    if "season" not in X_input.columns:

        st.error(
            "Season feature is missing from the selected observation."
        )

        st.stop()

    # Numerical features = everything except season.
    numerical_features = [
        column
        for column in X_input.columns
        if column not in categorical_features
    ]

    X_numeric = X_input[
        numerical_features
    ]

    X_categorical = X_input[
        categorical_features
    ].copy()

    # ======================================================
    # SEASON VALIDATION
    # ======================================================

    valid_seasons = [
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon"
    ]

    season_value = X_categorical[
        "season"
    ].iloc[0]

    if season_value not in valid_seasons:

        # Recalculate from timestamp if necessary.
        season_value = get_season(
            selected_timestamp.month
        )

        X_categorical.loc[
            X_categorical.index[0],
            "season"
        ] = season_value

    # ======================================================
    # APPLY SAME PREPROCESSING AS TRAINING
    # ======================================================

    try:

        X_numeric_imputed = (
            imputer.transform(
                X_numeric
            )
        )

        X_categorical_encoded = (
            encoder.transform(
                X_categorical
            )
        )

    except Exception as e:

        st.error(
            "The selected observation could not be transformed "
            "using the saved AirPulse preprocessing pipeline."
        )

        st.exception(e)

        st.stop()

    # ======================================================
    # COMBINE FEATURES
    # ======================================================

    X_processed = np.hstack(
        [
            X_numeric_imputed,
            X_categorical_encoded
        ]
    )

    # ======================================================
    # FEATURE COUNT CHECK
    # ======================================================

    if X_processed.shape[1] != 86:

        st.error(
            f"Feature mismatch: the model expects 86 processed "
            f"features, but {X_processed.shape[1]} were generated."
        )

        st.stop()

    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    model_score = model.predict_proba(
        X_processed
    )[0, 1]

    # ======================================================
    # SEASON-ADAPTIVE DECISION THRESHOLD
    # ======================================================
    #
    # Episode prevalence varies enormously by season (Monsoon
    # ~0.2% of hours, Post-Monsoon ~26%), so a single global
    # threshold miscalibrates badly across the year. Walk-forward
    # validation showed this directly: the same 0.015 threshold
    # swung from 0% to 78% recall fold to fold purely because of
    # seasonal prevalence shifts, not model quality. A threshold
    # tuned per season (see seasonal_threshold_analysis.py) raised
    # test F1 from 0.55 to 0.70 in back-testing.
    #
    # Falls back to the global threshold if the season-specific
    # file is missing or doesn't cover the observed season.

    GLOBAL_THRESHOLD = 0.015

    try:
        with open(SEASONAL_THRESHOLDS_PATH) as f:
            seasonal_thresholds = json.load(f)
        threshold = seasonal_thresholds.get(
            season_value, GLOBAL_THRESHOLD
        )
        threshold_source = (
            f"season-adaptive ({season_value})"
            if season_value in seasonal_thresholds
            else "global fallback"
        )
    except (FileNotFoundError, json.JSONDecodeError):
        threshold = GLOBAL_THRESHOLD
        threshold_source = "global fallback"

    prediction = (
        1
        if model_score >= threshold
        else 0
    )

    # ======================================================
    # WARNING RESULT
    # ======================================================

    st.subheader("Early-Warning Result")

    if prediction == 1:

        st.error(
            "⚠️ POTENTIAL POLLUTION EPISODE"
        )

        st.write(
            "The model score is above the selected "
            "decision threshold. AirPulse flags this "
            "observation for potential episode risk "
            "24 hours ahead."
        )

    else:

        st.success(
            "✅ NO EPISODE FLAG"
        )

        st.write(
            "The model score is below the selected "
            "decision threshold. AirPulse does not flag "
            "this observation for an episode 24 hours ahead."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Score",
            f"{model_score:.4f}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.3f}"
        )

    with col3:
        st.metric(
            "Prediction",
            "Episode" if prediction == 1 else "Normal"
        )

    st.caption(
        "Important: this Random Forest score is not a calibrated "
        "probability. It should be interpreted as a model score, "
        "not as a literal percentage probability."
    )

    # ======================================================
    # CURRENT CONDITIONS
    # ======================================================

    st.markdown("---")

    st.subheader("Current Observation")

    current_columns = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "SO2",
        "CO",
        "OZONE",
        "temperature_C",
        "RH",
        "pressure_hPa",
        "wind_speed"
    ]

    available_current_columns = [
        column
        for column in current_columns
        if column in selected_row.index
    ]

    current_data = {}

    for column in available_current_columns:

        value = selected_row[column]

        if pd.notna(value):

            current_data[column] = round(
                float(value),
                2
            )

        else:

            current_data[column] = None

    current_df = pd.DataFrame(
        [current_data]
    )

    st.dataframe(
        current_df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # HISTORICAL OUTCOME
    # ======================================================

    st.markdown("---")

    st.subheader("Historical Verification")

    actual_target = selected_row[
        "target_24h"
    ]

    if actual_target == 1:

        st.warning(
            "For this historical observation, an actual "
            "pollution episode was recorded 24 hours later."
        )

    elif actual_target == 0:

        st.info(
            "For this historical observation, no pollution "
            "episode was recorded 24 hours later."
        )

    else:

        st.warning(
            "The 24-hour future outcome is unknown for this "
            "observation because the required target was unavailable."
        )

    # ======================================================
    # PREDICTION VS ACTUAL
    # ======================================================

    if actual_target in [0, 1]:

        if prediction == 1 and actual_target == 1:

            st.success(
                "Historical result: TRUE POSITIVE"
            )

        elif prediction == 1 and actual_target == 0:

            st.warning(
                "Historical result: FALSE POSITIVE"
            )

        elif prediction == 0 and actual_target == 1:

            st.warning(
                "Historical result: FALSE NEGATIVE"
            )

        else:

            st.success(
                "Historical result: TRUE NEGATIVE"
            )

    # ======================================================
    # RUNNING TRACK RECORD (predicted vs actual over time)
    # ======================================================

    st.markdown("---")
    st.subheader("Model Track Record Over Time")

    st.caption(
        "Predicted vs actual outcome for every test-set hour, scored "
        "with the deployed model and each hour's season-adaptive "
        "threshold -- the same evaluation basis as the reported "
        "test metrics, viewable as a timeline rather than a single "
        "confusion matrix."
    )

    @st.cache_data
    def load_verification_timeline():
        df = pd.read_csv(
            Path(__file__).parent / "data" / "processed"
            / "verification_timeline.csv"
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    try:
        timeline = load_verification_timeline()

        outcome_colors = {
            "True Positive": "#4ADE80",
            "True Negative": "#5B8CFF",
            "False Positive": "#F5C64B",
            "False Negative": "#EF6A6A",
        }

        col_a, col_b, col_c, col_d = st.columns(4)
        counts = timeline["outcome"].value_counts()

        with col_a:
            st.metric("True Positives", int(counts.get("True Positive", 0)))
        with col_b:
            st.metric("False Positives", int(counts.get("False Positive", 0)))
        with col_c:
            st.metric("False Negatives", int(counts.get("False Negative", 0)))
        with col_d:
            st.metric("True Negatives", int(counts.get("True Negative", 0)))

        fig_track = px.scatter(
            timeline,
            x="timestamp",
            y="model_score",
            color="outcome",
            color_discrete_map=outcome_colors,
            hover_data=["season", "threshold_used"],
            title="Model score over time, colored by outcome"
        )

        fig_track.add_scatter(
            x=timeline["timestamp"],
            y=timeline["threshold_used"],
            mode="lines",
            name="Season threshold",
            line=dict(color="#7E8DA0", dash="dot")
        )

        st.plotly_chart(fig_track, use_container_width=True)

        st.caption(
            "False negatives (missed episodes) are the costliest error "
            "for an early-warning system -- note where they cluster "
            "above by hovering, and cross-reference with the season "
            "shown."
        )

    except FileNotFoundError:
        st.warning(
            "Verification timeline not found. Run "
            "`code/build_verification_timeline.py` to generate it."
        )


# ==========================================================
# EXPLAINABILITY
# ==========================================================

elif page == "Live Monitor":

    st.title("Live Monitor")

    st.caption(
        "Pulls current + recent hourly air quality and weather data for "
        "Katraj, Pune from Open-Meteo's free Air Quality and Weather APIs "
        "(no key required), builds the same features used in training, "
        "and scores them with the deployed model -- end to end, on demand."
    )

    st.info(
        "**Known limitations of this live feed** (documented, not hidden): "
        "Open-Meteo's pollutant data comes from a global atmospheric model "
        "(CAMS), not a ground sensor at Katraj specifically, so values will "
        "differ somewhat from a CPCB monitor reading. NO (nitric oxide) "
        "isn't available from this source and is imputed. This "
        "demonstrates the live-monitoring architecture end to end; a "
        "production deployment would ideally pair it with an actual CPCB/ "
        "CAAQMS real-time feed once API access is arranged."
    )

    @st.cache_resource
    def load_live_model_artifacts():
        return (
            joblib.load(MODEL_PATH),
            joblib.load(IMPUTER_PATH),
            joblib.load(ENCODER_PATH),
        )

    live_model, live_imputer, live_encoder = load_live_model_artifacts()

    with open(SCHEMA_PATH) as f:
        live_feature_schema = json.load(f)

    refresh = st.button("🔄 Fetch latest reading", type="primary")

    cache_key = "live_monitor_result"

    if refresh or cache_key not in st.session_state:

        with st.spinner("Fetching live data and running inference..."):

            try:
                feature_row, raw_df, latest_ts = (
                    live_monitor.get_live_feature_row()
                )

                num_cols = live_feature_schema["numerical_columns"]
                cat_cols = live_feature_schema["categorical_columns"]

                X_num = live_imputer.transform(
                    pd.DataFrame([feature_row])[num_cols]
                )
                X_cat = live_encoder.transform(
                    pd.DataFrame([feature_row])[cat_cols]
                )
                X_live = np.hstack([X_num, X_cat])

                live_score = live_model.predict_proba(X_live)[0, 1]

                season_now = feature_row["season"]

                try:
                    with open(SEASONAL_THRESHOLDS_PATH) as f:
                        seasonal_thresholds_live = json.load(f)
                    live_threshold = seasonal_thresholds_live.get(
                        season_now, live_feature_schema["threshold"]
                    )
                except (FileNotFoundError, json.JSONDecodeError):
                    live_threshold = live_feature_schema["threshold"]

                st.session_state[cache_key] = {
                    "feature_row": feature_row,
                    "raw_df": raw_df,
                    "latest_ts": latest_ts,
                    "score": live_score,
                    "threshold": live_threshold,
                    "season": season_now,
                    "error": None,
                }

            except Exception as fetch_error:

                st.session_state[cache_key] = {
                    "error": str(fetch_error)
                }

    result = st.session_state.get(cache_key)

    if result and result.get("error"):

        st.error(
            f"Couldn't fetch live data: {result['error']}\n\n"
            "This is usually a network issue (no internet access from "
            "wherever Streamlit is running) or Open-Meteo being "
            "temporarily unavailable. Try again shortly."
        )

    elif result:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Latest usable reading",
                result["latest_ts"].strftime("%d %b, %H:%M")
            )

        with col2:
            st.metric(
                "24h episode probability",
                f"{result['score']:.1%}"
            )

        with col3:
            st.metric(
                "Season-adaptive threshold",
                f"{result['threshold']:.3f}",
                help=f"Season detected: {result['season']}"
            )

        is_episode = result["score"] >= result["threshold"]

        if is_episode:
            st.warning(
                f"⚠️ **Elevated risk**: model score "
                f"({result['score']:.3f}) is above the "
                f"{result['season']} threshold "
                f"({result['threshold']:.3f}). A sustained PM2.5 episode "
                f"is more likely than usual in the next 24 hours."
            )
        else:
            st.success(
                f"✅ **Normal range**: model score "
                f"({result['score']:.3f}) is below the "
                f"{result['season']} threshold "
                f"({result['threshold']:.3f})."
            )

        st.subheader("Recent raw readings")

        display_cols = [
            "timestamp", "PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE",
            "temperature_C", "wind_speed", "pressure_hPa"
        ]

        st.dataframe(
            result["raw_df"][display_cols].tail(24),
            use_container_width=True,
            hide_index=True
        )

        fig_live = px.line(
            result["raw_df"].tail(72),
            x="timestamp",
            y="PM2.5",
            title="PM2.5 — last 72 hours (live feed)"
        )
        fig_live.add_hline(
            y=71.25,
            line_dash="dash",
            annotation_text="Episode threshold: 71.25 µg/m³"
        )
        st.plotly_chart(fig_live, use_container_width=True)


elif page == "Explainability":

    st.title("🔍 Model Explainability")

    st.markdown(
        """
        AirPulse uses SHAP (SHapley Additive exPlanations) to understand
        which features contributed to the Random Forest model's predictions.
        """
    )

    st.markdown("---")

    # ======================================================
    # LOAD SHAP DATA
    # ======================================================

    @st.cache_data
    def load_shap_data():

        global_shap = pd.read_csv(
            GLOBAL_SHAP_PATH
        )

        shap_values = pd.read_csv(
            SHAP_VALUES_PATH
        )

        shap_metadata = pd.read_csv(
            SHAP_METADATA_PATH
        )

        shap_metadata["timestamp"] = pd.to_datetime(
            shap_metadata["timestamp"],
            errors="coerce"
        )

        return (
            global_shap,
            shap_values,
            shap_metadata
        )

    try:

        (
            global_shap,
            shap_values,
            shap_metadata
        ) = load_shap_data()

    except Exception as e:

        st.error(
            f"Could not load SHAP results: {e}"
        )

        st.stop()

    # ======================================================
    # GLOBAL EXPLAINABILITY
    # ======================================================

    st.subheader("Global Feature Importance")

    st.markdown(
        """
        Global SHAP importance summarizes how strongly each feature
        influenced the model across the evaluated test observations.
        Higher mean absolute SHAP values indicate greater average
        influence on the model output.
        """
    )

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=20,
        value=10
    )

    top_features = (
        global_shap
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .head(top_n)
        .sort_values(
            "mean_abs_shap",
            ascending=True
        )
    )

    fig_global = px.bar(
        top_features,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={
            "mean_abs_shap": "Mean |SHAP value|",
            "feature": "Feature"
        },
        title="Top Features by Global SHAP Importance"
    )

    fig_global.update_layout(
        height=550
    )

    st.plotly_chart(
        fig_global,
        use_container_width=True
    )

    # ======================================================
    # TOP FEATURES TABLE
    # ======================================================

    st.subheader("Top Contributing Features")

    display_global = (
        global_shap
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    display_global["mean_abs_shap"] = (
        display_global["mean_abs_shap"]
        .round(6)
    )

    st.dataframe(
        display_global,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # INTERPRETATION
    # ======================================================

    st.info(
        """
        **Interpretation:** SHAP values describe the contribution
        of features to the model's prediction. They indicate model
        behaviour and should not be interpreted as causal effects.
        """
    )

    # ======================================================
    # LOCAL EXPLANATION
    # ======================================================

    st.markdown("---")

    st.subheader(
        "🔎 Local Prediction Explanation"
    )

    st.markdown(
        """
        Local SHAP analysis explains why the model produced a
        particular prediction for an individual observation.
        """
    )

    if len(shap_values) == len(shap_metadata):

        local_df = shap_metadata.copy()

        local_df["_row_index"] = range(
            len(local_df)
        )

    else:

        st.warning(
            "SHAP values and metadata have different lengths."
        )

        local_df = pd.DataFrame()

    if not local_df.empty:

        local_timestamp = st.selectbox(
            "Select observation for local explanation",
            local_df["timestamp"].tolist(),
            format_func=lambda x: x.strftime(
                "%d %b %Y %H:%M"
            )
        )

        selected_index = local_df.index[
            local_df["timestamp"]
            == local_timestamp
        ][0]

        selected_shap = shap_values.iloc[
            selected_index
        ]

        contribution_data = pd.DataFrame(
            {
                "feature": selected_shap.index,
                "SHAP value": pd.to_numeric(
                    selected_shap.values,
                    errors="coerce"
                )
            }
        )

        contribution_data = (
            contribution_data
            .dropna()
        )

        contribution_data[
            "absolute_value"
        ] = (
            contribution_data[
                "SHAP value"
            ].abs()
        )

        top_local = (
            contribution_data
            .sort_values(
                "absolute_value",
                ascending=False
            )
            .head(10)
            .sort_values(
                "SHAP value"
            )
        )

        fig_local = px.bar(
            top_local,
            x="SHAP value",
            y="feature",
            orientation="h",
            labels={
                "SHAP value": "SHAP contribution",
                "feature": "Feature"
            },
            title="Top Local Feature Contributions"
        )

        fig_local.add_vline(
            x=0,
            line_dash="dash"
        )

        fig_local.update_layout(
            height=500
        )

        st.plotly_chart(
            fig_local,
            use_container_width=True
        )

        st.caption(
            "Positive SHAP values push the model output upward; "
            "negative values push it downward."
        )

    # ======================================================
    # PRECOMPUTED CASE STUDIES
    # ======================================================

    st.markdown("---")

    st.subheader(
        "📌 Example Case Studies"
    )

    st.markdown(
        """
        AirPulse also includes precomputed local explanations
        for representative true-positive, false-positive and
        false-negative cases.
        """
    )

    case = st.selectbox(
        "Select case",
        [
            "True Positive",
            "False Positive",
            "False Negative"
        ]
    )

    case_files = {

        "True Positive":
            SHAP_DIR
            / "local_cases"
            / "plots"
            / "local_shap_tp.png",

        "False Positive":
            SHAP_DIR
            / "local_cases"
            / "plots"
            / "local_shap_fp.png",

        "False Negative":
            SHAP_DIR
            / "local_cases"
            / "plots"
            / "local_shap_fn.png"
    }

    selected_case_file = case_files[
        case
    ]

    if selected_case_file.exists():

        st.image(
            str(selected_case_file),
            use_container_width=True
        )

    else:

        st.warning(
            f"Case-study plot not found: {selected_case_file}"
        )


# ==========================================================
# DATASET UPLOAD
# ==========================================================

elif page == "Dataset Upload":

    st.title("📁 Dataset Upload")

    st.markdown(
        """
        Upload an air-quality dataset to inspect its structure and
        validate whether it contains the information required by
        the AirPulse workflow.
        """
    )

    st.markdown("---")

    # ======================================================
    # FILE UPLOAD
    # ======================================================

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV or Excel air-quality dataset to begin."
        )

    else:

        try:

            # ==================================================
            # READ FILE
            # ==================================================

            is_csv = uploaded_file.name.lower().endswith(".csv")

            def _read(skip=0):
                uploaded_file.seek(0)
                if is_csv:
                    return pd.read_csv(uploaded_file, skiprows=skip)
                return pd.read_excel(uploaded_file, skiprows=skip)

            uploaded_df = _read()

            # ==================================================
            # AUTO-DETECT CPCB-STYLE METADATA HEADER ROWS
            # Raw CPCB/CAAQMS exports put several report-metadata
            # rows (station name, frequency, etc.) above the real
            # column header, which otherwise reads in as
            # "Unnamed: 1", "Unnamed: 2", ... Scan the first 20
            # rows for the real header (a row containing a
            # recognizable timestamp label) and re-read from there.
            # ==================================================

            unnamed_ratio = (
                uploaded_df.columns.astype(str)
                .str.startswith("Unnamed")
                .mean()
            )

            if unnamed_ratio > 0.5:

                uploaded_file.seek(0)
                probe = (
                    pd.read_csv(uploaded_file, header=None, nrows=20)
                    if is_csv else
                    pd.read_excel(uploaded_file, header=None, nrows=20)
                )

                header_labels = {"from date", "date", "timestamp", "datetime"}
                detected_row = None

                for i in range(len(probe)):
                    row_values = (
                        probe.iloc[i].astype(str).str.strip().str.lower()
                    )
                    if row_values.isin(header_labels).any():
                        detected_row = i
                        break

                if detected_row is not None:
                    uploaded_df = _read(skip=detected_row)
                    st.info(
                        f"Detected {detected_row} metadata row(s) above "
                        f"the real header (common in raw CPCB/CAAQMS "
                        f"exports) and re-read the file from row "
                        f"{detected_row + 1}."
                    )

            st.success(
                f"Successfully loaded: {uploaded_file.name}"
            )

            # ==================================================
            # BASIC INFORMATION
            # ==================================================

            st.subheader(
                "Dataset Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Rows",
                    f"{uploaded_df.shape[0]:,}"
                )

            with col2:

                st.metric(
                    "Columns",
                    f"{uploaded_df.shape[1]:,}"
                )

            with col3:

                st.metric(
                    "Missing Values",
                    f"{uploaded_df.isna().sum().sum():,}"
                )

            # ==================================================
            # COLUMN INFORMATION
            # ==================================================

            st.markdown("---")

            st.subheader(
                "Column Information"
            )

            column_info = pd.DataFrame(
                {
                    "Column": uploaded_df.columns,

                    "Data Type": [
                        str(dtype)
                        for dtype in uploaded_df.dtypes
                    ],

                    "Missing Values": [
                        uploaded_df[column].isna().sum()
                        for column in uploaded_df.columns
                    ]
                }
            )

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True
            )

            # ==================================================
            # PREVIEW
            # ==================================================

            st.markdown("---")

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            # ==================================================
            # TIMESTAMP VALIDATION
            # ==================================================

            st.markdown("---")

            st.subheader(
                "AirPulse Schema Validation"
            )

            # ==================================================
            # NORMALIZED COLUMN MATCHING
            # Different stations/networks name the same quantity
            # differently ("PM2.5 (ug/m3)" vs "PM2.5", "Ozone" vs
            # "OZONE", "From Date" vs "timestamp"). Strip units,
            # punctuation and casing before matching so the schema
            # check reflects whether the DATA is usable, not
            # whether the column names happen to match verbatim.
            # ==================================================

            import re

            def _normalize(col):
                c = str(col).strip().lower()
                c = re.sub(r"\(.*?\)", "", c)   # drop "(ug/m3)" etc.
                c = c.replace(".", "").replace("_", " ")
                c = re.sub(r"\s+", " ", c).strip()
                return c

            normalized_to_original = {
                _normalize(c): c for c in uploaded_df.columns
            }

            TIMESTAMP_ALIASES = ["timestamp", "datetime", "date", "from date"]

            POLLUTANT_ALIASES = {
                "PM2.5": ["pm25", "pm 25", "pm2 5"],
                "PM10": ["pm10", "pm 10"],
                "NO": ["no"],
                "NO2": ["no2"],
                "NH3": ["nh3"],
                "SO2": ["so2"],
                "CO": ["co"],
                "OZONE": ["ozone", "o3"],
            }

            timestamp_column = None

            for alias in TIMESTAMP_ALIASES:

                if alias in normalized_to_original:

                    timestamp_column = normalized_to_original[alias]
                    break

            invalid_timestamps = 0

            if timestamp_column is None:

                st.error(
                    "❌ Timestamp column not found."
                )

            else:

                st.success(
                    f"✓ Timestamp column detected: "
                    f"`{timestamp_column}`"
                )

                converted_timestamp = pd.to_datetime(
                    uploaded_df[
                        timestamp_column
                    ],
                    errors="coerce"
                )

                invalid_timestamps = (
                    converted_timestamp.isna().sum()
                )

                if invalid_timestamps == 0:

                    st.success(
                        "✓ All timestamp values are valid."
                    )

                else:

                    st.warning(
                        f"⚠️ {invalid_timestamps:,} "
                        "timestamp values could not be parsed."
                    )

            # ==================================================
            # POLLUTANT SCHEMA
            # ==================================================

            st.subheader(
                "Air-Quality Column Validation"
            )

            required_pollutants = list(POLLUTANT_ALIASES.keys())

            pollutant_matches = {}

            for canonical, aliases in POLLUTANT_ALIASES.items():
                for alias in aliases:
                    if alias in normalized_to_original:
                        pollutant_matches[canonical] = (
                            normalized_to_original[alias]
                        )
                        break

            # NOx is commonly reported as one combined value rather than
            # separate NO / NO2 readings. It isn't force-mapped to either
            # (that would misrepresent the data) -- flagged separately
            # so the user knows why NO/NO2 may still show as missing.
            nox_present = "nox" in normalized_to_original

            available_pollutants = list(pollutant_matches.keys())

            missing_pollutants = [
                column
                for column in required_pollutants
                if column not in pollutant_matches
            ]

            if pollutant_matches:

                mapping_table = pd.DataFrame(
                    {
                        "AirPulse column": list(pollutant_matches.keys()),
                        "Matched from uploaded file": list(
                            pollutant_matches.values()
                        ),
                    }
                )

                st.dataframe(
                    mapping_table,
                    use_container_width=True,
                    hide_index=True
                )

            if nox_present and (
                "NO" in missing_pollutants or "NO2" in missing_pollutants
            ):
                st.caption(
                    "Note: this file reports a combined `NOx` value "
                    "rather than separate NO / NO2 readings, so those "
                    "two stay flagged as missing rather than being "
                    "guessed."
                )

            if len(missing_pollutants) == 0:

                st.success(
                    "✓ All required pollutant columns "
                    "are present."
                )

            else:

                st.warning(
                    "Missing pollutant columns: "
                    + ", ".join(
                        missing_pollutants
                    )
                )

            # ==================================================
            # DATASET STATUS
            # ==================================================

            st.markdown("---")

            if (
                timestamp_column is not None
                and len(available_pollutants)
                == len(required_pollutants)
                and invalid_timestamps == 0
            ):

                st.success(
                    """
                    ✅ Dataset passed the basic AirPulse
                    schema validation.

                    The dataset contains a timestamp and all
                    required pollutant measurements.
                    """
                )

                st.caption(
                    "Full episode detection, feature engineering "
                    "and 24-hour forecasting for new datasets "
                    "will be connected in a later pipeline stage."
                )

            else:

                st.warning(
                    """
                    ⚠️ Dataset requires additional preprocessing
                    before it can be used by the complete
                    AirPulse forecasting pipeline.
                    """
                )

        except Exception as e:

            st.error(
                f"Could not read the uploaded dataset: {e}"
            )