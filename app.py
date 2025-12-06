import os
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Dispersion Trading",
    layout="wide"
)

@st.cache_data
def load_data(factors_mtime: float, master_mtime: float):
    """Load and merge factors + master data.

    factors_mtime / master_mtime are only used so that the cache
    automatically refreshes whenever either CSV file changes.
    """
    df_factors = pd.read_csv("data2/dispersion_factors.csv")
    df_spy = pd.read_csv("data2/master_dispersion_data.csv")

    if "date_ny" not in df_factors.columns:
        raise ValueError("dispersion_factors.csv is missing 'date_ny' column.")
    if "date_ny" not in df_spy.columns:
        raise ValueError("master_dispersion_data.csv is missing 'date_ny' column.")

    df_factors["date"] = pd.to_datetime(
        df_factors["date_ny"], format="%Y-%m-%d", errors="raise"
    )
    df_spy["date"] = pd.to_datetime(
        df_spy["date_ny"], dayfirst=True, errors="raise"
    )

    df = (
        pd.merge(df_factors, df_spy, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df

# 👇 new lines: compute mtimes and pass into cached function
factors_mtime = os.path.getmtime("data2/dispersion_factors.csv")
master_mtime = os.path.getmtime("data2/master_dispersion_data.csv")
df = load_data(factors_mtime, master_mtime)

df = load_data()

st.title("Concentration of Dispersion Trading in the S&P 500")
st.markdown("Date Range: Post-Covid (starting mid-2020)")
st.markdown("Select series to display:")

rename_map = {
    "Dispersion_Z_60d": "Dispersion_Index",
    "Dispersion_Z_EWMA_20": "EWMA_20",
    "Dispersion_Z_EWMA_30": "EWMA_30",
    "Dispersion_Z_EWMA_60": "EWMA_60",
    "Dispersion_Z_SMA_60": "SMA_60",
    "Dispersion_Z_SMA_90": "SMA_90",
    "Dispersion_Z_SMA_120": "SMA_120",
}

dispersion_cols = [c for c in rename_map.keys() if c in df.columns]

if "SPY_IV" not in df.columns:
    st.error("SPY_IV column not found in master_dispersion_data.csv.")
    st.stop()

dispersion_labels = [rename_map[c] for c in dispersion_cols]

default_labels = []
if "Dispersion_Z_60d" in dispersion_cols:
    default_labels = [rename_map["Dispersion_Z_60d"]]

selected_labels = st.multiselect(
    "Dispersion series",
    options=dispersion_labels,
    default=default_labels,
    label_visibility="collapsed",
)

selected_cols = [
    orig
    for orig, label in rename_map.items()
    if label in selected_labels and orig in df.columns
]

slider_col, _, _ = st.columns([1, 2, 2])

with slider_col:
    st.markdown("Adjust left y-axis range:")
    half_range = st.slider(
        "Half-range",
        min_value=1.0,
        max_value=3.0,
        value=3.0,
        step=0.1,
        label_visibility="collapsed",
        help="Adjust to zoom dispersion series: range is [-value, +value].",
    )

y_min, y_max = -half_range, half_range

series_domain = ["SPY_IV"] + dispersion_labels
series_range = [
    "red",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
][: len(series_domain)]
color_scale = alt.Scale(domain=series_domain, range=series_range)
zoom = alt.selection_interval(bind="scales", encodings=["x"])

charts = []

has_left = len(selected_cols) > 0

if has_left:
    rename_subset = {orig: rename_map[orig] for orig in selected_cols}
    left_df = (
        df[["date"] + selected_cols]
        .rename(columns=rename_subset)
        .melt(id_vars="date", var_name="Series", value_name="Value")
    )

    left_chart = (
        alt.Chart(left_df)
        .mark_line()
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(format="%b %Y", labelAngle=-45, labelFlush=True),
            ),
            y=alt.Y(
                "Value:Q",
                title="",
                scale=alt.Scale(domain=[y_min, y_max]),
            ),
            color=alt.Color(
                "Series:N",
                scale=color_scale,
                legend=alt.Legend(title="Series"),
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
    )
    charts.append(left_chart)

spy_df = df[["date", "SPY_IV"]].melt(
    id_vars="date", var_name="Series", value_name="Value"
)
spy_df["Series"] = "SPY_IV"

spy_axis = alt.Axis(
    title="SPY_IV",
    orient="right" if has_left else "left",
)

spy_chart = (
    alt.Chart(spy_df)
    .mark_line()
    .encode(
        x=alt.X(
            "date:T",
            axis=alt.Axis(
                title="Date",
                format="%b %Y",
                labelAngle=-45,
                labelFlush=True,
                tickCount="month",
            ),
        ),
        y=alt.Y("Value:Q", axis=spy_axis),
        color=alt.Color(
            "Series:N",
            scale=color_scale,
            legend=alt.Legend(title="Series"),
        ),
        tooltip=["date:T", "Series:N", "Value:Q"],
    )
)
charts.append(spy_chart)

if has_left:
    chart = (
        alt.layer(*charts)
        .resolve_scale(y="independent")
        .add_params(zoom)
        .properties(height=500)
    )
else:
    chart = (
        alt.layer(*charts)
        .add_params(zoom)
        .properties(height=500)
    )

st.altair_chart(chart, width="stretch")
