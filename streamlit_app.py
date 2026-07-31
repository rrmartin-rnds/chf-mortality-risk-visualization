from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CHF Mortality Risk Explorer",
    page_icon="❤️",
    layout="wide",
)


# ---------------------------------------------------------
# PROJECT SETTINGS
# ---------------------------------------------------------
DATA_FILE = Path(__file__).with_name(
    "heart_failure_clinical_records_dataset.csv"
)
OUTCOME_COL = "DEATH_EVENT"

LABEL_MAP = {
    "low_ef": "Low EF",
    "high_creatinine": "High Creatinine",
    "low_sodium": "Low Sodium",
    "age_70_plus": "Age ≥ 70",
    "anaemia": "Anaemia",
    "high_cpk": "High CPK",
    "low_platelets": "Low Platelets",
    "diabetes": "Diabetes",
    "high_blood_pressure": "High Blood Pressure",
    "sex": "Male",
    "smoking": "Smoking",
}

FEATURE_ORDER = [
    "low_ef",
    "high_creatinine",
    "low_sodium",
    "age_70_plus",
    "anaemia",
    "high_cpk",
    "low_platelets",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]

HIGH_RISK_VARS = {
    "low_ef",
    "high_creatinine",
    "low_sodium",
    "age_70_plus",
    "anaemia",
    "high_blood_pressure",
}

DEFAULT_FEATURES = [
    "low_ef",
    "high_creatinine",
    "low_sodium",
    "age_70_plus",
    "anaemia",
    "high_blood_pressure",
]


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.name}. "
            "Place the CSV in the same repository folder as streamlit_app.py."
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
        OUTCOME_COL,
    }

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    df = df.copy()

    # Thresholds used in the original project notebook.
    df["low_ef"] = (df["ejection_fraction"] < 35).astype(int)
    df["high_creatinine"] = (
        df["serum_creatinine"] > 1.5
    ).astype(int)
    df["low_sodium"] = (df["serum_sodium"] < 135).astype(int)
    df["age_70_plus"] = (df["age"] >= 70).astype(int)
    df["high_cpk"] = (
        df["creatinine_phosphokinase"] > 250
    ).astype(int)
    df["low_platelets"] = (df["platelets"] < 150000).astype(int)

    return df


def summarize_group(
    dataframe: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, int, int, float]:
    if not features:
        subset = dataframe.copy()
    else:
        mask = dataframe[features].eq(1).all(axis=1)
        subset = dataframe.loc[mask].copy()

    n_patients = len(subset)
    deaths = (
        int(subset[OUTCOME_COL].sum())
        if n_patients > 0
        else 0
    )
    mortality = (
        float(subset[OUTCOME_COL].mean() * 100)
        if n_patients > 0
        else np.nan
    )

    return subset, n_patients, deaths, mortality


@st.cache_data
def build_combination_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    # This mirrors the original notebook's one- through four-way
    # combination analysis using the eight primary risk variables.
    combination_features = [
        "low_ef",
        "high_creatinine",
        "low_sodium",
        "age_70_plus",
        "anaemia",
        "smoking",
        "high_blood_pressure",
        "diabetes",
    ]

    short_labels = {
        "low_ef": "EF",
        "high_creatinine": "Creat",
        "low_sodium": "Na",
        "age_70_plus": "70+",
        "anaemia": "Anaemia",
        "smoking": "Smoking",
        "high_blood_pressure": "BP",
        "diabetes": "DM",
    }

    rows = []

    for size in range(1, 5):
        for combo in combinations(combination_features, size):
            _, n_patients, deaths, mortality = summarize_group(
                dataframe,
                list(combo),
            )

            if n_patients == 0:
                continue

            rows.append(
                {
                    "features": list(combo),
                    "combo_label": " + ".join(
                        short_labels[feature]
                        for feature in combo
                    ),
                    "n_factors": size,
                    "n_patients": n_patients,
                    "deaths": deaths,
                    "mortality_rate": mortality,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------
def build_summary_chart(
    mortality: float,
    overall_mortality: float,
) -> go.Figure:
    selected_value = 0.0 if pd.isna(mortality) else mortality

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Overall cohort", "Selected combination"],
            y=[overall_mortality, selected_value],
            text=[
                f"{overall_mortality:.1f}%",
                "N/A" if pd.isna(mortality) else f"{mortality:.1f}%",
            ],
            textposition="outside",
            hovertemplate="%{x}<br>Mortality: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Observed Mortality Compared with the Full Cohort",
        yaxis_title="Mortality rate (%)",
        yaxis_range=[0, 105],
        showlegend=False,
        height=390,
        margin=dict(l=20, r=20, t=70, b=30),
    )

    return fig


def build_progressive_chart(
    dataframe: pd.DataFrame,
    selected_features: list[str],
    overall_mortality: float,
) -> go.Figure | None:
    if not selected_features:
        return None

    rows = []
    current_features: list[str] = []
    previous_mortality = 0.0

    for feature in selected_features:
        current_features.append(feature)
        _, n_patients, deaths, mortality = summarize_group(
            dataframe,
            current_features,
        )

        mortality = 0.0 if pd.isna(mortality) else mortality
        change = mortality - previous_mortality

        rows.append(
            {
                "step_label": " + ".join(
                    LABEL_MAP[item]
                    for item in current_features
                ),
                "added_feature": LABEL_MAP[feature],
                "n_patients": n_patients,
                "deaths": deaths,
                "mortality": mortality,
                "previous_risk": previous_mortality,
                "change": change,
            }
        )

        previous_mortality = mortality

    chart_df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Previous cumulative risk",
            y=chart_df["step_label"],
            x=chart_df["previous_risk"],
            orientation="h",
            customdata=chart_df[
                ["n_patients", "deaths", "added_feature"]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Previous risk: %{x:.1f}%<br>"
                "Patients: %{customdata[0]}<br>"
                "Deaths: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Change from added attribute",
            y=chart_df["step_label"],
            x=chart_df["change"],
            orientation="h",
            text=[
                (
                    f"{mortality:.1f}% | "
                    f"+{added_feature} | n={n_patients}"
                )
                for mortality, added_feature, n_patients
                in zip(
                    chart_df["mortality"],
                    chart_df["added_feature"],
                    chart_df["n_patients"],
                )
            ],
            textposition="outside",
            customdata=chart_df[
                ["mortality", "n_patients", "deaths", "added_feature"]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Observed mortality: %{customdata[0]:.1f}%<br>"
                "Added factor: %{customdata[3]}<br>"
                "Patients: %{customdata[1]}<br>"
                "Deaths: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=overall_mortality,
        line_dash="dash",
        annotation_text=(
            f"Overall mortality: {overall_mortality:.1f}%"
        ),
        annotation_position="top",
    )

    fig.update_layout(
        title=(
            "Progressive Analysis: Change in Mortality "
            "as Attributes Are Added"
        ),
        barmode="stack",
        xaxis_title="Observed mortality rate (%)",
        yaxis_title="Cumulative combination",
        xaxis_range=[0, 110],
        height=max(470, 78 * len(chart_df)),
        margin=dict(l=20, r=155, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "selected_features" not in st.session_state:
    st.session_state.selected_features = DEFAULT_FEATURES.copy()

# A preset is applied at the beginning of a fresh rerun, before any
# checkbox widgets are created. Streamlit does not allow changing a
# widget's session-state value after that widget has already rendered.
if "pending_features" in st.session_state:
    pending_features = list(st.session_state.pop("pending_features"))
    st.session_state.selected_features = pending_features

    for feature in FEATURE_ORDER:
        st.session_state[f"feature_{feature}"] = (
            feature in pending_features
        )

for feature in FEATURE_ORDER:
    key = f"feature_{feature}"
    if key not in st.session_state:
        st.session_state[key] = (
            feature in st.session_state.selected_features
        )


def apply_selected_features(features: list[str]) -> None:
    # Store the requested preset under a non-widget key. The checkbox
    # values will be updated safely at the start of the next rerun.
    st.session_state.pending_features = list(features)


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #243b55 0%, #141e30 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 18px;
        margin-bottom: 14px;
    ">
        <div style="font-size: 30px; font-weight: 700;">
            CHF Mortality Risk Explorer
        </div>
        <div style="font-size: 15px; margin-top: 7px;">
            Start with a full risk profile, then remove variables to see
            which factors are driving the observed mortality rate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This exploratory tool summarizes observed outcomes in a "
    "299-patient dataset. It is not a validated clinical prediction "
    "model and should not be used for patient-care decisions."
)

try:
    df = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.subheader("Dataset Preview")
st.caption(
    "The table below shows the first five records from the "
    "Heart Failure Clinical Records dataset."
)
st.dataframe(
    df.head(),
    use_container_width=True,
    hide_index=True,
)

overall_mortality = float(df[OUTCOME_COL].mean() * 100)
combo_df = build_combination_table(df)

high_risk_combo_df = (
    combo_df[
        (combo_df["mortality_rate"] > 80)
        & (combo_df["n_patients"] >= 6)
    ]
    .sort_values(
        ["n_factors", "mortality_rate", "n_patients"],
        ascending=[True, False, False],
    )
    .reset_index(drop=True)
)

control_column, preset_column = st.columns([1, 1.25])

with control_column:
    st.subheader("Selected Risk Factors")
    st.caption(
        "Highlighted factors were identified as higher-risk variables "
        "during the exploratory analysis."
    )

    selected_features = []

    for feature in FEATURE_ORDER:
        label = LABEL_MAP[feature]

        if feature in HIGH_RISK_VARS:
            label = f"🔴 {label}"

        if st.checkbox(
            label,
            key=f"feature_{feature}",
        ):
            selected_features.append(feature)

    st.session_state.selected_features = selected_features

with preset_column:
    st.subheader("Select High-Risk Combination")
    st.write(
        "Choose a previously identified high-risk group, then load it "
        "into the explorer."
    )

    if high_risk_combo_df.empty:
        st.warning(
            "No combinations met the >80% mortality and "
            "n ≥ 6 criteria."
        )
    else:
        high_risk_combo_df = high_risk_combo_df.copy()
        high_risk_combo_df["display_label"] = (
            high_risk_combo_df["combo_label"]
            + " ("
            + high_risk_combo_df["mortality_rate"].map(
                lambda value: f"{value:.1f}%"
            )
            + ")"
        )

        preset_label = st.selectbox(
            "High-risk group",
            high_risk_combo_df["display_label"].tolist(),
            label_visibility="collapsed",
        )

        if st.button(
            "Load Selected Combo",
            type="primary",
            use_container_width=True,
        ):
            preset_features = high_risk_combo_df.loc[
                high_risk_combo_df["display_label"] == preset_label,
                "features",
            ].iloc[0]

            apply_selected_features(preset_features)
            st.rerun()

selected_features = st.session_state.selected_features
subset, n_patients, deaths, mortality = summarize_group(
    df,
    selected_features,
)

selected_labels = [
    LABEL_MAP[feature]
    for feature in selected_features
]
high_risk_selected = [
    LABEL_MAP[feature]
    for feature in selected_features
    if feature in HIGH_RISK_VARS
]

st.divider()
st.subheader("Current Combination")

st.write(
    " + ".join(selected_labels)
    if selected_labels
    else "All Patients"
)

st.caption(
    "Highlighted high-risk variables selected: "
    + (
        ", ".join(high_risk_selected)
        if high_risk_selected
        else "None selected"
    )
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Patients", n_patients)
metric_2.metric("Deaths", deaths)
metric_3.metric(
    "Observed mortality",
    "N/A" if pd.isna(mortality) else f"{mortality:.1f}%",
)
metric_4.metric(
    "Overall cohort mortality",
    f"{overall_mortality:.1f}%",
)

if n_patients == 0:
    st.warning(
        "No patients match the selected combination. "
        "Remove one or more variables."
    )
else:
    summary_chart = build_summary_chart(
        mortality,
        overall_mortality,
    )
    st.plotly_chart(
        summary_chart,
        use_container_width=True,
    )

    progressive_chart = build_progressive_chart(
        df,
        selected_features,
        overall_mortality,
    )

    if progressive_chart is not None:
        st.plotly_chart(
            progressive_chart,
            use_container_width=True,
        )

    with st.expander("View matching records"):
        st.dataframe(
            subset,
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.markdown(
    """
### Interpretation limitations

- Mortality percentages are observed subgroup outcomes, not individual
  patient predictions.
- Several combinations contain small patient groups and may produce
  unstable percentages.
- Associations are exploratory and should not be interpreted as causal.
- The tool has not been externally validated.
"""
)
