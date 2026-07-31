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
def build_progressive_chart(dataframe, selected_features, outcome_col="DEATH_EVENT"):
    if len(selected_features) == 0:
        return None

    rows = []
    current_features = []
    previous_mortality = 0

    for feat in selected_features:
        current_features.append(feat)
        _, n, deaths, mortality = summarize_group(dataframe, current_features, outcome_col)

        mortality = 0 if pd.isna(mortality) else mortality
        added_change = mortality - previous_mortality

        rows.append({
            "step_label": " + ".join(label_map[f] for f in current_features),
            "added_feature": label_map[feat],
            "n_patients": n,
            "mortality": mortality,
            "base_portion": previous_mortality,
            "added_portion": added_change
        })

        previous_mortality = mortality

    prog_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.9 * len(prog_df))))

    # Base cumulative portion
    ax.barh(
        prog_df["step_label"],
        prog_df["base_portion"],
        color="#cfe8ff",
        edgecolor="black",
        linewidth=0.8,
        label="Previous cumulative risk"
    )

    # Newly added change
    ax.barh(
        prog_df["step_label"],
        prog_df["added_portion"],
        left=prog_df["base_portion"],
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
        label="Change from added attribute"
    )

    ax.axvline(overall_mortality, linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Observed Mortality Rate (%)")
    ax.set_title("Progressive Analysis: Change in Mortality as Attributes Are Added")

    for i, row in prog_df.iterrows():
        ax.text(
            row["mortality"] + 1,
            i,
            f"{row['mortality']:.1f}% | +{row['added_feature']} | n={row['n_patients']}",
            va="center",
            fontsize=10
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=9,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# -------------------------
# CREATE BINARY RISK FEATURES
# -------------------------
df["low_ef"] = (df["ejection_fraction"] < 35).astype(int)
df["high_creatinine"] = (df["serum_creatinine"] > 1.5).astype(int)
df["low_sodium"] = (df["serum_sodium"] < 135).astype(int)
df["age_70_plus"] = (df["age"] >= 70).astype(int)
df["high_cpk"] = (df["creatinine_phosphokinase"] > 250).astype(int)
df["low_platelets"] = (df["platelets"] < 150000).astype(int)

# -------------------------
# LABELS AND FEATURE SETUP
# -------------------------
label_map = {
    "low_ef": "Low EF",
    "high_creatinine": "High Creatinine",
    "low_sodium": "Low Sodium",
    "age_70_plus": "Age ≥ 70",
    "high_cpk": "High CPK",
    "low_platelets": "Low Platelets",
    "anaemia": "Anaemia",
    "diabetes": "Diabetes",
    "high_blood_pressure": "High Blood Pressure",
    "sex": "Male",
    "smoking": "Smoking"
}

feature_order = [
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
    "smoking"
]

high_risk_vars = ["low_ef", "high_creatinine", "low_sodium", "age_70_plus", "anaemia", "high_blood_pressure"]
default_features = ["low_ef", "high_creatinine", "low_sodium", "age_70_plus", "anaemia", "high_blood_pressure"]

outcome_col = "DEATH_EVENT"
overall_mortality = df[outcome_col].mean() * 100

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def summarize_group(dataframe, features, outcome_col="DEATH_EVENT"):
    if len(features) == 0:
        subset = dataframe.copy()
    else:
        subset = dataframe[dataframe[features].eq(1).all(axis=1)].copy()

    n = len(subset)
    deaths = int(subset[outcome_col].sum()) if n > 0 else 0
    mortality = subset[outcome_col].mean() * 100 if n > 0 else np.nan
    return subset, n, deaths, mortality

def pretty_combo_text(features):
    if not features:
        return "All Patients"
    return " + ".join(label_map[f] for f in features)

def make_summary_html(selected_features, n, deaths, mortality):
    combo_text = pretty_combo_text(selected_features)
    mortality_text = f"{mortality:.1f}%" if pd.notna(mortality) else "N/A"
    mortality_color = "#c0392b" if pd.notna(mortality) and mortality >= overall_mortality else "#1f618d"

    high_risk_selected = [label_map[f] for f in selected_features if f in high_risk_vars]
    high_risk_note = ", ".join(high_risk_selected) if high_risk_selected else "None selected"

    return f"""
    <div style="
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
        border: 1px solid #d6e4ff;
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 14px;
        font-family: Arial, sans-serif;
    ">
        <div style="font-size: 22px; font-weight: 700; color: #1f2d3d; margin-bottom: 10px;">
            CHF Mortality Risk Explorer
        </div>
        <div style="font-size: 14px; color: #4a4a4a; margin-bottom: 8px;">
            <b>Current combination:</b> {combo_text}
        </div>
        <div style="font-size: 13px; color: #6b7280; margin-bottom: 14px;">
            <b>Highlighted high-risk variables selected:</b> {high_risk_note}
        </div>
        <div style="display: flex; gap: 14px; flex-wrap: wrap;">
            <div style="
                background: white;
                border-radius: 14px;
                padding: 12px 18px;
                min-width: 150px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                <div style="font-size: 12px; color: #6b7280;">Patients</div>
                <div style="font-size: 24px; font-weight: 700; color: #111827;">{n}</div>
            </div>
            <div style="
                background: white;
                border-radius: 14px;
                padding: 12px 18px;
                min-width: 150px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                <div style="font-size: 12px; color: #6b7280;">Deaths</div>
                <div style="font-size: 24px; font-weight: 700; color: #111827;">{deaths}</div>
            </div>
            <div style="
                background: white;
                border-radius: 14px;
                padding: 12px 18px;
                min-width: 170px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                <div style="font-size: 12px; color: #6b7280;">Observed Mortality</div>
                <div style="font-size: 24px; font-weight: 700; color: {mortality_color};">{mortality_text}</div>
            </div>
            <div style="
                background: white;
                border-radius: 14px;
                padding: 12px 18px;
                min-width: 190px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                <div style="font-size: 12px; color: #6b7280;">Overall Dataset Mortality</div>
                <div style="font-size: 24px; font-weight: 700; color: #374151;">{overall_mortality:.1f}%</div>
            </div>
        </div>
    </div>
    """

def build_main_bar_matplotlib(mortality):
    value = 0 if pd.isna(mortality) else mortality

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(["Current Group"], [value], edgecolor="black", linewidth=1)

    ax.axhline(overall_mortality, linestyle="--", color="gray", linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mortality Rate (%)")
    ax.set_title("Observed Mortality for Current Combination")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.text(
        0.4,
        overall_mortality + 1,
        f"Overall mortality = {overall_mortality:.1f}%",
        fontsize=10,
        color="gray"
    )

    plt.tight_layout()
    return fig

def build_progressive_chart(dataframe, selected_features, outcome_col="DEATH_EVENT"):
    if len(selected_features) == 0:
        return None

    rows = []
    current_features = []
    previous_mortality = 0

    for feat in selected_features:
        current_features.append(feat)
        _, n, deaths, mortality = summarize_group(dataframe, current_features, outcome_col)

        mortality = 0 if pd.isna(mortality) else mortality
        added_change = mortality - previous_mortality

        rows.append({
            "step_label": " + ".join(label_map[f] for f in current_features),
            "added_feature": label_map[feat],
            "n_patients": n,
            "mortality": mortality,
            "base": previous_mortality,
            "added": added_change
        })

        previous_mortality = mortality

    prog_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.9 * len(prog_df))))

    # BLUE = previous cumulative
    ax.barh(
        prog_df["step_label"],
        prog_df["base"],
        color="#cfe8ff",
        edgecolor="black"
    )

    # RED = new added effect
    ax.barh(
        prog_df["step_label"],
        prog_df["added"],
        left=prog_df["base"],
        color="#d62728",
        edgecolor="black"
    )

    ax.axvline(overall_mortality, linestyle="--", color="gray", linewidth=1.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Observed Mortality Rate (%)")
    ax.set_title("Progressive Analysis: Change in Mortality as Attributes Are Added")

    for i, row in prog_df.iterrows():
        ax.text(
            row["mortality"] + 1,
            i,
            f"{row['mortality']:.1f}% | +{row['added_feature']} | n={row['n_patients']}",
            va="center",
            fontsize=10
        )

    ax.legend(
        ["Previous cumulative risk", "Change from added attribute"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig

# -------------------------
# CHECKBOX CONTROLS
# -------------------------
checkbox_inputs = {}
checkbox_rows = []

legend_html = widgets.HTML("""
<div style="font-family:Arial, sans-serif; font-size:13px; margin-bottom:10px; line-height:1.5;">
    <span style="color:#c0392b; font-weight:600;">● High-risk variables identified in earlier analysis</span><br>
    <span style="color:#374151;">● Other available variables for exploration</span>
</div>
""")

for feat in feature_order:
    cb = widgets.Checkbox(
        value=(feat in default_features),
        indent=False,
        layout=widgets.Layout(width="28px")
    )
    checkbox_inputs[feat] = cb

    if feat in high_risk_vars:
        label_html = f"<span style='color:#c0392b; font-weight:600; font-size:14px;'>{label_map[feat]}</span>"
    else:
        label_html = f"<span style='color:#374151; font-size:14px;'>{label_map[feat]}</span>"

    row = widgets.HBox(
        [cb, widgets.HTML(value=label_html)],
        layout=widgets.Layout(align_items="center", margin="2px 0")
    )
    checkbox_rows.append(row)

controls_box = widgets.VBox(
    [
        widgets.HTML("<b style='font-size:16px; font-family:Arial;'>Selected Risk Factors</b>"),
        legend_html
    ] + checkbox_rows,
    layout=widgets.Layout(
        width="290px",
        padding="16px",
        border="1px solid #d1d9e6",
        border_radius="16px",
        background_color="white"
    )
)

title_html = widgets.HTML("""
<div style="
    background: linear-gradient(135deg, #243b55 0%, #141e30 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 18px;
    font-family: Arial, sans-serif;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    margin-bottom: 10px;
">
    <div style="font-size: 24px; font-weight: 700;">CHF Mortality Risk Explorer</div>
    <div style="font-size: 14px; margin-top: 6px; opacity: 0.92;">
        Start with a full risk profile, then remove variables to see which factors are truly driving mortality risk.
    </div>
</div>
""")

instruction_html = widgets.HTML("""
<div style="
    font-family: Arial, sans-serif;
    font-size: 14px;
    color: #374151;
    padding: 4px 2px 12px 2px;
">
    <b>How to use:</b> Uncheck a variable to remove it from the current risk combination.
</div>
""")

output = widgets.Output()

# -------------------------
# DASHBOARD UPDATE
# -------------------------
def update_dashboard(*args):
    selected_features = [feat for feat in feature_order if checkbox_inputs[feat].value]

    with output:
        plt.close('all')
        clear_output(wait=True)

        _, n, deaths, mortality = summarize_group(df, selected_features, outcome_col)

        display(HTML(make_summary_html(selected_features, n, deaths, mortality)))

        if n == 0:
            display(HTML("""
            <div style="
                background:#fff4f4;
                color:#8a1c1c;
                border:1px solid #f3c2c2;
                padding:16px;
                border-radius:14px;
                font-family:Arial,sans-serif;
                font-size:14px;
            ">
                No patients match the currently selected combination. Try removing one or more variables.
            </div>
            """))
            return

        fig_prog = build_progressive_chart(df, selected_features, outcome_col)
        if fig_prog is not None:
            display(fig_prog)
            plt.close(fig_prog)


for cb in checkbox_inputs.values():
    cb.observe(update_dashboard, names="value")

# -------------------------
# HIGH-RISK COMBO SELECTOR
# -------------------------
high_risk_combo_df = combo_df[
    (combo_df["mortality_rate"] > 80) &
    (combo_df["n_patients"] >= 6)
].copy()

high_risk_combo_df = high_risk_combo_df.sort_values(
    by=["n_factors", "mortality_rate", "n_patients"],
    ascending=[True, False, False]
).reset_index(drop=True)

combo_lookup = {}
combo_options = []

for _, row in high_risk_combo_df.iterrows():
    display_text = f"{row['combo_label']} ({row['mortality_rate']:.1f}%)"
    combo_options.append(display_text)
    combo_lookup[display_text] = list(row["combo"])

combo_selector_title = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:16px;font-weight:700;margin-bottom:10px;">
Select High-Risk Combination
</div>
""")

combo_selector_help = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:13px;color:#4b5563;margin-bottom:10px;line-height:1.4;">
Choose a high-risk group, then click the button to load it into the tool.
</div>
""")

combo_selector_widget = widgets.Select(
    options=combo_options,
    rows=min(14, len(combo_options)),
    layout=widgets.Layout(width="340px", height="255px")
)

load_combo_button = widgets.Button(
    description="Load Selected Combo",
    button_style="primary",
    layout=widgets.Layout(width="190px", height="36px")
)

combo_status = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:12px;color:#6b7280;margin-top:8px;">
Ready
</div>
""")

_is_loading_combo = False

def load_selected_combo(_):
    global _is_loading_combo

    selected_label = combo_selector_widget.value
    if selected_label is None:
        combo_status.value = """
        <div style="font-family:Arial,sans-serif;font-size:12px;color:#b91c1c;margin-top:8px;">
        No combination selected.
        </div>
        """
        return

    selected_combo = combo_lookup[selected_label]

    _is_loading_combo = True
    try:
        for feat in checkbox_inputs:
            checkbox_inputs[feat].value = False

        for feat in selected_combo:
            checkbox_inputs[feat].value = True
    finally:
        _is_loading_combo = False

    update_dashboard()

    combo_status.value = f"""
    <div style="font-family:Arial,sans-serif;font-size:12px;color:#065f46;margin-top:8px;">
    Loaded: {selected_label}
    </div>
    """

load_combo_button.on_click(load_selected_combo)

combo_selector_box = widgets.VBox(
    [
        combo_selector_title,
        combo_selector_help,
        combo_selector_widget,
        widgets.Box([load_combo_button], layout=widgets.Layout(margin="10px 0 0 0")),
        combo_status
    ],
    layout=widgets.Layout(
        width="370px",
        padding="16px",
        border="1px solid #d1d9e6",
        border_radius="16px",
        background_color="white"
    )
)

for feat, cb in checkbox_inputs.items():
    try:
        cb.unobserve_all()
    except Exception:
        pass

    def _make_observer():
        def _observer(change):
            if change["name"] == "value" and not _is_loading_combo:
                update_dashboard()
        return _observer

    cb.observe(_make_observer(), names="value")



st.metric("Mortality rate", f"{mortality:.1f}%")

st.write(f"Patients remaining: {len(filtered)}")

# HIGH-RISK COMBO SELECTOR
# integrated with dashboard
# =========================

import ipywidgets as widgets
from IPython.display import display, clear_output

# Build high-risk combo list from combo_df
high_risk_combo_df = combo_df[
    (combo_df["mortality_rate"] > 80) &
    (combo_df["n_patients"] >= 6)
].copy()

high_risk_combo_df = high_risk_combo_df.sort_values(
    by=["n_factors", "mortality_rate", "n_patients"],
    ascending=[True, False, False]
).reset_index(drop=True)

combo_lookup = {}
combo_options = []

for _, row in high_risk_combo_df.iterrows():
    display_text = f"{row['combo_label']} ({row['mortality_rate']:.1f}%)"
    combo_options.append(display_text)
    combo_lookup[display_text] = list(row["combo"])

combo_selector_title = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:16px;font-weight:700;margin-bottom:10px;">
Select High-Risk Combination
</div>
""")

combo_selector_help = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:13px;color:#4b5563;margin-bottom:10px;line-height:1.4;">
Choose a high-risk group, then click the button to load it into the tool.
</div>
""")

combo_selector_widget = widgets.Select(
    options=combo_options,
    rows=min(14, len(combo_options)),
    layout=widgets.Layout(width="340px", height="255px")
)

load_combo_button = widgets.Button(
    description="Load Selected Combo",
    button_style="primary",
    layout=widgets.Layout(width="190px", height="36px")
)

combo_status = widgets.HTML("""
<div style="font-family:Arial,sans-serif;font-size:12px;color:#6b7280;margin-top:8px;">
Ready
</div>
""")

# Prevent multiple refreshes while boxes are being set
_is_loading_combo = False

def load_selected_combo(_):
    global _is_loading_combo

    selected_label = combo_selector_widget.value
    if selected_label is None:
        combo_status.value = """
        <div style="font-family:Arial,sans-serif;font-size:12px;color:#b91c1c;margin-top:8px;">
        No combination selected.
        </div>
        """
        return

    selected_combo = combo_lookup[selected_label]

    _is_loading_combo = True
    try:
        # Clear all first
        for feat in checkbox_inputs:
            checkbox_inputs[feat].value = False

        # Apply selected combo
        for feat in selected_combo:
            checkbox_inputs[feat].value = True
    finally:
        _is_loading_combo = False

    # Single refresh after all boxes are updated

    update_dashboard()

    combo_status.value = f"""
    <div style="font-family:Arial,sans-serif;font-size:12px;color:#065f46;margin-top:8px;">
    Loaded: {selected_label}
    </div>
    """

load_combo_button.on_click(load_selected_combo)

combo_selector_box = widgets.VBox(
    [
        combo_selector_title,
        combo_selector_help,
        combo_selector_widget,
        widgets.Box([load_combo_button], layout=widgets.Layout(margin="10px 0 0 0")),
        combo_status
    ],
    layout=widgets.Layout(
        width="370px",
        padding="16px",
        border="1px solid #d1d9e6",
        border_radius="16px",
        background_color="white"
    )
)

# Optional: if your checkbox observers call update_dashboard directly,
# redefine them to ignore updates during combo loading.
# Only run this part if checkbox_inputs already exists.
for feat, cb in checkbox_inputs.items():
    try:
        cb.unobserve_all()
    except Exception:
        pass

    def _make_observer():
        def _observer(change):
            if change["name"] == "value" and not _is_loading_combo:
                update_dashboard()
        return _observer

    cb.observe(_make_observer(), names="value")


