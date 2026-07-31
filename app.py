from itertools import combinations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_FILE = "heart_failure_clinical_records_dataset.csv"
OUTCOME_COLUMN = "DEATH_EVENT"

LABELS = {
    "low_ef": "Low ejection fraction (<35%)",
    "high_creatinine": "High creatinine (>1.5 mg/dL)",
    "low_sodium": "Low sodium (<135 mEq/L)",
    "age_70_plus": "Age ≥70",
    "anaemia": "Anaemia",
    "high_cpk": "High CPK (>400 U/L)",
    "low_platelets": "Low platelets (<150,000)",
    "diabetes": "Diabetes",
    "high_blood_pressure": "High blood pressure",
    "sex": "Male",
    "smoking": "Smoking",
}

FEATURE_ORDER = list(LABELS)

DEFAULT_FEATURES = [
    "low_ef",
    "high_creatinine",
    "low_sodium",
    "age_70_plus",
    "anaemia",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    required_columns = {
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
        OUTCOME_COLUMN,
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    # Feature engineering used in the project
    df["low_ef"] = (df["ejection_fraction"] < 35).astype(int)
    df["high_creatinine"] = (df["serum_creatinine"] > 1.5).astype(int)
    df["low_sodium"] = (df["serum_sodium"] < 135).astype(int)
    df["age_70_plus"] = (df["age"] >= 70).astype(int)
    df["high_cpk"] = (
        df["creatinine_phosphokinase"] > 400
    ).astype(int)
    df["low_platelets"] = (df["platelets"] < 150000).astype(int)

    return df


def summarize_group(
    df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, int, int, float]:
    if not selected_features:
        subset = df.copy()
    else:
        mask = df[selected_features].eq(1).all(axis=1)
        subset = df.loc[mask].copy()

    patient_count = len(subset)
    deaths = int(subset[OUTCOME_COLUMN].sum())

    mortality = (
        float(subset[OUTCOME_COLUMN].mean() * 100)
        if patient_count > 0
        else 0.0
    )

    return subset, patient_count, deaths, mortality


@st.cache_data
def build_combination_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for size in range(1, 5):
        for combo in combinations(FEATURE_ORDER, size):
            _, patients, deaths, mortality = summarize_group(
                df,
                list(combo),
            )

            if patients == 0:
                continue

            rows.append(
                {
                    "features": list(combo),
                    "label": " + ".join(LABELS[x] for x in combo),
                    "number_of_factors": size,
                    "patients": patients,
                    "deaths": deaths,
                    "mortality": mortality,
                }
            )

    return pd.DataFrame(rows)


def build_progressive_chart(
    df: pd.DataFrame,
    selected_features: list[str],
    overall_mortality: float,
) -> go.Figure:
    rows = []
    cumulative_features = []

    for feature in selected_features:
        cumulative_features.append(feature)

        _, patients, deaths, mortality = summarize_group(
            df,
            cumulative_features,
        )

        rows.append(
            {
                "combination": " + ".join(
                    LABELS[x] for x in cumulative_features
                ),
                "mortality": mortality,
                "patients": patients,
                "deaths": deaths,
                "added_feature": LABELS[feature],
            }
        )

    chart_df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["mortality"],
            y=chart_df["combination"],
            orientation="h",
            text=[
                f"{mortality:.1f}% | n={patients}"
                for mortality, patients in zip(
                    chart_df["mortality"],
                    chart_df["patients"],
                )
            ],
            textposition="outside",
            customdata=chart_df[
                ["patients", "deaths", "added_feature"]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mortality: %{x:.1f}%<br>"
                "Patients: %{customdata[0]}<br>"
                "Deaths: %{customdata[1]}<br>"
                "Added factor: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=overall_mortality,
        line_dash="dash",
        annotation_text=(
            f"Overall cohort mortality: {overall_mortality:.1f}%"
        ),
        annotation_position="top",
    )

    fig.update_layout(
        title="Progressive Mortality as Risk Factors Are Added",
        xaxis_title="Observed mortality (%)",
        yaxis_title="Cumulative risk-factor combination",
        xaxis_range=[0, 105],
        height=max(440, len(chart_df) * 80),
        margin=dict(l=20, r=80, t=80, b=40),
    )

    return fig


st.set_page_config(
    page_title="CHF Mortality Risk Explorer",
    page_icon="❤️",
    layout="wide",
)

st.title("CHF Mortality Risk Explorer")

st.caption(
    "Explore observed mortality among patients who share selected "
    "clinical risk factors. This is an exploratory educational tool, "
    "not a validated clinical prediction model."
)

try:
    data = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

overall_mortality = float(data[OUTCOME_COLUMN].mean() * 100)
combination_table = build_combination_table(data)

high_risk_combinations = (
    combination_table[
        (combination_table["mortality"] > 80)
        & (combination_table["patients"] >= 6)
    ]
    .sort_values(
        ["number_of_factors", "mortality", "patients"],
        ascending=[True, False, False],
    )
    .reset_index(drop=True)
)

if "selected_features" not in st.session_state:
    st.session_state.selected_features = DEFAULT_FEATURES.copy()

with st.sidebar:
    st.header("Selected Risk Factors")

    selected_features = []

    for feature in FEATURE_ORDER:
        checked = st.checkbox(
            LABELS[feature],
            value=feature in st.session_state.selected_features,
            key=f"check_{feature}",
        )

        if checked:
            selected_features.append(feature)

    st.session_state.selected_features = selected_features

    st.divider()
    st.header("High-Risk Combination")

    if high_risk_combinations.empty:
        st.info(
            "No combinations met the current high-risk and "
            "minimum-sample criteria."
        )
    else:
        preset_labels = high_risk_combinations["label"].tolist()

        selected_preset = st.selectbox(
            "Choose a preset",
            preset_labels,
        )

        if st.button(
            "Load selected combination",
            use_container_width=True,
        ):
            preset_features = high_risk_combinations.loc[
                high_risk_combinations["label"] == selected_preset,
                "features",
            ].iloc[0]

            st.session_state.selected_features = preset_features

            for feature in FEATURE_ORDER:
                st.session_state[f"check_{feature}"] = (
                    feature in preset_features
                )

            st.rerun()

subset, patient_count, deaths, mortality = summarize_group(
    data,
    st.session_state.selected_features,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Observed mortality",
    f"{mortality:.1f}%",
    f"{mortality - overall_mortality:+.1f} points vs cohort",
)

metric_2.metric("Matching patients", patient_count)
metric_3.metric("Deaths", deaths)
metric_4.metric(
    "Overall cohort mortality",
    f"{overall_mortality:.1f}%",
)

selected_names = [
    LABELS[feature]
    for feature in st.session_state.selected_features
]

if selected_names:
    st.subheader("Current Risk Profile")
    st.write(" • ".join(selected_names))
else:
    st.info(
        "No risk factors are selected. Results represent the entire cohort."
    )

if patient_count == 0:
    st.warning(
        "No patients match this combination. Remove one or more factors."
    )
elif st.session_state.selected_features:
    figure = build_progressive_chart(
        data,
        st.session_state.selected_features,
        overall_mortality,
    )
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("View matching patient records"):
        st.dataframe(subset, use_container_width=True)

st.divider()

st.markdown(
    """
### Interpretation and limitations

The percentages shown are observed outcomes within this 299-patient
dataset. They are not individualized predictions. Results for small
subgroups may be unstable, and associations should not be interpreted
as causal or used for clinical decision-making.
"""
)
