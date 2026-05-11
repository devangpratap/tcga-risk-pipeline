import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Config & Data Loading
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TCGA Risk Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    features = pd.read_csv(DATA_DIR / "patient_features.csv")
    risk_labels = pd.read_csv(DATA_DIR / "risk_labels.csv")
    metrics = pd.read_csv(DATA_DIR / "model_metrics.csv")
    predictions = pd.read_csv(DATA_DIR / "model_predictions.csv")
    feat_meta = pd.read_csv(DATA_DIR / "feature_metadata.csv")
    dq = pd.read_csv(DATA_DIR / "data_quality.csv")
    return features, risk_labels, metrics, predictions, feat_meta, dq

features, risk_labels, metrics, predictions, feat_meta, dq = load_data()

# Merge for convenience
patients = risk_labels.merge(
    features.drop(columns=["risk_label"], errors="ignore"),
    on="patient_barcode", how="left"
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Risk badges */
    .risk-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-ambiguous {
        background: linear-gradient(135deg, #636e72, #b2bec3);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Pipeline step cards */
    .pipe-step {
        background: #1a1a2e;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .pipe-step h4 { margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .pipe-step p { margin: 0; font-size: 0.8rem; color: #8892a4; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f0f1a;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
COLORS = {
    "high_risk": "#ff416c",
    "low_risk": "#38ef7d",
    "ambiguous": "#636e72",
    "accent1": "#00d2ff",
    "accent2": "#3a7bd5",
    "accent3": "#a855f7",
    "bg": "#0e1117",
    "card_bg": "#1a1a2e",
    "grid": "rgba(255,255,255,0.04)",
    "text": "#c9d1d9",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#1a1a2e", font_size=12),
)

RISK_COLORS = {"high_risk": COLORS["high_risk"], "low_risk": COLORS["low_risk"], "ambiguous": COLORS["ambiguous"]}


def metric_card(value, label):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def risk_badge(label):
    cls = {"high_risk": "risk-high", "low_risk": "risk-low"}.get(label, "risk-ambiguous")
    display = label.replace("_", " ").title()
    return f'<span class="{cls}">{display}</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## TCGA Risk Pipeline")
    st.caption("Multimodal cancer risk prediction")
    st.markdown("---")

    labeled = risk_labels[risk_labels["risk_label"].isin(["high_risk", "low_risk"])]
    cancer_types = sorted(labeled["cancer_type_abbrev"].dropna().unique())

    selected_cancer = st.selectbox(
        "Filter by cancer type",
        ["All"] + cancer_types,
        index=0,
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem; color:#555;'>Built by Devang Pratap Singh<br>"
        "Florida Institute of Technology</p>",
        unsafe_allow_html=True,
    )

# Apply filter
if selected_cancer != "All":
    patients_filtered = patients[patients["cancer_type_abbrev"] == selected_cancer]
    risk_filtered = risk_labels[risk_labels["cancer_type_abbrev"] == selected_cancer]
    preds_filtered = predictions[predictions["patient_barcode"].isin(risk_filtered["patient_barcode"])]
else:
    patients_filtered = patients
    risk_filtered = risk_labels
    preds_filtered = predictions


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# TCGA Multimodal Risk Prediction")
st.caption("Late-fusion model combining histopathology image embeddings, pathology report text, and clinical staging")

# Top-level metrics
labeled_f = risk_filtered[risk_filtered["risk_label"].isin(["high_risk", "low_risk"])]
n_patients = len(risk_filtered)
n_cancer_types = risk_filtered["cancer_type_abbrev"].nunique()
n_high = (risk_filtered["risk_label"] == "high_risk").sum()
n_low = (risk_filtered["risk_label"] == "low_risk").sum()
pct_high = n_high / max(len(labeled_f), 1) * 100

cols = st.columns(4)
with cols[0]:
    st.markdown(metric_card(f"{n_patients:,}", "Total Patients"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(metric_card(str(n_cancer_types), "Cancer Types"), unsafe_allow_html=True)
with cols[2]:
    st.markdown(metric_card(f"{n_high:,}", "High Risk"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(metric_card(f"{n_low:,}", "Low Risk"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Patient Explorer", "Cohort Analytics", "Pipeline"])

# ========================== TAB 1: Patient Explorer ========================
with tab1:
    st.markdown("### Patient Explorer")

    # Patient selector
    available_patients = patients_filtered["patient_barcode"].dropna().unique()
    selected_patient = st.selectbox(
        "Select patient barcode",
        sorted(available_patients),
        index=0,
    )

    patient_row = patients_filtered[patients_filtered["patient_barcode"] == selected_patient].iloc[0]
    patient_preds = predictions[predictions["patient_barcode"] == selected_patient]

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("#### Profile")
        rl = patient_row["risk_label"]
        st.markdown(f"**Risk:** {risk_badge(rl)}", unsafe_allow_html=True)
        st.markdown(f"**Cancer type:** `{patient_row.get('cancer_type_abbrev', 'N/A')}`")
        st.markdown(f"**OS days:** {patient_row.get('os_days', 'N/A')}")
        st.markdown(f"**Vital status:** {'Deceased' if patient_row.get('os_event') == 1 else 'Alive/Censored'}")

        # Gender
        if patient_row.get("gender_clean_FEMALE", 0) == 1:
            gender = "Female"
        elif patient_row.get("gender_clean_MALE", 0) == 1:
            gender = "Male"
        else:
            gender = "Unknown"
        st.markdown(f"**Gender:** {gender}")

        # Stage
        stage = "Unknown"
        for s in ["Stage_I", "Stage_II", "Stage_III", "Stage_IV"]:
            if patient_row.get(f"ajcc_stage_simple_{s}", 0) == 1:
                stage = s.replace("_", " ")
                break
        st.markdown(f"**AJCC stage:** {stage}")

    with col_b:
        st.markdown("#### Model Predictions")
        if len(patient_preds) > 0:
            pred_display = patient_preds[["model_name", "predicted_label", "probability_high_risk"]].copy()
            pred_display.columns = ["Model", "Prediction", "P(High Risk)"]
            pred_display["P(High Risk)"] = pred_display["P(High Risk)"].apply(
                lambda x: f"{float(x):.1%}" if pd.notna(x) else "N/A"
            )
            st.dataframe(pred_display, hide_index=True, use_container_width=True)
        else:
            st.info("No predictions available for this patient (ambiguous label — excluded from training).")

        # Feature radar
        st.markdown("#### Feature Profile")
        radar_cols = {
            "num_age": "Age",
            "num_report_word_count": "Report Length",
            "num_patch_count": "Patch Count",
            "img_pc_0": "Image PC1",
            "tfidf_pc_0": "Text PC1",
        }
        available_radar = {k: v for k, v in radar_cols.items() if k in patient_row.index and pd.notna(patient_row[k])}
        if available_radar:
            vals = [float(patient_row[k]) for k in available_radar.keys()]
            # Normalize to 0-1 for radar
            all_vals = patients_filtered[list(available_radar.keys())].dropna()
            mins = all_vals.min()
            maxs = all_vals.max()
            normed = [(float(patient_row[k]) - mins[k]) / max(maxs[k] - mins[k], 1e-9) for k in available_radar.keys()]
            labels = list(available_radar.values())

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=normed + [normed[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(0,210,255,0.15)",
                line=dict(color=COLORS["accent1"], width=2),
                name=selected_patient,
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["grid"], tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor=COLORS["grid"]),
                ),
                showlegend=False,
                height=320,
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ========================== TAB 2: Cohort Analytics ========================
with tab2:
    st.markdown("### Cohort Analytics")

    col1, col2 = st.columns(2)

    # --- Risk distribution by cancer type ---
    with col1:
        st.markdown("#### Risk Distribution by Cancer Type")
        dist = (
            risk_filtered[risk_filtered["risk_label"].isin(["high_risk", "low_risk"])]
            .groupby(["cancer_type_abbrev", "risk_label"])
            .size()
            .reset_index(name="count")
        )
        fig_dist = px.bar(
            dist,
            x="cancer_type_abbrev",
            y="count",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            barmode="group",
        )
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="", yaxis_title="Patients",
                               xaxis_tickangle=-45, legend_title="")
        st.plotly_chart(fig_dist, use_container_width=True)

    # --- Survival distribution ---
    with col2:
        st.markdown("#### Overall Survival Distribution")
        surv_data = risk_filtered[risk_filtered["risk_label"].isin(["high_risk", "low_risk"])].copy()
        surv_data["os_days"] = pd.to_numeric(surv_data["os_days"], errors="coerce")
        fig_surv = px.histogram(
            surv_data,
            x="os_days",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            nbins=50,
            opacity=0.75,
            marginal="box",
        )
        fig_surv.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_title="OS Days", yaxis_title="Count",
                               legend_title="", barmode="overlay")
        st.plotly_chart(fig_surv, use_container_width=True)

    col3, col4 = st.columns(2)

    # --- Stage breakdown ---
    with col3:
        st.markdown("#### AJCC Stage Breakdown")
        stage_cols = [c for c in features.columns if c.startswith("ajcc_stage_simple_")]
        if stage_cols:
            stage_data = features[features["patient_barcode"].isin(risk_filtered["patient_barcode"])]
            stage_counts = {}
            for c in stage_cols:
                name = c.replace("ajcc_stage_simple_", "").replace("_", " ")
                stage_counts[name] = int(stage_data[c].sum())
            stage_df = pd.DataFrame(list(stage_counts.items()), columns=["Stage", "Count"])
            stage_df = stage_df.sort_values("Count", ascending=True)
            fig_stage = px.bar(
                stage_df, x="Count", y="Stage", orientation="h",
                color_discrete_sequence=[COLORS["accent2"]],
            )
            fig_stage.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_stage, use_container_width=True)

    # --- Age distribution by risk ---
    with col4:
        st.markdown("#### Age Distribution by Risk")
        age_data = patients_filtered[patients_filtered["risk_label"].isin(["high_risk", "low_risk"])].copy()
        if "num_age" in age_data.columns:
            fig_age = px.violin(
                age_data, x="risk_label", y="num_age", color="risk_label",
                color_discrete_map=RISK_COLORS, box=True, points=False,
            )
            fig_age.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="", yaxis_title="Age (scaled)",
                                  showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)

    # --- Model comparison heatmap ---
    st.markdown("#### Model Performance Comparison")
    test_metrics = metrics[metrics["split"] == "test"].copy()
    pivot = test_metrics.pivot_table(index="model_name", columns="metric_name", values="metric_value", aggfunc="first")
    display_metrics = ["balanced_accuracy", "f1_high_risk", "f1_low_risk", "precision_high_risk", "recall_high_risk"]
    available_metrics = [m for m in display_metrics if m in pivot.columns]
    if available_metrics:
        pivot_display = pivot[available_metrics].dropna(how="all")
        pivot_display = pivot_display.astype(float)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_display.values,
            x=[m.replace("_", " ").title() for m in pivot_display.columns],
            y=pivot_display.index,
            colorscale=[[0, "#0e1117"], [0.5, "#3a7bd5"], [1, "#38ef7d"]],
            text=np.round(pivot_display.values, 3),
            texttemplate="%{text:.1%}",
            textfont=dict(size=11),
            hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{z:.1%}<extra></extra>",
            colorbar=dict(tickformat=".0%", len=0.6),
        ))
        fig_heatmap.update_layout(
            **PLOTLY_LAYOUT,
            height=max(250, len(pivot_display) * 35 + 100),
            xaxis=dict(side="top", tickangle=-30),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Embedding space ---
    st.markdown("#### Feature Space (Image + Text PCA)")
    embed_cols = ["img_pc_0", "img_pc_1", "tfidf_pc_0"]
    if all(c in patients_filtered.columns for c in embed_cols):
        embed_data = patients_filtered[
            patients_filtered["risk_label"].isin(["high_risk", "low_risk"])
        ][["patient_barcode", "risk_label", "cancer_type_abbrev"] + embed_cols].dropna()
        if len(embed_data) > 2000:
            embed_data = embed_data.sample(2000, random_state=42)
        fig_scatter = px.scatter_3d(
            embed_data,
            x="img_pc_0", y="img_pc_1", z="tfidf_pc_0",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            hover_data=["patient_barcode", "cancer_type_abbrev"],
            opacity=0.6,
        )
        fig_scatter.update_traces(marker=dict(size=2.5))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            scene=dict(
                xaxis=dict(title="Image PC1", gridcolor=COLORS["grid"], backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="Image PC2", gridcolor=COLORS["grid"], backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="Text PC1", gridcolor=COLORS["grid"], backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend_title="",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ========================== TAB 3: Pipeline ================================
with tab3:
    st.markdown("### Pipeline Architecture")
    st.caption("Medallion architecture on Databricks — Bronze / Silver / Gold")

    # Pipeline flow
    pipe_cols = st.columns(5)
    steps = [
        ("Raw Sources", "3 modalities: images (HuggingFace), reports (Mendeley), clinical (GDC)"),
        ("Bronze", "Raw Delta tables, untouched ingestion"),
        ("Silver", "Cleaned, joined on patient barcode, standardized"),
        ("Gold", "Feature-engineered, risk labels, model-ready"),
        ("Model", "Late fusion: LR + LightGBM + MLP"),
    ]
    for i, (title, desc) in enumerate(steps):
        with pipe_cols[i]:
            st.markdown(f"""
            <div class="pipe-step">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("#### Feature Composition")
        feat_groups = feat_meta.groupby("feature_type").size().reset_index(name="count")
        fig_feat = px.pie(
            feat_groups, values="count", names="feature_type",
            color_discrete_sequence=[COLORS["accent1"], COLORS["accent2"], COLORS["accent3"],
                                     COLORS["high_risk"], COLORS["low_risk"]],
            hole=0.55,
        )
        fig_feat.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        fig_feat.update_traces(textinfo="label+value", textfont_size=11)
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_p2:
        st.markdown("#### Data Quality by Cancer Type")
        if "metric_name" in dq.columns and "cancer_type" in dq.columns:
            null_dq = dq[dq["metric_name"].str.contains("null", case=False, na=False)].copy()
            if len(null_dq) > 0:
                null_dq["metric_value"] = pd.to_numeric(null_dq["metric_value"], errors="coerce")
                null_dq = null_dq.sort_values("metric_value", ascending=False).head(20)
                fig_dq = px.bar(
                    null_dq, x="metric_value", y="cancer_type", color="metric_name",
                    orientation="h",
                    color_discrete_sequence=[COLORS["accent1"], COLORS["accent2"], COLORS["accent3"]],
                )
                fig_dq.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="Null Rate",
                                     yaxis_title="", legend_title="")
                st.plotly_chart(fig_dq, use_container_width=True)
            else:
                # Show join coverage instead
                join_dq = dq[dq["metric_name"].str.contains("join|coverage|count", case=False, na=False)].copy()
                if len(join_dq) > 0:
                    join_dq["metric_value"] = pd.to_numeric(join_dq["metric_value"], errors="coerce")
                    fig_dq = px.bar(
                        join_dq.head(20), x="metric_value", y="cancer_type", color="metric_name",
                        orientation="h",
                        color_discrete_sequence=[COLORS["accent1"], COLORS["accent2"]],
                    )
                    fig_dq.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="Value",
                                         yaxis_title="", legend_title="")
                    st.plotly_chart(fig_dq, use_container_width=True)
                else:
                    st.dataframe(dq.head(20), use_container_width=True)
        else:
            st.dataframe(dq.head(20), use_container_width=True)

    # Modality breakdown
    st.markdown("#### Multimodal Fusion Pipeline")
    st.markdown("""
    <div style="display:flex; gap:1rem; flex-wrap:wrap;">
        <div class="pipe-step" style="flex:1; min-width:200px; border-left: 3px solid #00d2ff;">
            <h4>Histopathology Images</h4>
            <p>ResNet-18 patch embeddings → Gated Attention MIL → 512-dim pooled vector → PCA to 50 dims</p>
        </div>
        <div class="pipe-step" style="flex:1; min-width:200px; border-left: 3px solid #a855f7;">
            <h4>Pathology Reports</h4>
            <p>Free-text OCR reports → TF-IDF (200 features) → PCA to 30 dims</p>
        </div>
        <div class="pipe-step" style="flex:1; min-width:200px; border-left: 3px solid #38ef7d;">
            <h4>Clinical Records</h4>
            <p>Age, gender, AJCC stage → one-hot encoding + standard scaling → 21 features</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
