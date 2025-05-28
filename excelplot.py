# Refactored Streamlit Analysis App with Advanced Reporting
# - Multiple sequential transforms including Remove T0 and Uniform Asymptotes
# - Interactive Plotly plots with lasso selection and manual table edits
# - Dynamic report tables with Add buttons for final data & fit curves

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit, root_scalar

# Optional import for selection events
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None
    st.sidebar.warning("Install 'streamlit-plotly-events' for point removal: pip install streamlit-plotly-events")

# -----------------------------
# Global storage
# -----------------------------
asymptotes = {}

# -----------------------------
# 1. Utility Cache
# -----------------------------
@st.cache_data
def load_excel_sheets(file) -> dict:
    return pd.read_excel(file, sheet_name=None)

# -----------------------------
# 2. Transform Definitions
# -----------------------------
# Placeholder for uniform transforms; helper below
TRANSFORMS = {
    "None": lambda y: y,
    "Remove T0": None,
    "Fix initial baseline": None,  # defined below
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.iloc[0] or 1),
    "Min-Max normalization (0–1)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1),
    "Scale to [18-55]": lambda y: ((y - y.min()) / ((y.max() - y.min()) or 1)) * (55 - 18) + 18,
    "Cumulative Max (Monotonic Rise)": lambda y: y.cummax(),
    "Uniform Asymptotes (0–1)": None,
    "Uniform Asymptotes (18–55)": None
}

# -----------------------------
# 3. Helper Functions
# -----------------------------
def fix_initial_baseline(y: pd.Series) -> pd.Series:
    """If first y > second y, set first to second"""
    y = y.copy()
    if len(y) > 1 and y.iloc[0] > y.iloc[1]:
        y.iloc[0] = y.iloc[1]
    return y

def uniform_asymptote(series: pd.Series, sample: str, out_range=(0,1)) -> pd.Series:
    A, D = asymptotes.get(sample, (series.max(), series.min()))
    y0 = (series - D) / ((A - D) or 1)
    return y0 * (out_range[1] - out_range[0]) + out_range[0]

# assign placeholder transforms now that helpers exist
def _none_arg(y): return y
TRANSFORMS["Remove T0"] = lambda y: y  # handled separately
TRANSFORMS["Fix initial baseline"] = fix_initial_baseline
TRANSFORMS["Uniform Asymptotes (0–1)"] = lambda y, sample=None: uniform_asymptote(y, sample, (0,1))
TRANSFORMS["Uniform Asymptotes (18–55)"] = lambda y, sample=None: uniform_asymptote(y, sample, (18,55))

# -----------------------------
# 4. Apply sequential transforms
# -----------------------------
def apply_transforms(df_group: pd.DataFrame, transforms: list, y_col: str, sample_col: str) -> pd.Series:
    y = df_group[y_col].copy()
    sample = df_group[sample_col].iloc[0]
    for t in transforms:
        func = TRANSFORMS.get(t, _none_arg)
        try:
            y = func(y, sample)
        except TypeError:
            y = func(y)
    return y

# -----------------------------
# 5. Model Definitions & Fitting
# -----------------------------
MODELS = {
    "Linear":  (lambda x, a, b: a * x + b,      [1, 0]),
    "Sigmoid": (lambda x, a, b: 1/(1+np.exp(-(x-a)/b)), [np.median, 1]),
    "4PL":     (lambda x, A, B, C, D: D + (A-D)/(1+(x/C)**B), [1,1,1,0]),
    "5PL":     (lambda x, A, B, C, D, G: D + (A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz":(lambda x, a, b, c: a * np.exp(-b * np.exp(-c*x)), [1,1,1])
}

def fit_and_evaluate(x, y, model_name, threshold):
    func, _ = MODELS[model_name]
    popt, pcov = curve_fit(func, x, y, maxfev=10000)
    y_fit = func(x, *popt)
    rss = ((y - y_fit)**2).sum(); tss = ((y - y.mean())**2).sum()
    r2 = 1 - rss/tss if tss else np.nan
    rmse = np.sqrt(rss/len(y))
    tt, tt_se = ("N/A","N/A")
    if y_fit.min() <= threshold <= y_fit.max():
        root = root_scalar(lambda t: func(t,*popt)-threshold, bracket=[x.min(), x.max()])
        if root.converged:
            tt = root.root
            d = (func(tt+1e-5,*popt)-func(tt-1e-5,*popt))/(2e-5)
            var = (d**-2) * pcov.sum() if pcov.size else 0
            tt_se = np.sqrt(var) if var>0 else np.nan
    return popt, pcov, y_fit, r2, rmse, tt, tt_se

# -----------------------------
# 6. Interactive Plot
# -----------------------------
def plot_interactive(current_df, original_df, x_col, y_col, sample_col, transforms, threshold):
    fig = go.Figure(); fig.update_layout(title='Interactive Data Plot', legend={'itemclick':'toggle'}, dragmode='lasso')
    remove_t0 = "Remove T0" in transforms
    real_transforms = [t for t in transforms if t != "Remove T0"]
    for sample in original_df[sample_col].dropna().unique():
        grp0 = original_df[original_df[sample_col]==sample].sort_values(x_col)
        fig.add_trace(go.Scatter(x=grp0[x_col], y=grp0[y_col], mode='markers', marker=dict(color='lightgrey'), showlegend=False))
    for sample in current_df[sample_col].unique():
        grp = current_df[current_df[sample_col]==sample].sort_values(x_col)
        if remove_t0 and len(grp)>1:
            grp = grp.iloc[1:]
        y_trans = apply_transforms(grp, real_transforms, y_col, sample_col)
        fig.add_trace(go.Scatter(x=grp[x_col], y=grp[y_col], mode='markers', name=f"{sample} raw", customdata=np.stack([grp.index, np.repeat(sample,len(grp))],axis=1)))
        fig.add_trace(go.Scatter(x=grp[x_col], y=y_trans, mode='lines', name=f"{sample} trans", line=dict(dash='dash')))
    fig.add_hline(y=threshold, line_dash='dot', annotation_text='Threshold')
    selected = []
    if plotly_events:
        selected = plotly_events(fig, select_event=True, click_event=False)
    st.plotly_chart(fig, use_container_width=True)
    if not plotly_events:
        st.info("Install 'streamlit-plotly-events' to enable point selection.")
    return selected

# -----------------------------
# 7. Main App
# -----------------------------
def main():
    st.sidebar.title("Upload & Settings")
    uploaded = st.sidebar.file_uploader("Excel file", type=["xls","xlsx"])
    if not uploaded:
        st.info("Please upload an Excel file.")
        return

    sheets = load_excel_sheets(uploaded)
    sheet = st.sidebar.selectbox("Sheet", list(sheets.keys()))
    df_orig = sheets[sheet]
    if 'df_int' not in st.session_state or st.session_state.df_int is None:
        st.session_state.df_int = df_orig.copy()
    df = st.session_state.df_int.copy()

    # Initialize report tables container
    if 'report_tables' not in st.session_state:
        st.session_state.report_tables = []

    st.dataframe(df.head())
    sample_col = st.sidebar.selectbox("Sample ID col", df.columns)
    sel = st.sidebar.multiselect("Samples", df[sample_col].dropna().unique(), default=list(df[sample_col].dropna().unique()))
    df = df[df[sample_col].isin(sel)]
    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols)<2:
        st.warning("Need >=2 numeric cols.")
        return
    x_col = st.sidebar.selectbox("X axis", num_cols)
    y_col = st.sidebar.selectbox("Y axis", num_cols)
    transform_opts = list(TRANSFORMS.keys())
    transforms = st.sidebar.multiselect("Transform sequence", transform_opts, default=["None"])
    threshold = st.sidebar.number_input("Threshold", value=1.0)

    # Global Model Override
    st.sidebar.markdown("### Global Model Fit")
    global_model = st.sidebar.selectbox("Fit all samples with", list(MODELS.keys()))
    fit_all = st.sidebar.checkbox("Use global model for all samples", value=False)

    # -----------------------------
    # Global Model Override
    # -----------------------------
    st.sidebar.markdown("### Global Model Fit")
    global_model = st.sidebar.selectbox("Fit all samples with", list(MODELS.keys()))
    fit_all = st.sidebar.checkbox("Use global model for all samples", value=False)


    # Interactive plot
    sel_pts = plot_interactive(df, df_orig, x_col, y_col, sample_col, transforms, threshold)
    if sel_pts and st.button("Remove Selected Points"):
        drop_idx = [pt['customdata'][0] for pt in sel_pts]
        st.session_state.df_int = df_orig.drop(index=drop_idx)
        st.experimental_rerun()

    # Editable table
    st.subheader("Data Table (Editable)")
    if hasattr(st, 'data_editor'):
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.warning("Streamlit too old for table editing.")
        edited_df = df.copy()
    if st.button("Apply Table Edits"):
        st.session_state.df_int = edited_df.copy()
        st.experimental_rerun()
    df = edited_df.copy()

    # Fitting & collecting final data
    st.header("Model Fitting Results")
    params_list, tt_list, fitted_curves = [], [], []
    fig_fit = go.Figure(); fig_fit.update_layout(title='Model Fitting Plot')
    x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    remove_t0 = "Remove T0" in transforms
    real_transforms = [t for t in transforms if t!="Remove T0"]
    for sample in df[sample_col].unique():
        # choose model per-sample or global override
        if fit_all:
            model = global_model
        else:
            # model selection handled above
