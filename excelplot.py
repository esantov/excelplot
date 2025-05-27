# Refactored Streamlit Analysis App with Interactive Plotting
# - Plotly for interactive removal of datasets (via legend click) and points (via lasso selection)
# - Background plotting of original (unmodified) data in light grey
# - Modular functions with caching
# - Streamlit session state to persist data edits

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from scipy.optimize import curve_fit, root_scalar

# -----------------------------
# 1. Utility & Cache
# -----------------------------
@st.cache_data
def load_excel_sheets(file) -> dict:
    """Load Excel sheets into a dict of DataFrames."""
    return pd.read_excel(file, sheet_name=None)

# Initialize session state for edited data
if "df_interactive" not in st.session_state:
    st.session_state.df_interactive = None

# -----------------------------
# 2. Transform Definitions
# -----------------------------
TRANSFORMS = {
    "None": lambda y: y,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.max() or 1),
    "Min-Max normalization (0–1, sample-wise)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1)
}

# -----------------------------
# 3. Model Definitions
# -----------------------------
MODELS = {
    "Linear":  (lambda x, a, b: a * x + b,      [1, 0]),
    "Sigmoid": (lambda x, a, b: 1/(1+np.exp(-(x-a)/b)), [np.median, 1]),
    "4PL":     (lambda x, A, B, C, D: D + (A-D)/(1+(x/C)**B), [1,1,1,0]),
    "5PL":     (lambda x, A, B, C, D, G: D + (A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz":(lambda x, a, b, c: a * np.exp(-b * np.exp(-c*x)), [1,1,1])
}

# -----------------------------
# 4. Data Transformation
# -----------------------------
def transform_df(df: pd.DataFrame, y_col: str, transform: str) -> pd.Series:
    func = TRANSFORMS.get(transform, TRANSFORMS["None"])
    return func(df[y_col])

# -----------------------------
# 5. Model Fitting
# -----------------------------
def fit_and_evaluate(x, y, model_name, threshold):
    model_func, _ = MODELS[model_name]
    popt, pcov = curve_fit(model_func, x, y, maxfev=10000)
    fit_vals = model_func(x, *popt)
    rss = ((y - fit_vals)**2).sum(); tss = ((y - y.mean())**2).sum()
    r2 = 1 - rss/tss if tss > 0 else np.nan
    rmse = np.sqrt(rss/len(y))
    tt, tt_se = ("N/A", "N/A")
    if min(fit_vals) <= threshold <= max(fit_vals):
        root = root_scalar(lambda t: model_func(t, *popt) - threshold, bracket=[x.min(), x.max()])
        if root.converged:
            tt = root.root
            d = (model_func(tt+1e-5,*popt)-model_func(tt-1e-5,*popt))/(2e-5)
            var = (d**-2) * pcov.sum() if pcov.size else 0
            tt_se = np.sqrt(var) if var>0 else np.nan
    return popt, fit_vals, r2, rmse, tt, tt_se

# -----------------------------
# 6. Interactive Plot Helpers
# -----------------------------
def plot_interactive(current_df, original_df, x_col, y_col, sample_col, transform, threshold):
    """Interactive Plotly figure with:
       - Original data in light grey
       - Current (possibly edited) raw data markers
       - Transformed lines
       - Threshold line
    """
    fig = go.Figure()
    # Plot original data background
    for sample in original_df[sample_col].dropna().unique():
        grp_orig = original_df[original_df[sample_col]==sample].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=grp_orig[x_col], y=grp_orig[y_col], mode='markers',
            name=f"{sample} original", marker={'color':'lightgrey'}, showlegend=False
        ))
    # Plot current data and transformed
    for sample in current_df[sample_col].unique():
        grp = current_df[current_df[sample_col]==sample].sort_values(x_col)
        y_trans = transform_df(grp, y_col, transform)
        fig.add_trace(go.Scatter(
            x=grp[x_col], y=grp[y_col], mode='markers',
            name=f"{sample} raw", customdata=np.stack([grp.index, np.repeat(sample, len(grp))], axis=1)
        ))
        fig.add_trace(go.Scatter(
            x=grp[x_col], y=y_trans, mode='lines', name=f"{sample} trans",
            line={'dash':'dash'}
        ))
    # Threshold
    fig.add_hline(y=threshold, line_dash='dot', annotation_text='Threshold', annotation_position='top left')
    fig.update_layout(
        title='Interactive Data Plot',
        legend={'itemclick':'toggle'},
        dragmode='lasso'
    )
    selected = plotly_events(fig, select_event=True, click_event=False)
    st.plotly_chart(fig, use_container_width=True)
    return selected

# -----------------------------
# 7. Main Application
# -----------------------------
def main():
    st.sidebar.title("Data & Settings")
    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xls","xlsx"])
    if not uploaded:
        st.info("Please upload an Excel file to begin.")
        return

    sheets = load_excel_sheets(uploaded)
    sheet = st.sidebar.selectbox("Select Sheet", list(sheets.keys()))
    df_raw = sheets[sheet]
    if st.session_state.df_interactive is None:
        st.session_state.df_interactive = df_raw.copy()
    df_current = st.session_state.df_interactive

    st.dataframe(df_current.head())
    sample_col = st.sidebar.selectbox("Sample Column", df_current.columns)
    samples = df_current[sample_col].dropna().unique()
    sel_samples = st.sidebar.multiselect("Filter Samples", samples, default=list(samples))
    df_current = df_current[df_current[sample_col].isin(sel_samples)]

    num_cols = df_current.select_dtypes('number').columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least two numeric columns to plot.")
        return

    x_col = st.sidebar.selectbox("X Column", num_cols)
    y_col = st.sidebar.selectbox("Y Column", num_cols)
    transform = st.sidebar.selectbox("Transformation", list(TRANSFORMS.keys()))
    threshold = st.sidebar.number_input("Threshold Value", value=1.0)

    # Interactive Plot with original and current
    selected = plot_interactive(df_current, df_raw, x_col, y_col, sample_col, transform, threshold)
    if selected:
        st.write(f"Selected {len(selected)} points.")
        if st.button("Remove Selected Points"):
            to_drop = [pt['customdata'][0] for pt in selected]
            st.session_state.df_interactive = df_raw.drop(index=to_drop)
            st.experimental_rerun()

    # ... Fitting section can be similarly updated for interactive behavior

if __name__=='__main__':
    main()
