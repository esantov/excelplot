# Refactored Streamlit Analysis App with Sequential Transformations, Interactive Plotting, and Model Fitting
# - Supports multiple sequential transformations
# - Plotly for interactive removal of datasets and points
# - Background original data in light grey
# - Modular functions, caching, and session state
# - Graceful fallback if streamlit-plotly-events missing

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
    st.sidebar.warning("Install 'streamlit-plotly-events' to enable point selection: pip install streamlit-plotly-events")

# -----------------------------
# 1. Utility & Cache
# -----------------------------
@st.cache_data
def load_excel_sheets(file) -> dict:
    return pd.read_excel(file, sheet_name=None)

# Initialize session state for edited data
if 'df_int' not in st.session_state:
    st.session_state.df_int = None

# -----------------------------
# 2. Transform Definitions
# -----------------------------
TRANSFORMS = {
    "None": lambda y: y,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.iloc[0] or 1),  # divide by initial value per sample
    "Min-Max normalization (0–1, sample-wise)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1),
    # Custom scaling to a fixed range [18,55] per sample group: each sample's min→18 and max→55
    "Scale to [18-55]": lambda y: ((y - y.min()) / ((y.max() - y.min()) or 1)) * (55 - 18) + 18
}

# -----------------------------
# 2. Transform Definitions
# -----------------------------
TRANSFORMS = {
    "None": lambda y: y,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.iloc[0] or 1),
    "Min-Max normalization (0–1, sample-wise)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1),
    "Scale to [18-55]": lambda y: ((y - y.min()) / ((y.max() - y.min()) or 1)) * (55 - 18) + 18,
    # Placeholder for uniform asymptote; actual function applied in apply_transforms
    "Uniform Asymptotes (0–1)": None,
    "Uniform Asymptotes (19–50)": None
}

# -----------------------------
# 3. Helper: uniform asymptotes per sample
# -----------------------------
def uniform_asymptote(series: pd.Series, sample: str, out_range=(0,1)) -> pd.Series:
    """Rescale series so its fitted asymptotes D->min(out_range), A->max(out_range)."""
    # get (A,D) from asymptotes dict
    A, D = asymptotes.get(sample, (series.max(), series.min()))
    # normalize to 0–1
    y0 = (series - D) / ((A - D) or 1)
    # scale to out_range
    return y0 * (out_range[1] - out_range[0]) + out_range[0]

# -----------------------------
# 4. Apply one or more transforms in sequence
# -----------------------------
def apply_transforms(df_group: pd.DataFrame, transforms: list, y_col: str, sample_col: str) -> pd.Series:
    y = df_group[y_col].copy()
    sample = df_group[sample_col].iloc[0]
    for t in transforms:
        if t == "Uniform Asymptotes (0–1)":
            y = uniform_asymptote(y, sample, (0,1))
        elif t == "Uniform Asymptotes (19–50)":
            y = uniform_asymptote(y, sample, (19,50))
        else:
            func = TRANSFORMS.get(t, TRANSFORMS["None"])
            y = func(y)
    return y

# -----------------------------
# 5. Model Definitions
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
# 4. Model Fitting
# -----------------------------
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
    return popt, y_fit, r2, rmse, tt, tt_se

# -----------------------------
# 5. Interactive Plot Helper
# -----------------------------
def plot_interactive(current_df, original_df, x_col, y_col, sample_col, transforms, threshold):
    fig = go.Figure()
    fig.update_layout(title='Interactive Data Plot', legend={'itemclick':'toggle'}, dragmode='lasso')
    # original backdrop
    for sample in original_df[sample_col].dropna().unique():
        grp0 = original_df[original_df[sample_col]==sample].sort_values(x_col)
        fig.add_trace(go.Scatter(x=grp0[x_col], y=grp0[y_col], mode='markers',
                                 marker=dict(color='lightgrey'), showlegend=False))
    # current raw + transformed
    for sample in current_df[sample_col].unique():
        grp = current_df[current_df[sample_col]==sample].sort_values(x_col)
        y_trans = apply_transforms(grp, transforms, y_col, sample_col)
        fig.add_trace(go.Scatter(x=grp[x_col], y=grp[y_col], mode='markers',
                                 name=f"{sample} raw", customdata=np.stack([grp.index, np.repeat(sample,len(grp))],axis=1)))
        fig.add_trace(go.Scatter(x=grp[x_col], y=y_trans, mode='lines',
                                 name=f"{sample} trans", line=dict(dash='dash')))
    fig.add_hline(y=threshold, line_dash='dot', annotation_text='Threshold')
    fig.update_layout(title='Interactive Plot', legend={'itemclick':'toggle'}, dragmode='lasso')
    selected=[]
    if plotly_events:
        selected = plotly_events(fig, select_event=True, click_event=False)
    st.plotly_chart(fig, use_container_width=True)
    if not plotly_events:
        st.info("Install 'streamlit-plotly-events' to enable point selection.")
    return selected

# -----------------------------
# 6. Main App
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
    if st.session_state.df_int is None:
        st.session_state.df_int = df_orig.copy()
    df = st.session_state.df_int
    st.dataframe(df.head())

    # selectors
    sample_col = st.sidebar.selectbox("Sample ID col", df.columns)
    samples = df[sample_col].dropna().unique()
    sel = st.sidebar.multiselect("Samples", samples, default=list(samples))
    df = df[df[sample_col].isin(sel)]

    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols)<2:
        st.warning("Need >=2 numeric cols.")
        return
    x_col = st.sidebar.selectbox("X axis", num_cols)
    y_col = st.sidebar.selectbox("Y axis", num_cols)

    # allow multiple transforms
    transform_opts = list(TRANSFORMS.keys())
    transforms = st.sidebar.multiselect("Transform sequence", transform_opts, default=["None"])

    threshold = st.sidebar.number_input("Threshold", value=1.0)

        # interactive plot
    sel_pts = plot_interactive(df, df_orig, x_col, y_col, sample_col, transforms, threshold)
    if sel_pts and st.button("Remove Selected Points"):
        drop_idx = [pt['customdata'][0] for pt in sel_pts]
        st.session_state.df_int = df_orig.drop(index=drop_idx)
        st.experimental_rerun()

    # -----------------------------
    # Manual Table Editing (with version check)
    # -----------------------------
    st.subheader("Data Table (Editable)")
    # Use st.data_editor if available, else fallback to st.experimental_data_editor
    if hasattr(st, 'data_editor'):
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.warning("Your Streamlit version does not support table editing. Update to v1.18+ for data_editor.")
        edited_df = df.copy()
    if st.button("Apply Table Edits"):
        # User may remove or modify rows; update session state
        st.session_state.df_int = edited_df.copy()
        st.experimental_rerun()

    # After edits, override df for subsequent calculations
    df = edited_df.copy()

    # fitting
    # Prepare to collect asymptotes for each sample
    asymptotes = {}
    params_list, tt_list = [], []
    fig_fit = go.Figure()
    fig_fit.update_layout(title='Model Fitting Plot')
    x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    for sample in df[sample_col].unique():
        grp = df[df[sample_col]==sample].sort_values(x_col)
        x_vals = grp[x_col].values
        y_vals = apply_transforms(grp, transforms, y_col, sample_col).values
        model = st.sidebar.selectbox(f"Model for {sample}", list(MODELS.keys()))
        try:
            popt, y_fit, r2, rmse, tt, tt_se = fit_and_evaluate(x_vals, y_vals, model, threshold)
            # record asymptotes A (upper) and D (lower)
            if model in ("4PL","5PL"):
                A, D = popt[0], popt[3]
            else:
                D, A = y_fit.min(), y_fit.max()
            asymptotes[sample] = (A, D)
            fig_fit.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name=f"{sample} data"))
            fig_fit.add_trace(go.Scatter(x=x_lin, y=MODELS[model][0](x_lin,*popt), mode='lines', name=f"{sample} fit"))
            params_list.append({"Sample":sample, "Model":model, "R2":round(r2,4), "RMSE":round(rmse,4)})
            tt_list.append((sample, tt, tt_se))
        except Exception as e:
            st.warning(f"Fit error {sample}: {e}")
    fig_fit.add_hline(y=threshold, line_dash='dash')
    st.plotly_chart(fig_fit, use_container_width=True)

    st.subheader("Fit Parameters")
    st.dataframe(pd.DataFrame(params_list))
    st.subheader("Time to Threshold")
    st.dataframe(pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]))
    
    # -----------------------------
    # 7. Report Export
    # -----------------------------
    if st.button("Export Report to Excel"):
        report_buf = io.BytesIO()
        with pd.ExcelWriter(report_buf, engine='xlsxwriter') as writer:
            # Original data
            df_orig.to_excel(writer, sheet_name='Original Data', index=False)
            # Edited data (full sheet after manual edits)
            st.session_state.df_int.to_excel(writer, sheet_name='Edited Data', index=False)
            # Processed data (filtered & transformed)
            df.to_excel(writer, sheet_name='Processed Data', index=False)
            # Fit results
            pd.DataFrame(params_list).to_excel(writer, sheet_name='Fit Parameters', index=False)
            pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]).to_excel(writer, sheet_name='Time to Threshold', index=False)
        report_buf.seek(0)
        st.download_button(
            label="Download Report",
            data=report_buf,
            file_name="Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__=='__main__':
    main()
