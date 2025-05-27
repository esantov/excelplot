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
# Global storage of fitted asymptotes
# -----------------------------
asymptotes = {}

# -----------------------------
# 1. Utility & Cache
# -----------------------------
@st.cache_data
def load_excel_sheets(file) -> dict:
    return pd.read_excel(file, sheet_name=None)

# -----------------------------
# 2. Transform Definitions
# -----------------------------
# Helper: if the second point exceeds the first, set the first equal to the second
def fix_initial_baseline(y: pd.Series) -> pd.Series:
    """If the first point exceeds the second, set the first equal to the second."""
    if len(y) > 1 and y.iloc[0] > y.iloc[1]:
        y.iloc[0] = y.iloc[1]
    return y

TRANSFORMS = {
    # Custom initial baseline correction (first two points only)
    "Fix initial baseline": fix_initial_baseline,
    "None": lambda y: y,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.iloc[0] or 1),  # divide by initial value per sample
    "Min-Max normalization (0–1, sample-wise)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1),
    "Scale to [18-55]": lambda y: ((y - y.min()) / ((y.max() - y.min()) or 1)) * (55 - 18) + 18,
    "Cumulative Max (Monotonic Rise)": lambda y: y.cummax(),
    "Uniform Asymptotes (0–1)": None,     # placeholder
    "Uniform Asymptotes (18–55)": None    # placeholder
}
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

    st.dataframe(df.head())
    sample_col = st.sidebar.selectbox("Sample ID col", df.columns)
    samples = df[sample_col].dropna().unique()
    sel = st.sidebar.multiselect("Samples", samples, default=list(samples))
    df = df[df[sample_col].isin(sel)]

    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need >=2 numeric cols.")
        return
    x_col = st.sidebar.selectbox("X axis", num_cols)
    y_col = st.sidebar.selectbox("Y axis", num_cols)

    transform_opts = list(TRANSFORMS.keys())
    transforms = st.sidebar.multiselect("Transform sequence", transform_opts, default=["None"])
    threshold = st.sidebar.number_input("Threshold", value=1.0)

    sel_pts = plot_interactive(df, df_orig, x_col, y_col, sample_col, transforms, threshold)
    if sel_pts and st.button("Remove Selected Points"):
        drop_idx = [pt['customdata'][0] for pt in sel_pts]
        st.session_state.df_int = df_orig.drop(index=drop_idx)
        st.experimental_rerun()

    st.subheader("Data Table (Editable)")
    if hasattr(st, 'data_editor'):
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.warning("Your Streamlit version does not support table editing. Update to v1.18+.")
        edited_df = df.copy()
    if st.button("Apply Table Edits"):
        st.session_state.df_int = edited_df.copy()
        st.experimental_rerun()
    df = edited_df.copy()

    # fitting
    params_list, tt_list = [], []
    fig_fit = go.Figure(); fig_fit.update_layout(title='Model Fitting Plot')
    x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    for sample in df[sample_col].unique():
        grp = df[df[sample_col]==sample].sort_values(x_col)
        x_vals = grp[x_col].values
        y_vals = apply_transforms(grp, transforms, y_col, sample_col).values
        model = st.sidebar.selectbox(f"Model for {sample}", list(MODELS.keys()))
        try:
            popt, y_fit, r2, rmse, tt, tt_se = fit_and_evaluate(x_vals, y_vals, model, threshold)
            # record asymptotes
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

    # report export
    if st.button("Export Report to Excel"):
        report_buf = io.BytesIO()
        with pd.ExcelWriter(report_buf, engine='xlsxwriter') as writer:
            df_orig.to_excel(writer, sheet_name='Original Data', index=False)
            st.session_state.df_int.to_excel(writer, sheet_name='Edited Data', index=False)
            df.to_excel(writer, sheet_name='Processed Data', index=False)
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
