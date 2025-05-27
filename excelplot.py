# Refactored Streamlit Analysis App
# - DRY transformations and fitting logic
# - Modular functions
# - Caching for repeated operations
# - Clear separation of UI and compute layers

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar

# -----------------------------
# 1. Utility Functions
# -----------------------------

def load_excel(file) -> pd.DataFrame:
    """Load an Excel sheet as DataFrame, caching repeated loads."""
    return pd.read_excel(file, sheet_name=None)

@st.cache_data
def get_sheet_df(xls: dict, sheet: str) -> pd.DataFrame:
    return xls[sheet]

TRANSFORMS = {
    "None": lambda y: y,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I₀ normalization": lambda y: y / (y.max() or 1),
    "Min-Max normalization (0–1, sample-wise)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1)
}

# Model definitions
MODELS = {
    "Linear":      (lambda x, a, b: a * x + b,        [1, 0]),
    "Sigmoid":     (lambda x, a, b: 1/(1+np.exp(-(x-a)/b)), [np.median, 1]),
    "4PL":         (lambda x, A, B, C, D: D + (A-D)/(1+(x/C)**B), [1, 1, 1, 0]),
    "5PL":         (lambda x, A, B, C, D, G: D + (A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz":    (lambda x, a, b, c: a * np.exp(-b * np.exp(-c*x)), [1,1,1])
}

# -----------------------------
# 2. Data Transformation
# -----------------------------

def transform_df(df: pd.DataFrame, x_col: str, y_col: str, transform: str) -> pd.DataFrame:
    picker = TRANSFORMS.get(transform, TRANSFORMS["None"])
    return df.assign(**{y_col: picker(df[y_col])})

# -----------------------------
# 3. Model Fitting
# -----------------------------

def fit_and_evaluate(x: np.ndarray, y: np.ndarray, model_name: str, threshold: float):
    func, _ = MODELS[model_name]
    popt, pcov = curve_fit(func, x, y, maxfev=10000)
    fit_vals = func(x, *popt)
    # metrics
    rss = ((y - fit_vals)**2).sum()
    tss = ((y - y.mean())**2).sum()
    r2  = 1 - rss/tss if tss > 0 else np.nan
    rmse = np.sqrt(rss/len(y))
    # time to threshold
    tt, tt_se = ("N/A","N/A")
    if min(fit_vals) <= threshold <= max(fit_vals):
        root = root_scalar(lambda t: func(t, *popt) - threshold,
                           bracket=[x.min(), x.max()])
        if root.converged:
            tt = root.root
            deriv = (func(tt+1e-5,*popt)-func(tt-1e-5,*popt))/(2e-5)
            var = ((deriv**-2) * pcov.sum()) if pcov.size else 0
            tt_se = np.sqrt(var) if var>0 else np.nan
    return popt, pcov, fit_vals, r2, rmse, tt, tt_se

# -----------------------------
# 4. Reporting
# -----------------------------

def add_to_report(name: str, content, is_plot: bool=False):
    if is_plot:
        buf = io.BytesIO()
        content.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.session_state.report_plots.append((name, buf.read()))
    else:
        st.session_state.report_tables.append((name, content.copy()))
    st.session_state.report_elements[name] = True

# -----------------------------
# 5. Main App Layout
# -----------------------------

def main():
    st.sidebar.title("Data & Settings")
    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xls","xlsx"])

    if uploaded:
        xls = load_excel(uploaded)
        sheet = st.sidebar.selectbox("Sheet", list(xls.keys()))
        df_raw = get_sheet_df(xls, sheet)

        st.dataframe(df_raw.head())
        # sample selector
        sample_col = st.sidebar.selectbox("Sample Column", df_raw.columns)
        all_samples = df_raw[sample_col].dropna().unique()
        sel_samples = st.sidebar.multiselect("Samples", all_samples, default=list(all_samples))
        df = df_raw[df_raw[sample_col].isin(sel_samples)] if sel_samples else df_raw

        num_cols = df.select_dtypes('number').columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns")
            return

        x_col = st.sidebar.selectbox("X Column", num_cols)
        y_col = st.sidebar.selectbox("Y Column", num_cols)

        transform = st.sidebar.selectbox("Transform", list(TRANSFORMS.keys()))
        threshold = st.sidebar.number_input("Threshold", value=1.0)
        model_default = st.sidebar.selectbox("Default Model", list(MODELS.keys()))

        st.header("Transformation Preview")
        tab1, tab2 = st.tabs(["Raw","Transformed"])
        for tab, use_transform in [(tab1, "None"),(tab2, transform)]:
            with tab:
                fig, ax = plt.subplots()
                for s in sel_samples:
                    grp = df[df[sample_col]==s].sort_values(x_col)
                    data = transform_df(grp, x_col, y_col, use_transform)
                    ax.plot(data[x_col], data[y_col], label=s)
                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                ax.legend(); st.pyplot(fig)

        st.header("Fitting & Results")
        all_params, all_tt = [], []
        fig_all, ax_all = plt.subplots(figsize=(8,5))
        for s in sel_samples:
            grp = df[df[sample_col]==s].sort_values(x_col)
            data = transform_df(grp, x_col, y_col, transform)
            x, y = data[x_col].values, data[y_col].values
            model = st.sidebar.selectbox(f"Model for {s}", list(MODELS.keys()), index=list(MODELS).index(model_default))

            try:
                popt, pcov, fit_vals, r2, rmse, tt, tt_se = fit_and_evaluate(x,y,model,threshold)
                ax_all.plot(x, y, 'o', label=f"{s} data")
                ax_all.plot(x, fit_vals, '-', label=f"{s} fit")
                all_params.append({
                    "Sample": s,
                    "Model": model,
                    "R2": round(r2,4),
                    "RMSE": round(rmse,4)
                })
                all_tt.append((s, tt, tt_se))
            except Exception as e:
                st.warning(f"Fit failed for {s}: {e}")

        ax_all.axhline(threshold, linestyle='--')
        ax_all.set_xlabel(x_col); ax_all.set_ylabel(y_col)
        ax_all.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        st.pyplot(fig_all)

        # Display tables
        param_df = pd.DataFrame(all_params)
        tt_df = pd.DataFrame(all_tt, columns=["Sample","TT","TT_SE"])
        st.subheader("Parameters"); st.dataframe(param_df)
        st.subheader("Time-to-Threshold"); st.dataframe(tt_df)

        # Report session state init
        for key in ["report_plots","report_tables","report_elements"]:
            st.session_state.setdefault(key, [])

        # Report buttons
        if st.button("Add Summary Plot to Report"):
            add_to_report("Summary Fit", fig_all, is_plot=True)
        if st.button("Add Params to Report"):
            add_to_report("Params Table", param_df)
        if st.button("Add TT to Report"):
            add_to_report("TT Table", tt_df)

        # Export
        if st.button("Export Report Excel"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                for name,tbl in st.session_state.report_tables:
                    tbl.to_excel(writer, sheet_name=name[:31], index=False)
                for name,img in st.session_state.report_plots:
                    ws = writer.book.add_worksheet(name[:31])
                    img_stream = io.BytesIO(img)
                    ws.insert_image('B2', f"{name}.png", {'image_data': img_stream})
            buf.seek(0)
            st.download_button("Download Report", buf, "report.xlsx")

if __name__ == '__main__':
    main()
