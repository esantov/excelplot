# Streamlit Analysis App with Interactive Transformations, Fitting, and Reporting
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit, root_scalar

# Optional import for point selection
HAS_PLOTLY_EVENTS = True
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None
    HAS_PLOTLY_EVENTS = False

# Global storage
asymptotes = {}
report_tables = []

# -----------------------------
# 1. Data Loading
# -----------------------------
@st.cache_data
def load_excel_sheets(file) -> dict:
    return pd.read_excel(file, sheet_name=None)

# -----------------------------
# 2. Transform Functions
# -----------------------------
def fix_initial_baseline(y: pd.Series) -> pd.Series:
    y = y.copy()
    if len(y) > 1 and y.iloc[0] > y.iloc[1]:
        y.iloc[0] = y.iloc[1]
    return y


def uniform_asymptote(y: pd.Series, sample: str, out_range=(0,1)) -> pd.Series:
    A, D = asymptotes.get(sample, (y.max(), y.min()))
    y0 = (y - D) / ((A - D) or 1)
    return y0 * (out_range[1] - out_range[0]) + out_range[0]

TRANSFORMS = {
    "None": lambda y: y,
    "Remove T0": lambda y: y.iloc[1:] if len(y)>1 else y,
    "Fix initial baseline": fix_initial_baseline,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score normalization": lambda y: (y - y.mean()) / (y.std() or 1),
    "I/I0 normalization": lambda y: y / (y.iloc[0] or 1),
    "Min-Max normalization (0-1)": lambda y: (y - y.min()) / ((y.max() - y.min()) or 1),
    "Scale to [18-55]": lambda y: ((y - y.min()) / ((y.max() - y.min()) or 1)) * (55 - 18) + 18,
    "Cumulative Max": lambda y: y.cummax(),
    "Uniform Asymptotes (0-1)": lambda y, s=None: uniform_asymptote(y, s, (0,1)),
    "Uniform Asymptotes (18-55)": lambda y, s=None: uniform_asymptote(y, s, (18,55)),
}

# -----------------------------
# 3. Model Definitions
# -----------------------------
MODELS = {
    "Linear":      (lambda x,a,b: a*x + b, [1,0]),
    "Sigmoid":     (lambda x,a,b: 1/(1 + np.exp(-(x-a)/b)), [np.median,1]),
    "4PL":         (lambda x,A,B,C,D: D + (A-D)/(1+(x/C)**B), [1,1,1,0]),
    "5PL":         (lambda x,A,B,C,D,G: D + (A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz":    (lambda x,a,b,c: a * np.exp(-b * np.exp(-c*x)), [1,1,1])
}

def fit_and_eval(x: np.ndarray, y: np.ndarray, model_name: str, threshold: float):
    func, _ = MODELS[model_name]
    popt, pcov = curve_fit(func, x, y, maxfev=10000)
    y_fit = func(x, *popt)
    rss = np.sum((y - y_fit)**2)
    tss = np.sum((y - np.mean(y))**2)
    r2 = 1 - rss/tss if tss else np.nan
    rmse = np.sqrt(rss/len(y))
    tt, tt_se = "N/A", "N/A"
    if y_fit.min() <= threshold <= y_fit.max():
        root = root_scalar(lambda t: func(t, *popt) - threshold, bracket=[x.min(), x.max()])
        if root.converged:
            tt = root.root
            deriv = (func(tt+1e-5,*popt) - func(tt-1e-5,*popt)) / 2e-5
            var = (deriv**-2) * np.sum(np.diag(pcov)) if pcov.size else 0
            tt_se = np.sqrt(var) if var>0 else np.nan
    return popt, pcov, y_fit, r2, rmse, tt, tt_se

# -----------------------------
# 4. Plotting
# -----------------------------

def plot_interactive(df, df0, x_col, y_col, sample_col, transforms, threshold):
    fig = go.Figure(); fig.update_layout(dragmode='lasso')
    remove_t0 = "Remove T0" in transforms
    core_trans = [t for t in transforms if t != "Remove T0"]
    # backdrop
    for sample in df0[sample_col].dropna().unique():
        g0 = df0[df0[sample_col]==sample].sort_values(x_col)
        fig.add_trace(go.Scatter(x=g0[x_col], y=g0[y_col], mode='markers', marker_color='lightgrey', showlegend=False))
    # current
    for sample in df[sample_col].unique():
        g = df[df[sample_col]==sample].sort_values(x_col)
        if remove_t0 and len(g)>1:
            g = g.iloc[1:]
        y_tr = apply_transforms(g, core_trans, y_col, sample_col)
        fig.add_trace(go.Scatter(x=g[x_col], y=g[y_col], mode='markers', name=f"{sample} raw", customdata=g.index))
        fig.add_trace(go.Scatter(x=g[x_col], y=y_tr, mode='lines', name=f"{sample} trans", line_dash='dash'))
    fig.add_hline(y=threshold, line_dash='dot')
    selected = []
    if HAS_PLOTLY_EVENTS:
        selected = plotly_events(fig, select_event=True)
    st.plotly_chart(fig, use_container_width=True)
    return selected

# -----------------------------
# 5. Transform Application
# -----------------------------

def apply_transforms(group_df, transforms, y_col, sample_col):
    y = group_df[y_col].copy()
    sample = group_df[sample_col].iloc[0]
    for t in transforms:
        func = TRANSFORMS[t]
        try:
            y = func(y, sample)
        except TypeError:
            y = func(y)
    return y

# -----------------------------
# 6. Main App
# -----------------------------
def main():
    st.sidebar.title("Settings")
    if not HAS_PLOTLY_EVENTS:
        st.sidebar.warning("Install 'streamlit-plotly-events' for selection.")

    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx","xls"])
    if not uploaded:
        st.info("Upload a file to begin.")
        return
    sheets = load_excel_sheets(uploaded)
    sheet = st.sidebar.selectbox("Sheet", list(sheets.keys()))
    df0 = sheets[sheet]
    if 'dfi' not in st.session_state:
        st.session_state.dfi = df0.copy()
    df = st.session_state.dfi.copy()

    st.dataframe(df.head())
    sample_col = st.sidebar.selectbox("Sample Column", df.columns)
    sel_samps = st.sidebar.multiselect("Samples", df[sample_col].dropna().unique(), default=list(df[sample_col].dropna().unique()))
    df = df[df[sample_col].isin(sel_samps)]

    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols)<2:
        st.warning("Need ≥2 numeric columns.")
        return
    x_col = st.sidebar.selectbox("X Column", num_cols)
    y_col = st.sidebar.selectbox("Y Column", num_cols)

    transforms = st.sidebar.multiselect("Transforms", list(TRANSFORMS.keys()), default=["None"])
    threshold = st.sidebar.number_input("Threshold", value=1.0)
    global_model = st.sidebar.selectbox("Global Model", list(MODELS.keys()))
    use_global = st.sidebar.checkbox("Use Global Model for All")

    selpts = plot_interactive(df, df0, x_col, y_col, sample_col, transforms, threshold)
    if selpts and st.button("Remove Selected Points"):
        idxs = [pt['customdata'] for pt in selpts]
        st.session_state.dfi = df0.drop(index=idxs)
        st.experimental_rerun()

    # Editable table
    st.subheader("Data Table (Editable)")
    if hasattr(st, 'data_editor'):
        edf = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edf = st.experimental_data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.warning("Editable tables unsupported.")
        edf = df.copy()
    if st.button("Apply Table Edits"):
        st.session_state.dfi = edf.copy()
        st.experimental_rerun()
    df = edf.copy()

    # Model fitting & report
    st.header("Model Fitting Results")
    x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    params_list, tt_list, fit_data = [], [], []
    fig_fit = go.Figure(); fig_fit.update_layout(title='Fitted Curves')

    for sample in df[sample_col].unique():
        grp = df[df[sample_col]==sample].sort_values(x_col)
        # apply T0 removal
        if 'Remove T0' in transforms and len(grp)>1:
            grp = grp.iloc[1:]
        model = global_model if use_global else st.sidebar.selectbox(f"Model for {sample}", list(MODELS.keys()))
        x_vals = grp[x_col].values
        y_vals = apply_transforms(grp, transforms, y_col, sample_col).values
        try:
            popt, pcov, y_fit_vals, r2, rmse, tt, ttse = fit_and_eval(x_vals, y_vals, model, threshold)
            if model in ('4PL','5PL'):
                A, D = popt[0], popt[3]
            else:
                D, A = y_fit_vals.min(), y_fit_vals.max()
            asymptotes[sample] = (A, D)
            fig_fit.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name=f"{sample} data"))
            y_line = MODELS[model][0](x_lin, *popt)
            fig_fit.add_trace(go.Scatter(x=x_lin, y=y_line, mode='lines', name=f"{sample} fit"))
            params_list.append({"Sample": sample, "Model": model, "R2": round(r2,4), "RMSE": round(rmse,4)})
            tt_list.append((sample, tt, ttse))
            fit_df = pd.DataFrame({sample_col: sample, x_col: x_lin, f"{y_col}_fit": y_line})
            fit_data.append(fit_df)
        except Exception as e:
            st.warning(f"Fit error for {sample}: {e}")

    fig_fit.add_hline(y=threshold, line_dash='dash')
    st.plotly_chart(fig_fit, use_container_width=True)

    if st.button("Add Final Processed Data to Report"):
        report_tables.append(("Final Data", df.copy()))
    if st.button("Add Final Fitting Data to Report"):
        report_tables.append(("Fitting Curves", pd.concat(fit_data, ignore_index=True)))

    st.subheader("Fit Parameters")
    st.dataframe(pd.DataFrame(params_list))
    st.subheader("Time to Threshold")
    st.dataframe(pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]))

    if st.button("Export Report"):
        # Always include final processed and fitting data
        fit_df_all = pd.concat(fit_data, ignore_index=True)
        report_items = [
            ("Original Data", df0),
            ("Edited Data", st.session_state.dfi),
            ("Processed Data", df),
            ("Final Processed Data", df),
            ("Fitting Curves Data", fit_df_all),
        ] + report_tables

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            for name, tbl in report_items:
                # ensure sheet name length <=31
                writer_df = tbl.copy()
                safe_name = name[:31]
                writer_df.to_excel(writer, sheet_name=safe_name, index=False)

            # Fitted parameters and TT
            pd.DataFrame(params_list).to_excel(writer, sheet_name='Fit Parameters', index=False)
            pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]).to_excel(writer, sheet_name='Time to Threshold', index=False)

            # Insert fitted plot image
            try:
                img_buf = io.BytesIO()
                fig_fit.write_image(img_buf, format='png')
                img_buf.seek(0)
                worksheet = writer.book.add_worksheet('Fitted Plot')
                worksheet.insert_image('B2', 'fitted_plot.png', {'image_data': img_buf})
            except Exception:
                pass

        buf.seek(0)
        st.download_button(
            label="Download Report",
            data=buf,
            file_name="Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
"Download Report",
data=buf,
file_name="Analysis_Report.xlsx",
mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
