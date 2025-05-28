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
MODEL_PARAM_NAMES = {
    "Linear": ["a", "b"],
    "Sigmoid": ["a", "b"],
    "4PL": ["A", "B", "C", "D"],
    "5PL": ["A", "B", "C", "D", "G"],
    "Gompertz": ["a", "b", "c"],
    "Don Levin Sigmoid 2D": ["a1","b1","c1","a2","b2","c2","a3","b3","c3"]
}

# Formula templates for export
FORMULA_TEMPLATES = {
    "Linear":      "y = {a}*x + {b}",
    "Sigmoid":     "y = 1/(1+exp(-(x-{a})/{b}))",
    "4PL":         "y = {D} + ({A}-{D})/(1+(x/{C})**{B})",
    "5PL":         "y = {D} + ({A}-{D})/((1+(x/{C})**{B})**{G})",
    "Gompertz":    "y = {a} * exp(-{b} * exp(-{c}*x))",
    "Don Levin Sigmoid 2D": "y = {a1}/(1+exp(-(x-{b1})/{c1})) + {a2}/(1+exp(-(x-{b2})/{c2})) + {a3}/(1+exp(-(x-{b3})/{c3}))"
}


# Inverse formula templates for export
FORMULA_INV_TEMPLATES = {
    "Linear":      "x = (y - {b}) / {a}",
    "Sigmoid":     "x = {a} + {b} * LN(y/(1-y))",
    "4PL":         "x = {C} * POWER(((A-D)/(y-D) - 1), 1/{B})",
    "5PL":         "x = {C} * POWER((( (A-D)/(y-D))^(1/{G}) - 1), 1/{B})",
    "Gompertz":    "x = -(1/{c}) * LN(-LN(y/{a})/{b})",
    "Don Levin Sigmoid 2D": "(No closed-form inverse)"
} / {a}",
    "Sigmoid":     "x = {a} + {b} * ln(y/(1-y))",
    "4PL":         "x = {C} * ((({A}-{D})/(y - {D}) - 1))**(1/{B})",
    "5PL":         "x = {C} * (((( {A}-{D})/(y - {D}))**(1/{G}) - 1))**(1/{B})",
    "Gompertz":    "x = -(1/{c}) * ln(-ln(y/{a})/{b})",
    "Don Levin Sigmoid 2D": "(No closed-form inverse)"
}) / {a}",
    "Sigmoid":     "x = {a} + {b} * ln(y/(1-y))",
    "4PL":         "x = {C} * ((({A}-{D})/(y - {D}) - 1))**(1/{B})",
    "5PL":         "x = {C} * (((( {A}-{D})/(y - {D}))**(1/{G}) - 1))**(1/{B})",
    "Gompertz":    "x = -(1/{c}) * ln(-ln(y/{a})/{b})"
}

MODELS = {
    "Linear":      (lambda x,a,b: a*x + b, [1,0]),
    "Sigmoid":     (lambda x,a,b: 1/(1 + np.exp(-(x-a)/b)), [np.median,1]),
    "4PL":         (lambda x,A,B,C,D: D + (A-D)/(1+(x/C)**B), [1,1,1,0]),
    "5PL":         (lambda x,A,B,C,D,G: D + (A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz":    (lambda x,a,b,c: a * np.exp(-b * np.exp(-c*x)), [1,1,1]),
    "Don Levin Sigmoid 2D": (
        lambda x,a1,b1,c1,a2,b2,c2,a3,b3,c3:
            a1/(1+np.exp(-(x-b1)/c1))
            + a2/(1+np.exp(-(x-b2)/c2))
            + a3/(1+np.exp(-(x-b3)/c3)),
        [1, np.median, 1, 1, np.median, 1, 1, np.median, 1]
    )
}

# -----------------------------
# 4. Transform Application
# -----------------------------
def apply_transforms(group_df: pd.DataFrame, transforms: list, y_col: str, sample_col: str) -> pd.Series:
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
# 5. Model Fitting Helper
# -----------------------------
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
# 6. Interactive Plotting
# -----------------------------
def plot_interactive(df: pd.DataFrame, df0: pd.DataFrame, x_col: str, y_col: str,
                     sample_col: str, transforms: list, threshold: float):
    fig = go.Figure(); fig.update_layout(dragmode='lasso')
    core_trans = transforms[:]  # no Remove T0
    # backdrop original data
    for sample in df0[sample_col].dropna().unique():
        g0 = df0[df0[sample_col]==sample].sort_values(x_col)
        fig.add_trace(go.Scatter(x=g0[x_col], y=g0[y_col], mode='markers', marker_color='lightgrey', showlegend=False))
    # current data and transforms
    for sample in df[sample_col].unique():
        g = df[df[sample_col]==sample].sort_values(x_col)
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
# 7. Main App
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
    sheet = st.sidebar.selectbox("Sheet", list(sheets.keys()), key="sheet_select")
    # Reset interactive dataframe when sheet changes
    if 'current_sheet' not in st.session_state or st.session_state.current_sheet != sheet:
        st.session_state.dfi = sheets[sheet].copy()
        st.session_state.current_sheet = sheet
    df0_raw = sheets[sheet]
    # Select data layout: long table or side-by-side blocks
    layout = st.sidebar.radio("Data layout", ["Long table", "Blocks side-by-side"], index=0)
    # Allow user to specify detection patterns (case-insensitive)
    vial_pat = st.sidebar.text_input("Vial column contains", "vial").lower()
    time_pat = st.sidebar.text_input("Time column contains", "time").lower()
    signal_pat = st.sidebar.text_input("Signal column contains", "signal").lower()
    if layout == "Blocks side-by-side":
        # Detect repeating (Vial ID, Time, Signal) blocks based on user patterns
        blocks = []
        cols = df0_raw.columns.tolist()
        for i in range(len(cols)-2):
            c0, c1, c2 = cols[i].lower(), cols[i+1].lower(), cols[i+2].lower()
            if vial_pat in c0 and time_pat in c1 and signal_pat in c2:
                blocks.append((cols[i], cols[i+1], cols[i+2]))
        if blocks:
            parsed = []
            for cid, ctime, csignal in blocks:
                tmp = df0_raw[[cid, ctime, csignal]].copy()
                tmp.columns = ["Vial ID", "Time", "Signal"]
                parsed.append(tmp)
            df0 = pd.concat(parsed, ignore_index=True)
        else:
            st.warning(f"No ({{vial_pat}}, {{time_pat}}, {{signal_pat}}) blocks detected; using sheet as-is.")
            df0 = df0_raw.copy()
    else:
        # Assume sheet already in long format with appropriate columns
        df0 = df0_raw.copy()
    else:
        # Assume sheet already in long format with appropriate columns
        df0 = df0_raw.copy()
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
    global_model = st.sidebar.selectbox(
        "Global Model", list(MODELS.keys()), index=list(MODELS.keys()).index("4PL")
    )
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

    # Prepare processed data with transformations applied
    processed_df = []
    for sample in df[sample_col].unique():
        grp = df[df[sample_col]==sample].sort_values(x_col)
        y_trans = apply_transforms(grp, transforms, y_col, sample_col)
        grp_proc = grp.copy()
        grp_proc[y_col] = y_trans.values
        processed_df.append(grp_proc)
    processed_df = pd.concat(processed_df, ignore_index=True)

    # Model fitting & report
    st.header("Model Fitting Results")
    x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    params_list, tt_list, fit_data = [], [], []
    fig_fit = go.Figure(); fig_fit.update_layout(title='Fitted Curves')

    for sample in df[sample_col].unique():
        grp = df[df[sample_col]==sample].sort_values(x_col)
        model = global_model if use_global else st.sidebar.selectbox(
                f"Model for {sample}", list(MODELS.keys()), index=list(MODELS.keys()).index("4PL")
            )
        x_vals = grp[x_col].values
        y_vals = apply_transforms(grp, transforms, y_col, sample_col).values
        try:
            popt, pcov, y_fit_vals, r2, rmse, tt, ttse = fit_and_eval(x_vals, y_vals, model, threshold)
            # record asymptotes
            if model in ('4PL','5PL'):
                A, D = popt[0], popt[3]
            else:
                D, A = y_fit_vals.min(), y_fit_vals.max()
            asymptotes[sample] = (A, D)
            # plot data and fit
            fig_fit.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name=f"{sample} data"))
            y_line = MODELS[model][0](x_lin, *popt)
            fig_fit.add_trace(go.Scatter(x=x_lin, y=y_line, mode='lines', name=f"{sample} fit"))            # record parameters
            param_names = MODEL_PARAM_NAMES[model]
            param_dict = {name: round(val, 4) for name, val in zip(param_names, popt)}
            # add forward formula
            try:
                formula = FORMULA_TEMPLATES[model].format(**param_dict)
            except Exception:
                formula = ''
            # add inverse formula
            try:
                inv_formula = FORMULA_INV_TEMPLATES[model].format(**param_dict)
            except Exception:
                inv_formula = ''
            param_dict.update({
                "Sample": sample,
                "Model": model,
                "Formula": formula,
                "Inverse Formula": inv_formula,
                "R2": round(r2, 4),
                "RMSE": round(rmse, 4)
            })
            params_list.append(param_dict)
            tt_list.append((sample, tt, ttse))
            # collect fit curve data
            fit_df = pd.DataFrame({sample_col: sample, x_col: x_lin, f"{y_col}_fit": y_line})
            fit_data.append(fit_df)
        except Exception as e:
            st.warning(f"Fit error for {sample}: {e}")

    fig_fit.add_hline(y=threshold, line_dash='dash')
    st.plotly_chart(fig_fit, use_container_width=True)

    if st.button("Add Final Processed Data to Report"):
        # Append the processed (transformed) data
        report_tables.append(("Final Processed Data", processed_df.copy()))
    if st.button("Add Final Fitting Data to Report"):
        report_tables.append(("Fitting Curves", pd.concat(fit_data, ignore_index=True)))

    st.subheader("Fit Parameters")
    st.dataframe(pd.DataFrame(params_list))
    st.subheader("Time to Threshold")
    st.dataframe(pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]))

    if st.button("Export Report"):
        # Prepare a sheet with generic formulas for each model (including parameter names)
        formula_records = []
        for m in FORMULA_TEMPLATES.keys():
            params = ", ".join(MODEL_PARAM_NAMES.get(m, []))
            formula_records.append({
                "Model": m,
                "Parameters": params,
                "Formula": FORMULA_TEMPLATES[m],
                "Inverse Formula": FORMULA_INV_TEMPLATES[m]
            })
        formula_df = pd.DataFrame(formula_records)

        fit_df_all = pd.concat(fit_data, ignore_index=True)
        report_items = [
            ("Model Formulas", formula_df),
            ("Original Data", df0),
            ("Edited Data", st.session_state.dfi),
            ("Processed Data", processed_df),
            ("Final Processed Data", processed_df),
            ("Fitting Curves Data", fit_df_all)
        ] + report_tables

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            for name,tbl in report_items:
                tbl.to_excel(writer, sheet_name=name[:31], index=False)
            pd.DataFrame(params_list).to_excel(writer, sheet_name='Fit Parameters', index=False)
            pd.DataFrame(tt_list, columns=["Sample","TT","TT_SE"]).to_excel(writer, sheet_name='Time to Threshold', index=False)
            try:
                img_buf = io.BytesIO()
                fig_fit.write_image(img_buf, format='png')
                img_buf.seek(0)
                ws = writer.book.add_worksheet('Fitted Plot')
                ws.insert_image('B2', 'fitted_plot.png', {'image_data': img_buf})
            except Exception:
                pass
        buf.seek(0)
        st.download_button(
            label="Download Report",
            data=buf,
            file_name="Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()
