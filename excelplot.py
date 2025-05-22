import numpy as np
from scipy.optimize import curve_fit, root_scalar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import datetime

# --- Sidebar ---
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx","xls"])

# Transformation and fitting options
transform_option = st.sidebar.selectbox(
    "Transformation",
    [
        "None",
        "Baseline subtraction",
        "Log transform",
        "Delta from initial",
        "Z-score normalization",
        "I/I₀ normalization",
        "Min-Max normalization (0–1, sample-wise)"
    ],
    index=0,
    key="trans_opt"
)
fit_transformed = st.sidebar.checkbox("Fit transformed data instead of raw?", True)
threshold_value = st.sidebar.number_input("Threshold value", 0.0)

# Define available models
model_choices = ["Linear","Sigmoid","4PL","5PL","Gompertz","Exponential"]
definitions = {
    "Linear": lambda x,a,b: a*x + b,
    "Sigmoid": lambda x,a,b: 1/(1 + np.exp(-(x - a)/b)),
    "4PL": lambda x,A,B,C,D: D + (A-D)/(1 + (x/C)**B),
    "5PL": lambda x,A,B,C,D,G: D + (A-D)/((1 + (x/C)**B)**G),
    "Gompertz": lambda x,a,b,c: a * np.exp(-b * np.exp(-c*x)),
    "Exponential": lambda x,a,b: a * np.exp(b*x),
}

def select_models(samples):
    default = st.sidebar.selectbox("Default model", model_choices, key="default_model_all")
    sample_models = {}
    for s in samples:
        sample_models[s] = st.sidebar.selectbox(
            f"Model for {s}",
            model_choices,
            index=model_choices.index(default),
            key=f"mod_{s}"
        )
    return sample_models

# Session state for report
for key in ["report_plots","report_tables","report_elements"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# Main app logic
if uploaded_file:
    # Read all sheets
    df_all = pd.read_excel(uploaded_file, sheet_name=None)
    sheet = st.sidebar.selectbox("Sheet", list(df_all.keys()))
    df = df_all[sheet]
    st.write("Preview:", df.head())

    # Sample selection
    sample_col = st.sidebar.selectbox("Sample column", df.columns, key="sample_col")
    samples = df[sample_col].dropna().unique().tolist()
    selected = st.sidebar.multiselect("Samples to include", samples, default=samples, key="samples")
    df = df[df[sample_col].isin(selected)]

    # Axis selection
    numeric_cols = df.select_dtypes("number").columns.tolist()
    xcol = st.sidebar.selectbox("X axis", numeric_cols, key="xcol")
    ycol = st.sidebar.selectbox("Y axis", numeric_cols, key="ycol")

    # Transformation preview
    st.subheader("Transformation Preview")
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for s in selected:
        g = df[df[sample_col]==s].sort_values(xcol)
        x = g[xcol]; y = g[ycol]
        ax1.plot(x, y, label=s)
        # apply transform
        if transform_option=="Baseline subtraction":
            yt = y - y.iloc[0]
        elif transform_option=="Log transform":
            yt = np.log1p(y)
        elif transform_option=="Delta from initial":
            yt = y - y.iloc[0]
        elif transform_option=="Z-score normalization":
            yt = (y - y.mean())/y.std() if y.std()!=0 else y
        elif transform_option=="I/I₀ normalization":
            yt = y/y.max() if y.max()!=0 else y
        elif transform_option=="Min-Max normalization (0–1, sample-wise)":
            ym, yM = y.min(), y.max()
            yt = (y-ym)/(yM-ym) if yM!=ym else y
        else:
            yt = y
        ax2.plot(x, yt, label=s)
    ax1.legend(bbox_to_anchor=(1,1))
    ax2.legend(bbox_to_anchor=(1,1))
    st.pyplot(fig1)
    st.pyplot(fig2)

    # Model fitting
    sample_models = select_models(selected)
    st.subheader("Fitting Results")
    fig, ax = plt.subplots()
    param_list = []
    for s in selected:
        g = df[df[sample_col]==s].sort_values(xcol)
        x = g[xcol].values
        y_raw = g[ycol].values
        y = y_raw.copy()
        if fit_transformed:
            # same transform logic as above
            if transform_option=="Baseline subtraction":
                y = y - y[0]
            elif transform_option=="Log transform":
                y = np.log1p(y)
            elif transform_option=="Delta from initial":
                y = y - y[0]
            elif transform_option=="Z-score normalization":
                y = (y - np.mean(y))/np.std(y) if np.std(y)!=0 else y
            elif transform_option=="I/I₀ normalization":
                y = y/np.max(y) if np.max(y)!=0 else y
            elif transform_option=="Min-Max normalization (0–1, sample-wise)":
                ym, yM = np.min(y), np.max(y)
                y = (y-ym)/(yM-ym) if yM!=ym else y

        func = definitions[sample_models[s]]
        try:
            popt, _ = curve_fit(func, x, y, maxfev=10000)
            yfit = func(x, *popt)
            ax.plot(x, y, 'o', label=f"{s} data")
            ax.plot(x, yfit, '-', label=f"{s} fit")
            # compute TT
            root = root_scalar(lambda t: func(t,*popt)-threshold_value,
                               bracket=[x.min(), x.max()])
            tt = root.root if root.converged else np.nan
            ax.axvline(tt, linestyle="--", color="red", label=f"{s} TT")
            param_list.append({"Sample":s, "Params":list(popt), "TT":tt})
        except Exception as e:
            st.warning(f"Fit failed for {s}: {e}")

    ax.set_title("Fits + Thresholds")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.legend(bbox_to_anchor=(1,1))
    st.pyplot(fig)

    # Results table
    res_df = pd.DataFrame(param_list)
    st.subheader("Fit Parameters & TT")
    st.dataframe(res_df)

    # Export
    if st.sidebar.button("Export Excel"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Input", index=False)
            res_df.to_excel(writer, sheet_name="Results", index=False)
        buf.seek(0)
        st.sidebar.download_button(
            "Download Report",
            data=buf,
            file_name="analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
