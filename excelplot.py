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

# Transformation and fitting options in sidebar
transform_option = st.sidebar.selectbox(
    "Transformation",
    ["None","Baseline subtraction","Log transform","Delta from initial","Z-score normalization","I/I₀ normalization","Min-Max normalization (0–1, sample-wise)"],
    index=0,key="trans_opt"
)
fit_transformed = st.sidebar.checkbox("Fit transformed data instead of raw?", True)
threshold_value = st.sidebar.number_input("Threshold value", 0.0)

model_choices = ["Linear","Sigmoid","4PL","5PL","Gompertz","Exponential"]
definitions = {
    "Linear": lambda x,a,b: a*x+b,
    "Sigmoid": lambda x,a,b: 1/(1+np.exp(-(x-a)/b)),
    "4PL": lambda x,A,B,C,D: D + (A-D)/(1+(x/C)**B),
    "5PL": lambda x,A,B,C,D,G: D + (A-D)/((1+(x/C)**B)**G),
    "Gompertz": lambda x,a,b,c: a*np.exp(-b*np.exp(-c*x)),
    "Exponential": lambda x,a,b: a*np.exp(b*x)
}
def select_models(samples):
    default = st.sidebar.selectbox("Default model", model_choices)
    models = {}
    for s in samples:
        models[s] = st.sidebar.selectbox(f"Model {s}", model_choices, index=model_choices.index(default), key=f"mod_{s}")
    return models

# Session state for report
for key in ["report_plots","report_tables","report_elements"]:
    if key not in st.session_state: st.session_state[key]={}

# Main
if uploaded_file:
    df_all = pd.read_excel(uploaded_file, sheet_name=None)
    sheet = st.sidebar.selectbox("Sheet", list(df_all.keys()))
    df = df_all[sheet]
    st.write(df.head())
    # sample selection
    sample_col = st.sidebar.selectbox("Sample column", df.columns)
    samples = df[sample_col].dropna().unique().tolist()
    selected = st.sidebar.multiselect("Samples to include", samples, default=samples)
    df = df[df[sample_col].isin(selected)]
    # axis select
    nums = df.select_dtypes('number').columns.tolist()
    xcol = st.sidebar.selectbox("X axis", nums)
    ycol = st.sidebar.selectbox("Y axis", nums)
    # preview
    st.subheader("Transformation Preview")
    fig1,ax1=plt.subplots(); fig2,ax2=plt.subplots()
    for s in selected:
        g=df[df[sample_col]==s].sort_values(xcol)
        x=g[xcol];y=g[ycol]
        ax1.plot(x,y,label=s)
        # transform
        if transform_option=="Baseline subtraction": yt=y-y.iloc[0]
        elif transform_option=="Log transform": yt=np.log1p(y)
        elif transform_option=="Delta from initial": yt=y-y.iloc[0]
        elif transform_option=="Z-score normalization": yt=(y-y.mean())/y.std()
        elif transform_option=="I/I₀ normalization": yt=y/y.max()
        elif transform_option=="Min-Max normalization (0–1, sample-wise)": yt=(y-y.min())/(y.max()-y.min())
        else: yt=y
        ax2.plot(x,yt,label=s)
    ax1.legend(bbox_to_anchor=(1,1));ax2.legend(bbox_to_anchor=(1,1))
    st.pyplot(fig1);st.pyplot(fig2)
    # model fitting
    models=select_models(selected)
    st.subheader("Fitting Results")
    fig,ax=plt.subplots()
    param_list=[]; tt_list=[]
    for s in selected:
        g=df[df[sample_col]==s].sort_values(xcol)
        x=g[xcol].values; yraw=g[ycol].values
        y=yraw.copy()
        # apply transform before fit?
        if fit_transformed:
            if transform_option=="Baseline subtraction": y=y-y[0]
            elif transform_option=="Log transform": y=np.log1p(y)
            elif transform_option=="Delta from initial": y=y-y[0]
            elif transform_option=="Z-score normalization": y=(y-y.mean())/y.std()
            elif transform_option=="I/I₀ normalization": y=y/y.max()
            elif transform_option=="Min-Max normalization (0–1, sample-wise)": y=(y-y.min())/(y.max()-y.min())
        try:
            func=definitions[models[s]]
            popt,_=curve_fit(func,x,y,maxfev=10000)
            yfit=func(x, *popt)
            ax.plot(x,y,'o')
            ax.plot(x,yfit,'-')
            # TT
            root=root_scalar(lambda t: func(t,*popt)-threshold_value, bracket=[x.min(),x.max()])
            tt=root.root if root.converged else np.nan
n            ax.axvline(tt,ls='--',c='r')
            param_list.append({"Sample":s,"Params":popt.tolist(),"TT":tt})
        except:
            continue
    ax.set_title("Fits")
    st.pyplot(fig)
    # display tables
    st.table(pd.DataFrame(param_list))
    # export all
    if st.sidebar.button("Export Excel"):
        buf=io.BytesIO(); writer=pd.ExcelWriter(buf,engine='xlsxwriter')
        df.to_excel(writer,'Input',index=False)
        pd.DataFrame(param_list).to_excel(writer,'Results',index=False)
        writer.save();buf.seek(0)
        st.sidebar.download_button("Download",data=buf,file_name='report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
