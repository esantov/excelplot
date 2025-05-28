# Streamlit Analysis App with Global Model Override, Interactive Plotting, and Reporting
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit, root_scalar

# Optional import
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None
    st.sidebar.warning("Install 'streamlit-plotly-events' for point selection.")

# Global asymptote storage
asymptotes = {}

# -----------------------------
# Utility
# -----------------------------
@st.cache_data
def load_excel_sheets(file):
    return pd.read_excel(file, sheet_name=None)

# -----------------------------
# Transforms
# -----------------------------
def fix_initial_baseline(y):
    y = y.copy()
    if len(y)>1 and y.iloc[0] > y.iloc[1]: y.iloc[0] = y.iloc[1]
    return y

def uniform_asymptote(y, sample, rng):
    A,D = asymptotes.get(sample, (y.max(), y.min()))
    y0 = (y - D)/((A-D) or 1)
    return y0*(rng[1]-rng[0])+rng[0]

TRANSFORMS = {
    "None": lambda y: y,
    "Remove T0": lambda y: y.iloc[1:] if len(y)>1 else y,
    "Fix initial baseline": fix_initial_baseline,
    "Baseline subtraction": lambda y: y - y.iloc[0],
    "Log transform": lambda y: np.log1p(y),
    "Delta from initial": lambda y: y - y.iloc[0],
    "Z-score norm": lambda y: (y - y.mean())/(y.std() or 1),
    "I/I0 norm": lambda y: y/(y.iloc[0] or 1),
    "MinMax 0-1": lambda y: (y - y.min())/((y.max()-y.min()) or 1),
    "Scale 18-55": lambda y: ((y - y.min())/((y.max()-y.min()) or 1))*(55-18)+18,
    "CumMax": lambda y: y.cummax(),
    "Uniform 0-1": lambda y, s: uniform_asymptote(y,s,(0,1)),
    "Uniform 18-55": lambda y, s: uniform_asymptote(y,s,(18,55))
}

# sequential apply
def apply_transforms(grp, trans_list, ycol, sc):
    y = grp[ycol]
    for t in trans_list:
        func = TRANSFORMS[t]
        try: y = func(y, grp[sc].iloc[0])
        except: y = func(y)
    return y

# -----------------------------
# Models
# -----------------------------
MODELS = {
    "Linear": (lambda x,a,b: a*x+b, [1,0]),
    "Sigmoid": (lambda x,a,b:1/(1+np.exp(-(x-a)/b)), [np.median,1]),
    "4PL": (lambda x,A,B,C,D: D+(A-D)/(1+(x/C)**B), [1,1,1,0]),
    "5PL": (lambda x,A,B,C,D,G: D+(A-D)/((1+(x/C)**B)**G), [1,1,1,0,1]),
    "Gompertz": (lambda x,a,b,c: a*np.exp(-b*np.exp(-c*x)), [1,1,1])
}

def fit_and_eval(x,y,model,thr):
    func,_= MODELS[model]
    popt,pcov=curve_fit(func,x,y,maxfev=10000)
    yfit=func(x,*popt)
    rss=((y-yfit)**2).sum(); tss=((y-y.mean())**2).sum()
    r2=1-rss/tss if tss else np.nan
    rmse=np.sqrt(rss/len(y))
    tt,ttse="N/A","N/A"
    if yfit.min()<=thr<=yfit.max():
        r=root_scalar(lambda t:func(t,*popt)-thr, bracket=[x.min(),x.max()])
        if r.converged:
            tt=r.root
            d=(func(tt+1e-5,*popt)-func(tt-1e-5,*popt))/(2e-5)
            var=(d**-2)*pcov.sum() if pcov.size else 0
            ttse=np.sqrt(var) if var>0 else np.nan
    return popt,pcov,yfit,r2,rmse,tt,ttse

# -----------------------------
# Interactive Plot
# -----------------------------
def plot_interactive(df, df0, xc,yc,sc,trans,thr):
    fig=go.Figure(); fig.update_layout(dragmode='lasso')
    rem='Remove T0' in trans
    real=[t for t in trans if t!='Remove T0']
    for s in df0[sc].unique():
        g0=df0[df0[sc]==s].sort_values(xc)
        fig.add_trace(go.Scatter(x=g0[xc], y=g0[yc], mode='markers', marker_color='lightgrey', showlegend=False))
    for s in df[sc].unique():
        g=df[df[sc]==s].sort_values(xc)
        if rem and len(g)>1: g=g.iloc[1:]
        ytr=apply_transforms(g, real,yc,sc)
        fig.add_trace(go.Scatter(x=g[xc],y=g[yc],mode='markers',name=f'{s} raw', customdata=g.index))
        fig.add_trace(go.Scatter(x=g[xc],y=ytr,mode='lines',name=f'{s} trans', line_dash='dash'))
    fig.add_hline(y=thr)
    sel=[]
    if plotly_events: sel=plotly_events(fig, select_event=True)
    st.plotly_chart(fig)
    return sel

# -----------------------------
# Main
# -----------------------------
def main():
    st.sidebar.title('Settings')
    f=st.sidebar.file_uploader('Excel', type=['xlsx','xls'])
    if not f: return
    sheets=load_excel_sheets(f)
    sh=st.sidebar.selectbox('Sheet', list(sheets.keys()))
    df0=sheets[sh]
    if 'dfi' not in st.session_state: st.session_state.dfi=df0.copy()
    df=st.session_state.dfi.copy()
    st.dataframe(df.head())
    sc=st.sidebar.selectbox('Sample col',df.columns)
    sel=st.sidebar.multiselect('Samples',df[sc].unique(),default=list(df[sc].unique()))
    df=df[df[sc].isin(sel)]
    nums=df.select_dtypes('number').columns.tolist()
    xc=st.sidebar.selectbox('X',nums);  yc=st.sidebar.selectbox('Y',nums)
    opts=list(TRANSFORMS.keys()); trans=st.sidebar.multiselect('Transforms',opts,default=['None'])
    thr=st.sidebar.number_input('Thr',value=1.0)
    # global model
    gm=st.sidebar.selectbox('Global model',list(MODELS.keys()))
    fa=st.sidebar.checkbox('Use global model')
    selpts=plot_interactive(df,df0,xc,yc,sc,trans,thr)
    if selpts and st.button('Remove Selected'): st.session_state.dfi=df0.drop(index=[p['customdata'] for p in selpts])
        # edits (make table editing version-agnostic)
    st.subheader("Data Table (Editable)")
    if hasattr(st, 'data_editor'):
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edited_df = st.experimental_data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.warning("Your Streamlit version does not support an editable table. Skipping.")
        edited_df = df.copy()
    if st.button('Apply edits'):
        st.session_state.dfi = edited_df.copy()
        st.experimental_rerun()
    df = edited_df.copy()

    # fitting & reporting
    st.header('Fits')
    xlin=np.linspace(df[xc].min(),df[xc].max(),200)
    params=[]; tts=[]; allfit=[]
    fig=go.Figure(); fig.update_layout(title='Fits')
    for s in df[sc].unique():
        g=df[df[sc]==s].sort_values(xc)
        if fa: model=gm
        else: model=st.sidebar.selectbox(f'Model {s}',list(MODELS.keys()))
        xarr=g[xc].values
        yarr=apply_transforms(g,trans,yc,sc).values
        popt,pc,yf,r2,rmse,tt,ttse=fit_and_eval(xarr,yarr,model,thr)
        if model in ('4PL','5PL'): A,D=popt[0],popt[3]
        else: D,A=yf.min(),yf.max()
        asymptotes[s]=(A,D)
        fig.add_trace(go.Scatter(x=xarr,y=yarr,mode='markers',name=f'{s} data'))
        yfit=MODELS[model][0](xlin,*popt)
        fig.add_trace(go.Scatter(x=xlin,y=yfit,mode='lines',name=f'{s} fit'))
        params.append({'Sample':s,'Model':model,'R2':round(r2,3),'RMSE':round(rmse,3)})
        tts.append((s,tt,ttse))
        dfc=pd.DataFrame({sc:s,xc:xlin,yc+'_fit':yfit})
        allfit.append(dfc)
    fig.add_hline(y=thr)
    st.plotly_chart(fig)
    if st.button('Add final data'): st.session_state.report_tables.append(('Final Data', apply_transforms(df,trans,yc,sc).to_frame()))
    if st.button('Add final fits'): st.session_state.report_tables.append(('Fit Curves',pd.concat(allfit,ignore_index=True)))
    # export
    if st.button('Export Report'):
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='xlsxwriter') as w:
            df0.to_excel(w,'Original',index=False)
            st.session_state.dfi.to_excel(w,'Edited',index=False)
            for name,tab in st.session_state.report_tables: tab.to_excel(w,name[:31],index=False)
        buf.seek(0)
        st.download_button('Download',buf,'report.xlsx')

if __name__=='__main__': main()
