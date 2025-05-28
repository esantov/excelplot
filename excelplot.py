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
# Main Application
# -----------------------------

def main():
    st.sidebar.title('Settings')
    f = st.sidebar.file_uploader('Excel', type=['xlsx','xls'])
    if not f:
        return
    sheets = load_excel_sheets(f)
    sh = st.sidebar.selectbox('Sheet', list(sheets.keys()))
    df0 = sheets[sh]

    # Initialize or load interactive df
    if 'dfi' not in st.session_state:
        st.session_state.dfi = df0.copy()
    df = st.session_state.dfi.copy()

    # Display raw data
    st.dataframe(df.head())

    # Sample selection
    sc = st.sidebar.selectbox('Sample column', df.columns)
    sel = st.sidebar.multiselect('Samples', df[sc].dropna().unique(), default=list(df[sc].dropna().unique()))
    df = df[df[sc].isin(sel)]

    # X and Y axis selection
    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols) < 2:
        st.warning('Need at least two numeric columns.')
        return
    xc = st.sidebar.selectbox('X axis', num_cols)
    yc = st.sidebar.selectbox('Y axis', num_cols)

    # Transform sequence
    transform_opts = list(TRANSFORMS.keys())
    transforms = st.sidebar.multiselect('Transforms', transform_opts, default=['None'])

    # Threshold
    thr = st.sidebar.number_input('Threshold', value=1.0)

    # Global model override
    gm = st.sidebar.selectbox('Global model', list(MODELS.keys()))
    use_global = st.sidebar.checkbox('Use global model')

    # Interactive plot
    selpts = plot_interactive(df, df0, xc, yc, sc, transforms, thr)
    if selpts and st.button('Remove Selected'):
        indices = [pt['customdata'] for pt in selpts]
        st.session_state.dfi = df0.drop(index=indices)
        st.experimental_rerun()

    # Editable table
    st.subheader('Data Table (Editable)')
    if hasattr(st, 'data_editor'):
        edited_df = st.data_editor(df, num_rows='dynamic', use_container_width=True)
    elif hasattr(st, 'experimental_data_editor'):
        edited_df = st.experimental_data_editor(df, num_rows='dynamic', use_container_width=True)
    else:
        st.warning('Streamlit version does not support data editing.')
        edited_df = df.copy()
    if st.button('Apply Edits'):
        st.session_state.dfi = edited_df.copy()
        st.experimental_rerun()
    df = edited_df.copy()

    # Model fitting and reporting
    st.header('Model Fitting Results')
    x_lin = np.linspace(df[xc].min(), df[xc].max(), 200)
    params = []
    tts = []
    fit_curves = []
    fig_fit = go.Figure()
    fig_fit.update_layout(title='Fitted Curves')

    for sample in df[sc].unique():
        grp = df[df[sc] == sample].sort_values(xc)
        # choose model
        model = gm if use_global else st.sidebar.selectbox(f'Model for {sample}', list(MODELS.keys()))
        x_vals = grp[xc].values
        y_vals = apply_transforms(grp, transforms, yc, sc).values
        try:
            popt, pcov, y_fit_vals, r2, rmse, tt, ttse = fit_and_eval(x_vals, y_vals, model, thr)
            # record asymptotes
            if model in ('4PL', '5PL'):
                A, D = popt[0], popt[3]
            else:
                D, A = y_fit_vals.min(), y_fit_vals.max()
            asymptotes[sample] = (A, D)
            # plot
            fig_fit.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name=f'{sample} data'))
            y_line = MODELS[model][0](x_lin, *popt)
            fig_fit.add_trace(go.Scatter(x=x_lin, y=y_line, mode='lines', name=f'{sample} fit'))
            # collect parameters
            params.append({'Sample': sample, 'Model': model, 'R2': round(r2, 4), 'RMSE': round(rmse, 4)})
            tts.append((sample, tt, ttse))
            fit_df = pd.DataFrame({sc: sample, xc: x_lin, f'{yc}_fit': y_line})
            fit_curves.append(fit_df)
        except Exception as e:
            st.warning(f'Fit error for {sample}: {e}')

    fig_fit.add_hline(y=thr, line_dash='dash')
    st.plotly_chart(fig_fit, use_container_width=True)

    # Add to report buttons
    if st.button('Add Final Processed Data to Report'):
        st.session_state.report_tables.append(('Final Data', df.copy()))
    if st.button('Add Final Fitting Curves to Report'):
        st.session_state.report_tables.append(('Fitting Curves', pd.concat(fit_curves, ignore_index=True)))

    # Display summary tables
    st.subheader('Fit Parameters')
    st.dataframe(pd.DataFrame(params))
    st.subheader('Time to Threshold')
    st.dataframe(pd.DataFrame(tts, columns=['Sample', 'TT', 'TT_SE']))

    # Export report
    if st.button('Export Report'):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df0.to_excel(writer, sheet_name='Original Data', index=False)
            st.session_state.dfi.to_excel(writer, sheet_name='Edited Data', index=False)
            for name, table in st.session_state.report_tables:
                table.to_excel(writer, sheet_name=name[:31], index=False)
        buf.seek(0)
        st.download_button('Download Report', data=buf, file_name='Analysis_Report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    main()
