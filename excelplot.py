import numpy as np
from scipy.optimize import curve_fit, root_scalar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import xlsxwriter
import datetime

with st.sidebar:
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if "report_plots" not in st.session_state:
    st.session_state["report_plots"] = []
if "report_tables" not in st.session_state:
    st.session_state["report_tables"] = []
if "report_index" not in st.session_state:
    st.session_state["report_index"] = 0
if "report_elements" not in st.session_state:
    st.session_state["report_elements"] = {}

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        with st.sidebar:
            selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.dataframe(df.head())

        with st.sidebar:
            sample_column = st.selectbox("Select the column that contains sample identifiers", df.columns, key="sample_column")
            unique_samples = df[sample_column].dropna().unique()
            selected_samples = st.multiselect("Filter by sample (optional)", unique_samples, default=list(unique_samples), key="sample_filter")

        if selected_samples:
            df = df[df[sample_column].isin(selected_samples)]

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) >= 2:
            with st.sidebar:
                x_column = st.selectbox("Select X-axis column", numeric_columns, key="x_axis")
                y_column = st.selectbox("Select Y-axis column", numeric_columns, key="y_axis")
                transform_option = st.selectbox("Select transformation", ["None", "Baseline subtraction", "Log transform", "Delta from initial", "Z-score normalization", "I/I₀ normalization", "Min-Max normalization (0–1, sample-wise)"], key="transform_option")
                preview_samples = st.multiselect("Select samples to preview transformation", selected_samples, default=selected_samples, key="preview_samples")
                threshold_value = st.number_input(f"Enter Y-axis threshold value for '{y_column}'", value=1.0, key="threshold_value")

            st.subheader("Data Transformation Preview")
            fig_raw, ax_raw = plt.subplots()
            legend_toggle_raw = st.checkbox("Show raw legend", value=True, key="legend_raw")
            fig_trans, ax_trans = plt.subplots()
            legend_toggle_trans = st.checkbox("Show transformed legend", value=True, key="legend_trans")

            for sample in preview_samples:
                group = df[df[sample_column] == sample].sort_values(x_column)
                x_vals = group[x_column].values
                y_vals = group[y_column].values
                ax_raw.plot(x_vals, y_vals, label=f"{sample} raw")

                if transform_option == "Baseline subtraction":
                    y_vals_trans = y_vals - y_vals[0]
                elif transform_option == "Log transform":
                    y_vals_trans = np.log1p(y_vals)
                elif transform_option == "Delta from initial":
                    y_vals_trans = y_vals - y_vals[0]
                elif transform_option == "Z-score normalization":
                    y_vals_trans = (y_vals - np.mean(y_vals)) / np.std(y_vals) if np.std(y_vals) != 0 else y_vals
                elif transform_option == "I/I₀ normalization":
                    y_vals_trans = y_vals / np.max(y_vals) if np.max(y_vals) != 0 else y_vals
                elif transform_option == "Min-Max normalization (0–1, sample-wise)":
                    y_vals_trans = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals)) if np.max(y_vals) != np.min(y_vals) else y_vals
                else:
                    y_vals_trans = y_vals

                ax_trans.plot(x_vals, y_vals_trans, label=f"{sample} transformed")

            ax_raw.set_title("Before Transformation")
            ax_trans.set_title("After Transformation")
            ax_raw.set_xlabel(x_column)
            ax_trans.set_xlabel(x_column)
            ax_raw.set_ylabel(y_column)
            ax_trans.set_ylabel(y_column)
            if legend_toggle_raw:
                ax_raw.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
            if legend_toggle_trans:
                ax_trans.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
            st.pyplot(fig_raw)
            st.pyplot(fig_trans)

            st.markdown("---")
            with st.sidebar:
                st.markdown("### Model Fitting")
                model_choices = ["Linear", "Sigmoid (Logistic)", "4PL", "5PL", "Gompertz"]
                default_model = st.selectbox("Set default model for all", model_choices, key="default_model_all")
                sample_models = {}
                for sample in selected_samples:
                    sample_models[sample] = st.selectbox(f"Model for {sample}", model_choices, index=model_choices.index(default_model), key=f"model_select_{sample}")
            

            def linear(x, a, b): return a * x + b
            def sigmoid(x, a, b): return 1 / (1 + np.exp(-(x - a) / b))
            def four_pl(x, A, B, C, D): return D + (A - D) / (1 + (x / C)**B)
            def five_pl(x, A, B, C, D, G): return D + (A - D) / ((1 + (x / C)**B)**G)
            def gompertz(x, a, b, c): return a * np.exp(-b * np.exp(-c * x))

            models = {
                "Linear": linear,
                "Sigmoid (Logistic)": sigmoid,
                "4PL": four_pl,
                "5PL": five_pl,
                "Gompertz": gompertz
            }

            fitted_params, tt_results, fitted_data = [], [], []
            x_range = np.linspace(df[x_column].min(), df[x_column].max(), 500)
            fig, ax = plt.subplots(figsize=(8, 5))
            legend_toggle = st.checkbox("Show legend", value=True)

            for sample in selected_samples:
                                    group = df[df[sample_column] == sample].sort_values(x_column)
                x_data = group[x_column].values
                y_data = group[y_column].values

                if transform_option == "Baseline subtraction":
                    y_data = y_data - y_data[0]
                elif transform_option == "Log transform":
                    y_data = np.log1p(y_data)
                elif transform_option == "Delta from initial":
                    y_data = y_data - y_data[0]
                elif transform_option == "Z-score normalization":
                    y_data = (y_data - np.mean(y_data)) / np.std(y_data) if np.std(y_data) != 0 else y_data
                elif transform_option == "I/I₀ normalization":
                    y_data = y_data / np.max(y_data) if np.max(y_data) != 0 else y_data
                elif transform_option == "Min-Max normalization (0–1, sample-wise)":
                    y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data)) if np.max(y_data) != np.min(y_data) else y_data

                model_type = sample_models[sample]
                model_func = models[model_type]

                try:
                    popt, pcov = curve_fit(model_func, x_data, y_data, maxfev=10000)
                    fit_func = lambda x: model_func(x, *popt)
                    y_fit = fit_func(x_data)
                    init_val = y_fit.min()
                    max_val = y_fit.max()
                    lag_time = popt[2] if model_type in ["4PL", "5PL"] else (popt[0] if model_type == "Sigmoid (Logistic)" else np.nan)
                    growth_rate = popt[1] if len(popt) > 1 else np.nan
                    r_squared = 1 - (np.sum((y_data - y_fit)**2) / np.sum((y_data - np.mean(y_data))**2))
                    rmse = np.sqrt(np.mean((y_data - y_fit)**2))
                    fit_y = fit_func(x_range)
                    ci = 1.96 * np.sqrt(np.diag(pcov)) if pcov.size else np.zeros_like(x_range)
                    upper = fit_y + ci[0] if len(ci) > 0 else fit_y
                    lower = fit_y - ci[0] if len(ci) > 0 else fit_y

                    fit_df = pd.DataFrame({
                        x_column: x_range,
                        y_column: fit_y,
                        "Lower CI": lower,
                        "Upper CI": upper,
                        sample_column: sample,
                        "Model": model_type
                    })
                    fitted_data.append(fit_df)

                    fitted_params.append({
                        "Sample": sample,
                        "Model": model_type,
                        "Initial Value": round(init_val, 4),
                        "Lag Time": round(lag_time, 4) if not np.isnan(lag_time) else "N/A",
                        "Growth Rate": round(growth_rate, 4) if not np.isnan(growth_rate) else "N/A",
                        "Max Value": round(max_val, 4),
                        "R²": round(r_squared, 4),
                        "RMSE": round(rmse, 4)
                    })

                    if np.min(y_fit) <= threshold_value <= np.max(y_fit):
                        root = root_scalar(lambda x: fit_func(x) - threshold_value, bracket=[min(x_data), max(x_data)])
                        if root.converged:
                            deriv = (fit_func(root.root + 1e-5) - fit_func(root.root - 1e-5)) / (2e-5)
                            tt_var = (deriv ** -2) * np.sum(np.diag(pcov)) if pcov.size else 0
                            tt_stderr = np.sqrt(tt_var) if tt_var > 0 else np.nan
                            tt_results.append((sample, round(root.root, 4), round(tt_stderr, 4)))
                            ax.scatter(root.root, threshold_value, label=f"{sample} TT", marker='x', zorder=5)
                        else:
                            tt_results.append((sample, "N/A", "N/A"))
                    else:
                        tt_results.append((sample, "N/A", "N/A"))

                    ax.plot(x_data, y_data, 'o', label=f"{sample} data")
                    ax.plot(x_range, fit_y, '-', label=f"{sample} fit")

                except Exception as e:
                    st.warning(f"Error fitting {sample}: {e}")
                    fitted_params.append({"Sample": sample, "Model": model_type, "Initial Value": "Error", "Lag Time": "Error", "Growth Rate": "Error", "Max Value": "Error", "R²": "Error", "RMSE": "Error"})
                    tt_results.append((sample, "Error", "Error"))

            ax.axhline(threshold_value, color='red', linestyle='--', label="Threshold")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title("Fitted Curves and Threshold")

            with st.expander("Raw + Fitted Curve + CI + TT", expanded=False):
    for sample in selected_samples:
        fig_single, ax_single = plt.subplots(figsize=(7, 4))
        group = df[df[sample_column] == sample].sort_values(x_column)
                x_data = group[x_column].values
                y_data = group[y_column].values

                if transform_option == "Baseline subtraction":
                    y_data = y_data - y_data[0]
                elif transform_option == "Log transform":
                    y_data = np.log1p(y_data)
                elif transform_option == "Delta from initial":
                    y_data = y_data - y_data[0]
                elif transform_option == "Z-score normalization":
                    y_data = (y_data - np.mean(y_data)) / np.std(y_data) if np.std(y_data) != 0 else y_data
                elif transform_option == "I/I₀ normalization":
                    y_data = y_data / np.max(y_data) if np.max(y_data) != 0 else y_data
                elif transform_option == "Min-Max normalization (0–1, sample-wise)":
                    y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data)) if np.max(y_data) != np.min(y_data) else y_data

                model_type = sample_models[sample]
                model_func = models[model_type]
                try:
                    popt, pcov = curve_fit(model_func, x_data, y_data, maxfev=10000)
                    fit_func = lambda x: model_func(x, *popt)
                    y_fit = fit_func(x_range)
                    ci = 1.96 * np.sqrt(np.diag(pcov)) if pcov.size else np.zeros_like(x_range)
                    upper = y_fit + ci[0] if len(ci) > 0 else y_fit
                    lower = y_fit - ci[0] if len(ci) > 0 else y_fit
                    ax_single.plot(x_data, y_data, 'o', label='Raw Data')
                    ax_single.plot(x_range, y_fit, '-', color='green', label='Fitted Curve')
                    ax_single.fill_between(x_range, lower, upper, color='green', alpha=0.2, label='95% CI')

                    root = root_scalar(lambda x: fit_func(x) - threshold_value, bracket=[min(x_data), max(x_data)])
                    if root.converged:
                        tt_val = round(root.root, 4)
                        deriv = (fit_func(root.root + 1e-5) - fit_func(root.root - 1e-5)) / (2e-5)
                        tt_var = (deriv ** -2) * np.sum(np.diag(pcov)) if pcov.size else 0
                        tt_stderr = round(np.sqrt(tt_var), 4) if tt_var > 0 else 'N/A'
                        ax_single.axvline(tt_val, linestyle='--', color='red', label=f'TT = {tt_val} ± {tt_stderr}')
                    ax_single.axhline(threshold_value, color='gray', linestyle='--', label='Threshold')

                except:
                    ax_single.plot(x_data, y_data, 'o', label='Raw Data')
                    ax_single.text(0.5, 0.5, 'Fit Error', ha='center')

                ax_single.set_title(f"{sample}: Raw + Fit + CI + TT")
                ax_single.set_xlabel(x_column)
                ax_single.set_ylabel(y_column)
                ax_single.legend()
                st.pyplot(fig_single)
            if legend_toggle:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    borderaxespad=0.,
                    fontsize='small',
                    ncol=1,
                    fancybox=True,
                    shadow=True,
                    title='Legend',
                    handlelength=1.5
                )

        # Report management
            if st.button("Add Plot to Report"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                name = f"Fitted Plot {len(st.session_state.report_plots) + 1}"
                st.session_state.report_plots.append((name, buf.read()))
                st.session_state.report_elements[name] = True

            param_df = pd.DataFrame(fitted_params)
            st.subheader("Fitted Parameters")
            st.dataframe(param_df)
            if st.button("Add Fitted Parameters to Report"):
                name = f"Fitted Parameters {len(st.session_state.report_tables) + 1}"
                st.session_state.report_tables.append((name, param_df.copy()))
                st.session_state.report_elements[name] = True

            tt_df = pd.DataFrame(tt_results, columns=["Sample", "Time to Threshold", "Std Error"])
            st.subheader("Estimated Time to Threshold")
            st.dataframe(tt_df)
            if st.button("Add Time to Threshold to Report"):
                name = f"Time to Threshold {len(st.session_state.report_tables) + 1}"
                st.session_state.report_tables.append((name, tt_df.copy()))
                st.session_state.report_elements[name] = True

            if fitted_data:
                fitdata_df = pd.concat(fitted_data, ignore_index=True)
                fitdata_df['Transformation'] = transform_option
                st.subheader("Fitted Curve Data")
                st.dataframe(fitdata_df)
                if st.button("Add Fitted Curve Data to Report"):
                    name = f"Fitted Curve Data {len(st.session_state.report_tables) + 1}"
                    st.session_state.report_tables.append((name, fitdata_df.copy()))
                    st.session_state.report_elements[name] = True

            st.subheader("Select elements to include in report")
            for key in st.session_state.report_elements:
                st.session_state.report_elements[key] = st.checkbox(f"Include: {key}", value=st.session_state.report_elements[key])

            st.subheader("Export Report")
            if st.button("📥 Export All Results to Excel"):
                report_buf = io.BytesIO()
                with pd.ExcelWriter(report_buf, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="Input Data", index=False)
                    summary_rows = []
                    for name, table in st.session_state.report_tables:
                        if st.session_state.report_elements.get(name):
                            safe_name = name[:31]
                            table.to_excel(writer, sheet_name=safe_name, index=False)
                            summary_rows.append({"Included Table": safe_name})
                    for name, plot_bytes in st.session_state.report_plots:
                        if st.session_state.report_elements.get(name):
                            worksheet = writer.book.add_worksheet(name[:31])
                            image_stream = io.BytesIO(plot_bytes)
                            worksheet.insert_image("B2", f"{name}.png", {"image_data": image_stream})
                            summary_rows.append({"Included Plot": name[:31]})
                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                report_buf.seek(0)
                st.download_button(
                    label="Download Excel Report",
                    data=report_buf,
                    file_name="Analysis_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # end of try block
    except Exception as e:
        st.error(f"An error occurred: {e}")
