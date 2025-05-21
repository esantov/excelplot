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

            # === Transformation Options ===
            st.subheader("Data Transformation")
                        with st.sidebar:
                                transform_option = st.selectbox("Select transformation", ["None", "Baseline subtraction", "Log transform", "Delta from initial", "Z-score normalization", "I/I₀ normalization", "Min-Max normalization (0–1, sample-wise)"], key="transform_option")

                                            preview_samples = st.multiselect("Select samples to preview transformation", selected_samples, default=selected_samples, key="preview_samples")

            fig_raw, ax_raw = plt.subplots()
            fig_trans, ax_trans = plt.subplots()

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
                elif transform_option == "Min-Max normalization (0–1, shared)":
                    global_min = df[y_column].min()
                    global_max = df[y_column].max()
                    y_vals_trans = (y_vals - global_min) / (global_max - global_min) if global_max != global_min else y_vals
                else:
                    y_vals_trans = y_vals

                ax_trans.plot(x_vals, y_vals_trans, label=f"{sample} transformed")

            ax_raw.set_title("Before Transformation")
            ax_trans.set_title("After Transformation")
            ax_raw.set_xlabel(x_column)
            ax_trans.set_xlabel(x_column)
            ax_raw.set_ylabel(y_column)
            ax_trans.set_ylabel(y_column)
            ax_raw.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_trans.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_raw)
            st.pyplot(fig_trans)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)

                        with st.sidebar:
                                threshold_value = st.number_input(f"Enter Y-axis threshold value for '{y_column}'", value=1.0, key="threshold_value")

            model_choices = ["Linear", "Sigmoid (Logistic)", "4PL", "5PL", "Gompertz"]
            with st.expander("Model selection per sample", expanded=False):
                default_model = st.selectbox("Set default model for all", model_choices, key="default_model_all")
                            cols = st.columns(3)
                sample_models = {}
                for idx, sample in enumerate(selected_samples):
                    with cols[idx % 3]:
                        model = st.selectbox(f"{sample}", model_choices, index=model_choices.index(default_model), key=f"model_select_{sample}")
                        sample_models[sample] = model

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
            elif transform_option == "Min-Max normalization (0–1, shared)":
                global_min = df[y_column].min()
                global_max = df[y_column].max()
                y_data = (y_data - global_min) / (global_max - global_min) if global_max != global_min else y_data

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
                            tt_var = (deriv ** -2) * np.dot(np.dot(np.gradient(y_fit), pcov), np.gradient(y_fit)) / len(x_data)
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
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', ncol=1, fancybox=True, shadow=True, title='Legend')
            st.pyplot(fig)

            if st.button("Add Plot to Report"):
                idx = len(st.session_state.report_plots)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                plot_key = f"Plot_{idx+1}"
                st.session_state.report_plots.append((plot_key, buf.read()))
                st.session_state.report_elements[plot_key] = True

            param_df = pd.DataFrame(fitted_params)
            best_models = param_df.loc[param_df.groupby("Sample")['R²'].idxmax()]
            param_df["Best Model"] = param_df.apply(lambda row: "✅" if ((best_models['Sample'] == row['Sample']) & (best_models['Model'] == row['Model'])).any() else "", axis=1)
            st.subheader("Fitted Parameters")
            st.dataframe(param_df)
            if st.button("Add Table to Report"):
                name = f"Fitted Parameters {len([n for n, _ in st.session_state.report_tables if 'Fitted Parameters' in n]) + 1}"
                st.session_state.report_tables.append((name, param_df.copy()))
                st.session_state.report_elements[name] = True

            tt_df = pd.DataFrame(tt_results, columns=["Sample", "Time to Threshold", "Std Error"])
            st.subheader("Estimated Time to Threshold")
            st.dataframe(tt_df)
            if st.button("Add Time to Threshold Table to Report"):
                name = f"Time to Threshold {len([n for n, _ in st.session_state.report_tables if 'Time to Threshold' in n]) + 1}"
                st.session_state.report_tables.append((name, tt_df.copy()))
                st.session_state.report_elements[name] = True

            if fitted_data:
                raw_fitdata_df = pd.concat(fitted_data, ignore_index=True)
                raw_fitdata_df['Transformation'] = transform_option
                st.subheader("Fitted Curve Data")
                st.dataframe(raw_fitdata_df)
                if st.button("Add Raw + Fitted Data to Report"):
                    name = f"Raw + Fitted Data {len([n for n, _ in st.session_state.report_tables if 'Raw + Fitted Data' in n]) + 1}"
                    st.session_state.report_tables.append((name, raw_fitdata_df.copy()))
                    st.session_state.report_elements[name] = True


                st.subheader("Fitted Curve Data")
                st.dataframe(fitdata_df)
                if st.button("Add Fitted Data to Report"):
                    name = f"Transformed + Fitted Data {len([n for n, _ in st.session_state.report_tables if 'Fitted Curve Data' in n]) + 1}"
                    st.session_state.report_tables.append((name, fitdata_df.copy()))
                    st.session_state.report_elements[name] = True
                    st.session_state.report_tables.append((name, fitdata_df.copy()))
                    st.session_state.report_elements[name] = True

        else:
            st.warning("Not enough numeric columns available for plotting.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.subheader("Select elements to include in report")
for key in st.session_state.report_elements:
    st.session_state.report_elements[key] = st.checkbox(f"Include: {key}", value=st.session_state.report_elements[key])

st.subheader("Transformation Descriptions")
with st.expander("📘 View transformation method definitions"):
    st.markdown("""
    **Transformation Types**
    
    - **None**: No transformation applied.
    - **Baseline subtraction**: Subtracts the first y-value of each sample (y = y - y₀).
    - **Log transform**: Applies a natural log transformation (log(1 + y)).
    - **Delta from initial**: Computes the difference from the first value (y = y - y₀).
    - **Z-score normalization**: Scales each sample to mean 0 and standard deviation 1.
    - **I/I₀ normalization**: Scales each sample to its maximum (y = y / max(y)).
    - **Min-Max normalization (0–1, sample-wise)**: Scales each sample between 0 and 1 using its own minimum and maximum values.
