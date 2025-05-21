import numpy as np
from scipy.optimize import curve_fit, root_scalar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import xlsxwriter
import datetime

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
        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.dataframe(df.head())

        sample_column = st.selectbox("Select the column that contains sample identifiers", df.columns)
        unique_samples = df[sample_column].dropna().unique()
        selected_samples = st.multiselect("Filter by sample (optional)", unique_samples, default=list(unique_samples))

        if selected_samples:
            df = df[df[sample_column].isin(selected_samples)]

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) >= 2:
            x_column = st.selectbox("Select X-axis column", numeric_columns)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)

            threshold_value = st.number_input(f"Enter Y-axis threshold value for '{y_column}'", value=1.0)

            model_choices = ["Linear", "Sigmoid (Logistic)", "4PL", "5PL", "Gompertz"]
            sample_models = {}
            for sample in selected_samples:
                model = st.selectbox(f"Model for {sample}", model_choices, key=f"model_{sample}")
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
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
                fitdata_df = pd.concat(fitted_data, ignore_index=True)
                st.subheader("Fitted Curve Data")
                st.dataframe(fitdata_df)
                if st.button("Add Fitted Data to Report"):
                    name = f"Fitted Curve Data {len([n for n, _ in st.session_state.report_tables if 'Fitted Curve Data' in n]) + 1}"
                    st.session_state.report_tables.append((name, fitdata_df.copy()))
                    st.session_state.report_elements[name] = True

        else:
            st.warning("Not enough numeric columns available for plotting.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.subheader("Select elements to include in report")
for key in st.session_state.report_elements:
    st.session_state.report_elements[key] = st.checkbox(f"Include: {key}", value=st.session_state.report_elements[key])

st.subheader("Generate Report")
if st.button("🔄 Reset Session State"):
    st.session_state["trigger_reset"] = True

if st.session_state.get("trigger_reset", False):
    for key in ["report_plots", "report_tables", "report_index", "report_elements", "trigger_reset"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

if st.button("Download Report as Excel"):
    report_buf = io.BytesIO()
    report_title = f"Fitting Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    with pd.ExcelWriter(report_buf, engine="xlsxwriter") as writer:
        workbook = writer.book
        included_tables = [k for k in st.session_state.report_elements if st.session_state.report_elements[k] and (k.startswith("Fitted") or k.startswith("Time to Threshold"))]
        included_plots = [k for k in st.session_state.report_elements if st.session_state.report_elements[k] and k.startswith("Plot")]
        max_len = max(len(included_tables), len(included_plots))
        included_tables.extend([None] * (max_len - len(included_tables)))
        included_plots.extend([None] * (max_len - len(included_plots)))
        summary_df = pd.DataFrame({"Included Tables": included_tables, "Included Plots": included_plots})
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for sheet_name, df in st.session_state.report_tables:
            if st.session_state.report_elements.get(sheet_name):
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)

        for plot_item in st.session_state.report_plots:
            if isinstance(plot_item, tuple) and len(plot_item) == 2:
                name, plot_data = plot_item
                if st.session_state.report_elements.get(name):
                    worksheet = workbook.add_worksheet(name[:31])
                    image_stream = io.BytesIO(plot_data)
                    worksheet.insert_image("B2", f"{name}.png", {'image_data': image_stream})
        report_buf.seek(0)
    st.download_button(
        label="📥 Download Excel Report",
        data=report_buf,
        file_name=f"{report_title}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
