import numpy as np
from scipy.optimize import curve_fit, root_scalar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import xlsxwriter

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Initialize session state
if "report_plots" not in st.session_state:
    st.session_state["report_plots"] = []
if "report_tables" not in st.session_state:
    st.session_state["report_tables"] = []
if "analyzed_samples" not in st.session_state:
    st.session_state["analyzed_samples"] = []

# Model functions
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

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success("File uploaded successfully!")

        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.dataframe(df.head())

        sample_column = st.selectbox("Select the column that contains sample identifiers", df.columns)
        all_samples = df[sample_column].dropna().unique()
        remaining_samples = [s for s in all_samples if s not in st.session_state.analyzed_samples]

        if remaining_samples:
            selected_samples = st.multiselect("Select samples to analyze", remaining_samples)

            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            x_column = st.selectbox("Select X-axis column", numeric_columns)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)
            threshold_value = st.number_input(f"Enter Y-axis threshold value for '{y_column}'", value=1.0)

            sample_models = {}
            st.subheader("Choose a fitting model for each sample")
            for sample in selected_samples:
                model = st.selectbox(f"Model for {sample}", list(models.keys()), key=f"model_{sample}")
                sample_models[sample] = model

            if st.button("Run Analysis"):
                fitted_params = []
                tt_results = []
                x_range = np.linspace(df[x_column].min(), df[x_column].max(), 500)
                fig, ax = plt.subplots()

                for sample in selected_samples:
                    group = df[df[sample_column] == sample].sort_values(x_column)
                    x_data = group[x_column].values
                    y_data = group[y_column].values
                    model_name = sample_models[sample]
                    model_func = models[model_name]

                    try:
                        popt, _ = curve_fit(model_func, x_data, y_data, maxfev=10000)
                        fit_func = lambda x: model_func(x, *popt)

                        # Parameter logic per model
                        init_val = fit_func(min(x_data))
                        max_val = fit_func(max(x_data))
                        lag_time = np.nan
                        growth_rate = np.nan
                        if model_name == "Linear":
                            growth_rate = popt[0]
                        elif model_name == "Sigmoid (Logistic)":
                            lag_time = popt[0]
                            growth_rate = 1 / popt[1]
                        elif model_name in ["4PL", "5PL"]:
                            lag_time = popt[2]
                            growth_rate = popt[1]
                        elif model_name == "Gompertz":
                            lag_time = np.log(popt[1]) / popt[2]
                            growth_rate = popt[2]

                        fitted_params.append({
                            "Sample": sample,
                            "Model": model_name,
                            "Initial Value": round(init_val, 4),
                            "Lag Time": round(lag_time, 4) if not np.isnan(lag_time) else "N/A",
                            "Growth Rate": round(growth_rate, 4) if not np.isnan(growth_rate) else "N/A",
                            "Max Value": round(max_val, 4)
                        })

                        try:
                            root = root_scalar(lambda x: fit_func(x) - threshold_value, bracket=[min(x_data), max(x_data)])
                            if root.converged:
                                tt_results.append((sample, root.root))
                                ax.scatter(root.root, threshold_value, label=f"{sample} TT", marker='x', zorder=5)
                        except:
                            tt_results.append((sample, "N/A"))

                        ax.plot(x_data, y_data, 'o', label=f"{sample} data")
                        ax.plot(x_range, fit_func(x_range), '-', label=f"{sample} fit")

                    except Exception as e:
                        st.warning(f"Error fitting {sample}: {e}")
                        fitted_params.append({
                            "Sample": sample,
                            "Model": model_name,
                            "Initial Value": "Error",
                            "Lag Time": "Error",
                            "Growth Rate": "Error",
                            "Max Value": "Error"
                        })
                        tt_results.append((sample, "Error"))

                ax.axhline(threshold_value, color='red', linestyle='--', label="Threshold")
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title("Data, Fitted Curves, and Time to Threshold")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)

                if st.button("Add Plot to Report"):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.session_state.report_plots.append(buf.read())
                    st.success("Plot added to report")

                param_df = pd.DataFrame(fitted_params)
                st.subheader("Fitted Parameters")
                st.dataframe(param_df)
                if st.button("Add Table to Report"):
                    st.session_state.report_tables.append(("Fitted Parameters", param_df.copy()))
                    st.success("Table added to report")

                tt_df = pd.DataFrame(tt_results, columns=["Sample", "Time to Threshold"])
                st.subheader("Estimated Time to Threshold")
                st.dataframe(tt_df)
                if st.button("Add Time to Threshold Table to Report"):
                    st.session_state.report_tables.append(("Time to Threshold", tt_df.copy()))
                    st.success("Table added to report")

                st.session_state.analyzed_samples.extend(selected_samples)

                # Ask to analyze more
                if len([s for s in all_samples if s not in st.session_state.analyzed_samples]) > 0:
                    if st.button("Analyze more samples"):
                        st.experimental_rerun()
        else:
            st.info("🎉 All samples have been analyzed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Final Report Download
st.subheader("Generate Report")
if st.button("Download Report as Excel"):
    report_buf = io.BytesIO()
    with pd.ExcelWriter(report_buf, engine="xlsxwriter") as writer:
        workbook = writer.book
        for sheet_name, df in st.session_state.report_tables:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        for i, plot_data in enumerate(st.session_state.report_plots):
            image = Image.open(io.BytesIO(plot_data))
            image_path = f"plot_{i}.png"
            image.save(image_path)
            worksheet = workbook.add_worksheet(f"Plot_{i+1}")
            worksheet.insert_image("B2", image_path)

    report_buf.seek(0)
    st.download_button(
        label="📥 Download Excel Report",
        data=report_buf,
        file_name="fitting_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
