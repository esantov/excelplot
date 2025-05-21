import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import xlsxwriter

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Initialize session state for report
if "report_plots" not in st.session_state:
    st.session_state["report_plots"] = []
if "report_tables" not in st.session_state:
    st.session_state["report_tables"] = []

if uploaded_file is not None:
    try:
        # Load Excel file and get sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success("File uploaded successfully!")

        # Select sheet
        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.subheader(f"Preview of '{selected_sheet}'")
        st.dataframe(df.head())

        # Select sample-identifying column
        sample_column = st.selectbox("Select the column that contains sample identifiers", df.columns)

        # Optional filtering by sample
        if sample_column:
            unique_samples = df[sample_column].dropna().unique()
            selected_samples = st.multiselect("Filter by sample (optional)", unique_samples, default=list(unique_samples))

            if selected_samples:
                df = df[df[sample_column].isin(selected_samples)]

        # Select X and Y axis columns (only numeric columns)
        st.subheader("Select Data Columns for Plotting")
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_columns) >= 2:
            x_column = st.selectbox("Select X-axis column", numeric_columns)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)

            st.write(f"Plotting `{y_column}` vs `{x_column}`")

            # User input for threshold
            st.subheader("Threshold Analysis")
            threshold_value = st.number_input(f"Enter Y-axis threshold value for '{y_column}'", value=1.0)

            # Choose model to fit
            model_type = st.selectbox("Choose a model to fit", ["Linear", "Sigmoid (Logistic)", "4PL", "5PL", "Gompertz"])

            # Define model functions
            def linear(x, a, b):
                return a * x + b

            def sigmoid(x, a, b):
                return 1 / (1 + np.exp(-(x - a) / b))

            def four_pl(x, A, B, C, D):
                return D + (A - D) / (1 + (x / C)**B)

            def five_pl(x, A, B, C, D, G):
                return D + (A - D) / ((1 + (x / C)**B)**G)

            def gompertz(x, a, b, c):
                return a * np.exp(-b * np.exp(-c * x))

            fitted_params = []
            tt_results = []
            x_range = np.linspace(df[x_column].min(), df[x_column].max(), 500)

            fig, ax = plt.subplots()

            for sample in selected_samples:
                group = df[df[sample_column] == sample].sort_values(x_column)
                x_data = group[x_column].values
                y_data = group[y_column].values

                try:
                    if model_type == "Linear":
                        popt, pcov = curve_fit(linear, x_data, y_data)
                        fit_func = lambda x: linear(x, *popt)
                        init_val = linear(min(x_data), *popt)
                        max_val = linear(max(x_data), *popt)
                        lag_time = np.nan
                        growth_rate = popt[0]
                    elif model_type == "Sigmoid (Logistic)":
                        popt, pcov = curve_fit(sigmoid, x_data, y_data, maxfev=10000)
                        fit_func = lambda x: sigmoid(x, *popt)
                        init_val = sigmoid(min(x_data), *popt)
                        max_val = sigmoid(max(x_data), *popt)
                        lag_time = popt[0]
                        growth_rate = 1 / popt[1]
                    elif model_type == "4PL":
                        popt, pcov = curve_fit(four_pl, x_data, y_data, maxfev=10000)
                        fit_func = lambda x: four_pl(x, *popt)
                        init_val = four_pl(min(x_data), *popt)
                        max_val = popt[0]
                        lag_time = popt[2]
                        growth_rate = popt[1]
                    elif model_type == "5PL":
                        popt, pcov = curve_fit(five_pl, x_data, y_data, maxfev=10000)
                        fit_func = lambda x: five_pl(x, *popt)
                        init_val = five_pl(min(x_data), *popt)
                        max_val = popt[0]
                        lag_time = popt[2]
                        growth_rate = popt[1]
                    elif model_type == "Gompertz":
                        popt, pcov = curve_fit(gompertz, x_data, y_data, maxfev=10000)
                        fit_func = lambda x: gompertz(x, *popt)
                        init_val = gompertz(min(x_data), *popt)
                        max_val = popt[0]
                        lag_time = np.log(popt[1]) / popt[2]
                        growth_rate = popt[2]

                    fitted_params.append({
                        "Sample": sample,
                        "Initial Value": round(init_val, 4),
                        "Lag Time": round(lag_time, 4) if not np.isnan(lag_time) else "N/A",
                        "Growth Rate": round(growth_rate, 4),
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
                    fitted_params.append({
                        "Sample": sample,
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

            st.subheader("Fitted Parameters")
            param_df = pd.DataFrame(fitted_params)
            st.dataframe(param_df)
            if st.button("Add Table to Report"):
                st.session_state.report_tables.append(("Fitted Parameters", param_df.copy()))
                st.success("Table added to report")

            st.subheader("Estimated Time to Threshold")
            tt_df = pd.DataFrame(tt_results, columns=["Sample", "Time to Threshold"])
            st.dataframe(tt_df)
            if st.button("Add Time to Threshold Table to Report"):
                st.session_state.report_tables.append(("Time to Threshold", tt_df.copy()))
                st.success("Table added to report")

        else:
            st.warning("Not enough numeric columns available for plotting.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Button to export report
st.subheader("Generate Report")
if st.button("Download Report as Excel"):
    report_buf = io.BytesIO()
    with pd.ExcelWriter(report_buf, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Write all tables
        for sheet_name, df in st.session_state.report_tables:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        # Add plots as images
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
