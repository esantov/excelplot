
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Excel Data Analyzer and Plotter")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

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
            selected_samples = st.multiselect("Filter by sample (optional)", unique_samples)

            if selected_samples:
                df = df[df[sample_column].isin(selected_samples)]

        # Select X and Y axis columns (only numeric columns)
        st.subheader("Select Data Columns for Plotting")
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_columns) >= 2:
            x_column = st.selectbox("Select X-axis column", numeric_columns)
            y_column = st.selectbox("Select Y-axis column", numeric_columns)

            if x_column and y_column:
                st.write(f"Plotting `{y_column}` vs `{x_column}`")

                # Plotting
                fig, ax = plt.subplots()

                # Plot each sample group as its own line
                for sample_name, group in df.groupby(sample_column):
                    group_sorted = group.sort_values(by=x_column)
                    ax.plot(group_sorted[x_column], group_sorted[y_column], marker='o', linestyle='-', label=str(sample_name))

                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_title(f"{y_column} vs {x_column} by {sample_column}")
                ax.legend(title=sample_column, bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns available for plotting.")

    except ImportError:
        st.error("Missing dependency: openpyxl. Add it to requirements.txt.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
