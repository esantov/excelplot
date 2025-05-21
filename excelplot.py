import streamlit as st
import pandas as pd
import os
os.system("pip install openpyxl")

st.title("Excel Data Analyzer")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success("File uploaded successfully!")
        
        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        
        st.subheader(f"Preview of '{selected_sheet}'")
        st.dataframe(df.head())

        st.write("Data shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
    except ImportError as e:
        st.error("Missing dependency: openpyxl. Add it to requirements.txt.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Step 1: Choose the sample-identifying column
sample_column = st.selectbox("Select the column that contains sample identifiers", df.columns)

if sample_column:
    st.write(f"You selected: `{sample_column}` to identify samples.")
    st.dataframe(df[[sample_column]].drop_duplicates().head())

# Step 2: Choose columns for X and Y axes
st.subheader("Select Data Columns for Plotting")

# Filter numeric columns for plotting options
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

x_column = st.selectbox("Select X-axis column", numeric_columns)
y_column = st.selectbox("Select Y-axis column", numeric_columns)

# Show selected columns
if x_column and y_column:
    st.write(f"Plotting `{y_column}` vs `{x_column}`")
