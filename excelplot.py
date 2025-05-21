import streamlit as st
import pandas as pd

st.title("Excel Data Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load Excel file and list sheets
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    st.success("File uploaded successfully!")
    selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)

    # Read and display selected sheet
    df = pd.read_excel(xls, sheet_name=selected_sheet)
    st.subheader(f"Preview of '{selected_sheet}'")
    st.dataframe(df.head())

    # Show basic info
    st.write("Data shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
