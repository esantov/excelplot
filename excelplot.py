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
