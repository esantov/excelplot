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
        # Your main app logic goes here
        pass  # Replace this with your actual logic
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
