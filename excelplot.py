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
if "report_index" not in st.session_state:
    st.session_state["report_index"] = 0
if "report_elements" not in st.session_state:
    st.session_state["report_elements"] = {}

if uploaded_file is not None:
    try:
                y_fit_vals = fit_func(x_data)
                if np.min(y_fit_vals) <= threshold_value <= np.max(y_fit_vals):
                    root = root_scalar(lambda x: fit_func(x) - threshold_value, bracket=[min(x_data), max(x_data)])
                    if root.converged:
                        deriv = (fit_func(root.root + 1e-5) - fit_func(root.root - 1e-5)) / (2e-5)
                        tt_var = (deriv ** -2) * np.dot(np.dot(np.gradient(fit_func(x_data)), pcov), np.gradient(fit_func(x_data))) / len(x_data)
                        tt_stderr = np.sqrt(tt_var) if tt_var > 0 else np.nan
                        tt_results.append((sample, round(root.root, 4), round(tt_stderr, 4)))
                        ax.scatter(root.root, threshold_value, label=f"{sample} TT", marker='x', zorder=5)
                    else:
                        tt_results.append((sample, "N/A", "N/A"))
                else:
                    tt_results.append((sample, "N/A", "N/A")) - threshold_value, bracket=[min(x_data), max(x_data)])
                        # Estimate std error using delta method
                        deriv = (fit_func(root.root + 1e-5) - fit_func(root.root - 1e-5)) / (2e-5)
                        tt_var = (deriv ** -2) * np.dot(np.dot(np.gradient(fit_func(x_data)), pcov), np.gradient(fit_func(x_data))) / len(x_data)
                        tt_stderr = np.sqrt(tt_var) if tt_var > 0 else np.nan
                        if root.converged:
                            tt_results.append((sample, round(root.root, 4), round(tt_stderr, 4)))
                            ax.scatter(root.root, threshold_value, label=f"{sample} TT", marker='x', zorder=5)
                                            else:
                            tt_results.append((sample, "N/A", "N/A"))
                    except:
                        tt_results.append((sample, "N/A", "N/A"))

                    ax.plot(x_data, y_data, 'o', label=f"{sample} data")
                    ax.plot(x_range, fit_func(x_range), '-', label=f"{sample} fit")

                except Exception as e:
                    st.warning(f"Error fitting {sample}: {e}")
                    fitted_params.append({"Sample": sample, "Model": model_type, "Initial Value": "Error", "Lag Time": "Error", "Growth Rate": "Error", "Max Value": "Error"})
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
            # Mark best model per sample (based on highest R²)
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

# Report selector UI
st.subheader("Select elements to include in report")
for key in st.session_state.report_elements:
    st.session_state.report_elements[key] = st.checkbox(f"Include: {key}", value=st.session_state.report_elements[key])

# Button to export report
import datetime

st.subheader("Generate Report")
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
        summary_df = pd.DataFrame({
            "Included Tables": included_tables,
            "Included Plots": included_plots
        })
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
