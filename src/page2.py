import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.title(" Compare Imputation Methods")

# File upload section
st.header("📁 Upload Multiple CSV Files")

# Allow multiple file uploads
uploaded_files = st.file_uploader(
    "Upload your CSV files (select multiple files)",
    type=["csv"],
    accept_multiple_files=True,
)

if len(uploaded_files) >= 2:
    pass
else:
    st.info("👆 Please upload at least 2 CSV files to start the comparison analysis.")
