import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_data(uploaded_file: UploadedFile) -> pd.DataFrame | None:
    """Charge les données selon le format du fichier"""
    file_extension = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_extension in ["csv"]:
            return pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            return pd.read_excel(uploaded_file)
        elif file_extension in ["json"]:
            return pd.read_json(uploaded_file)
        else:
            # Essayer de lire comme CSV par défaut
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return None
