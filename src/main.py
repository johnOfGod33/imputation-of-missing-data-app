import numpy as np
import pandas as pd
import streamlit as st

from models.comparison_engine import ComparisonEngine
from models.data_analyzer import DataAnalyzer
from models.imputation_engine import ImputationEngine
from models.missing_detector import MissingDetector

st.set_page_config(page_title="Imputation Manager", layout="wide")


def main():
    st.title("🔧 Gestion d'Imputation des Valeurs Manquantes")

    if "df_original" not in st.session_state:
        st.session_state.df_original = None

    # Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_original = df

        # Analyse exploratoire
        st.header("📊 Analyse Exploratoire")
        analyzer = DataAnalyzer(df)
        analyzer.display_summary()

        # Configuration des valeurs manquantes
        st.header("🔧 Configuration des Valeurs Manquantes")
        detector = MissingDetector(df)
        missing_config = detector.configure_missing_values()

        if missing_config:
            # Préparation des données
            df_processed = detector.apply_missing_detection(missing_config)

            # Sélection des méthodes d'imputation
            st.header("🔄 Méthodes d'Imputation")
            imputer = ImputationEngine(df_processed)
            methods = imputer.select_methods()

            if methods:
                # Exécution des imputations
                results = imputer.execute_imputation(methods)

                # Comparaison
                st.header("📈 Comparaison des Résultats")
                comparator = ComparisonEngine(df_processed, results)
                comparator.display_comparison()

                # Export
                st.header("💾 Export")
                comparator.export_results()


if __name__ == "__main__":
    main()
