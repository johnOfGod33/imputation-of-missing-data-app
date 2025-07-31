import numpy as np
import pandas as pd
import streamlit as st
from models.comparison_engine import ComparisonEngine
from models.data_analyzer import DataAnalyzer
from models.imputation_engine import ImputationEngine
from models.missing_detector import MissingDetector
from utils import load_data

st.set_page_config(page_title="Imputation Manager", layout="wide")


def main():
    st.title("🔧 Gestion d'Imputation des Valeurs Manquantes")

    if "df_original" not in st.session_state:
        st.session_state.df_original = None

    # Upload avec support multi-format
    uploaded_file = st.file_uploader(
        "Upload fichier de données",
        type=["csv", "xlsx", "xls", "json"],
    )

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.session_state.df_original = df

            # Analyse exploratoire AVEC la colonne target
            st.header("📊 Analyse Exploratoire")
            analyzer = DataAnalyzer(df)  # Utilise df complet avec target
            analyzer.display_summary()

            # Exclure la colonne target seulement pour le traitement
            # Sélection de la colonne target
            st.header("🎯 Sélection de la colonne target")
            st.write(
                "Sélectionnez la colonne target (variable à prédire) si elle existe dans vos données :"
            )

            target_col = st.selectbox(
                "Colonne target (optionnel)",
                options=["Aucune"] + list(df.columns),
                help="Cette colonne sera exclue du traitement d'imputation",
            )
            if target_col != "Aucune":
                df_features = df.drop(columns=[target_col])
                st.info(f"ℹ️ Colonne '{target_col}' exclue du traitement d'imputation")
            else:
                df_features = df.copy()
            # Configuration des valeurs manquantes et outliers SANS la colonne target
            st.header("🔧 Configuration des Valeurs Manquantes et Outliers")

            detector = MissingDetector(df_features)
            missing_config = detector.configure_missing_values()

            if missing_config:
                # Préparation des données
                df_processed = detector.apply_missing_detection(missing_config)

                # Vérification finale avant imputation
                if df_processed.isnull().sum().sum() > 0:
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
                else:
                    st.success(
                        "✅ Aucune valeur manquante à imputer après le traitement des outliers"
                    )


if __name__ == "__main__":
    main()
