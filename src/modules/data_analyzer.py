import numpy as np
import pandas as pd
import streamlit as st
from utils.visualizer import Visualizer


class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.visualizer = Visualizer()

    def display_summary(self):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Lignes", self.df.shape[0])
        with col2:
            st.metric("Colonnes", self.df.shape[1])
        with col3:
            st.metric("Mémoire", f"{self.df.memory_usage().sum() / 1024:.1f} KB")
        with col4:
            missing_pct = (
                self.df.isnull().sum().sum()
                / (self.df.shape[0] * self.df.shape[1])
                * 100
            )
            st.metric("Valeurs manquantes", f"{missing_pct:.1f}%")

        # Aperçu des données
        st.subheader("Aperçu des données")
        st.dataframe(self.df.head())

        # Analyse par colonne
        st.subheader("Analyse par colonne")

        tabs = st.tabs(["Numériques", "Catégorielles", "Valeurs manquantes"])

        with tabs[0]:
            self._analyze_numeric_columns()

        with tabs[1]:
            self._analyze_categorical_columns()

        with tabs[2]:
            self._analyze_missing_values()

    def _analyze_numeric_columns(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Sélectionner une colonne", numeric_cols)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Statistiques descriptives**")
                st.dataframe(self.df[selected_col].describe())

            with col2:
                st.write("**Distribution**")
                self.visualizer.plot_distribution(self.df[selected_col])
        else:
            st.info("Aucune colonne numérique détectée")

    def _analyze_categorical_columns(self):
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        if len(categorical_cols) > 0:
            selected_col = st.selectbox(
                "Sélectionner une colonne", categorical_cols, key="cat_select"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Valeurs uniques**")
                st.metric("Nombre", self.df[selected_col].nunique())

                top_values = self.df[selected_col].value_counts().head(10)
                st.dataframe(top_values)

            with col2:
                st.write("**Distribution**")
                self.visualizer.plot_categorical_distribution(self.df[selected_col])
        else:
            st.info("Aucune colonne catégorielle détectée")

    def _analyze_missing_values(self):
        missing_info = self.df.isnull().sum()
        missing_info = missing_info[missing_info > 0].sort_values(ascending=False)

        if len(missing_info) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Valeurs manquantes par colonne**")
                missing_df = pd.DataFrame(
                    {
                        "Colonne": missing_info.index,
                        "Nombre": missing_info.values,
                        "Pourcentage": (missing_info.values / len(self.df) * 100).round(
                            2
                        ),
                    }
                )
                st.dataframe(missing_df)

            with col2:
                st.write("**Heatmap des valeurs manquantes**")
                self.visualizer.plot_missing_heatmap(self.df)
        else:
            st.success("Aucune valeur manquante détectée (NaN)")

    def get_column_info(self):
        return {
            "numeric": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical": list(self.df.select_dtypes(include=["object"]).columns),
            "missing": list(self.df.columns[self.df.isnull().any()]),
        }
