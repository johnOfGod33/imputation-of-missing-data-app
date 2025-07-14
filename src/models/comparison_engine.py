import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error

from .visualizer import Visualizer


class ComparisonEngine:
    def __init__(self, original_df, imputed_results):
        self.original_df = original_df
        self.imputed_results = imputed_results
        self.visualizer = Visualizer()

    def display_comparison(self):
        if not self.imputed_results:
            st.warning("Aucun résultat d'imputation à comparer")
            return

        # Métriques globales
        st.subheader("Métriques de comparaison")
        self._display_global_metrics()

        # Comparaison par colonne
        st.subheader("Comparaison par colonne")
        self._display_column_comparison()

        # Visualisations
        st.subheader("Visualisations comparatives")
        self._display_visualizations()

    def _display_global_metrics(self):
        metrics_data = []

        for method_name, imputed_df in self.imputed_results.items():
            # Valeurs manquantes restantes
            remaining_missing = imputed_df.isnull().sum().sum()

            # Pourcentage d'imputation
            original_missing = self.original_df.isnull().sum().sum()
            imputation_rate = (
                ((original_missing - remaining_missing) / original_missing * 100)
                if original_missing > 0
                else 100
            )

            # Cohérence des types
            type_consistency = self._check_type_consistency(imputed_df)

            metrics_data.append(
                {
                    "Méthode": method_name,
                    "Valeurs manquantes restantes": remaining_missing,
                    "Taux d'imputation (%)": round(imputation_rate, 2),
                    "Cohérence des types (%)": round(type_consistency, 2),
                }
            )

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    def _display_column_comparison(self):
        # Sélection de colonne
        missing_cols = [
            col
            for col in self.original_df.columns
            if self.original_df[col].isnull().any()
        ]

        if not missing_cols:
            st.info("Aucune colonne avec des valeurs manquantes")
            return

        selected_col = st.selectbox("Sélectionner une colonne", missing_cols)

        # Métriques par colonne
        col_metrics = []

        for method_name, imputed_df in self.imputed_results.items():
            if pd.api.types.is_numeric_dtype(self.original_df[selected_col]):
                # Métriques pour colonnes numériques
                metric_value = self._calculate_distribution_similarity(
                    self.original_df[selected_col].dropna(),
                    imputed_df[selected_col].dropna(),
                )
                metric_name = "Similarité distribution"
            else:
                # Métriques pour colonnes catégorielles
                metric_value = self._calculate_categorical_consistency(
                    self.original_df[selected_col].dropna(),
                    imputed_df[selected_col].dropna(),
                )
                metric_name = "Cohérence catégorielle"

            col_metrics.append(
                {
                    "Méthode": method_name,
                    metric_name: round(metric_value, 3),
                    "Valeurs imputées": imputed_df[selected_col].isnull().sum() == 0,
                }
            )

        col_metrics_df = pd.DataFrame(col_metrics)
        st.dataframe(col_metrics_df, use_container_width=True)

    def _display_visualizations(self):
        missing_cols = [
            col
            for col in self.original_df.columns
            if self.original_df[col].isnull().any()
        ]

        if not missing_cols:
            return

        col_to_plot = st.selectbox(
            "Colonne à visualiser", missing_cols, key="viz_select"
        )

        if pd.api.types.is_numeric_dtype(self.original_df[col_to_plot]):
            self._plot_numeric_comparison(col_to_plot)
        else:
            self._plot_categorical_comparison(col_to_plot)

    def _plot_numeric_comparison(self, column):
        cols = st.columns(len(self.imputed_results) + 1)

        # Distribution originale
        with cols[0]:
            st.write("**Original**")
            self.visualizer.plot_distribution(self.original_df[column].dropna())

        # Distributions imputées
        for i, (method_name, imputed_df) in enumerate(self.imputed_results.items()):
            with cols[i + 1]:
                st.write(f"**{method_name}**")
                self.visualizer.plot_distribution(imputed_df[column])

    def _plot_categorical_comparison(self, column):
        cols = st.columns(len(self.imputed_results) + 1)

        # Distribution originale
        with cols[0]:
            st.write("**Original**")
            self.visualizer.plot_categorical_distribution(
                self.original_df[column].dropna()
            )

        # Distributions imputées
        for i, (method_name, imputed_df) in enumerate(self.imputed_results.items()):
            with cols[i + 1]:
                st.write(f"**{method_name}**")
                self.visualizer.plot_categorical_distribution(imputed_df[column])

    def _check_type_consistency(self, imputed_df):
        consistent_types = 0
        total_cols = len(self.original_df.columns)

        for col in self.original_df.columns:
            if self.original_df[col].dtype == imputed_df[col].dtype:
                consistent_types += 1

        return (consistent_types / total_cols) * 100

    def _calculate_distribution_similarity(self, original_series, imputed_series):
        try:
            # Test de Kolmogorov-Smirnov
            statistic, p_value = stats.ks_2samp(original_series, imputed_series)
            return 1 - statistic  # Plus proche de 1 = plus similaire
        except:
            return 0.5

    def _calculate_categorical_consistency(self, original_series, imputed_series):
        try:
            # Comparaison des distributions de fréquence
            original_counts = original_series.value_counts(normalize=True)
            imputed_counts = imputed_series.value_counts(normalize=True)

            # Calcul de la similarité
            common_values = set(original_counts.index) & set(imputed_counts.index)
            if not common_values:
                return 0

            similarity = 0
            for value in common_values:
                similarity += min(original_counts[value], imputed_counts[value])

            return similarity
        except:
            return 0.5

    def export_results(self):
        if not self.imputed_results:
            return

        st.subheader("Export des résultats")

        # Sélection de la méthode à exporter
        method_to_export = st.selectbox(
            "Sélectionner la méthode à exporter",
            options=list(self.imputed_results.keys()),
        )

        if st.button("Télécharger CSV"):
            df_to_export = self.imputed_results[method_to_export]

            # Conversion en CSV
            csv_buffer = io.StringIO()
            df_to_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Télécharger le fichier CSV",
                data=csv_data,
                file_name=f"dataset_imputed_{method_to_export.replace(' ', '_')}.csv",
                mime="text/csv",
            )

        # Aperçu du résultat
        st.write("**Aperçu du résultat sélectionné :**")
        st.dataframe(self.imputed_results[method_to_export].head())
