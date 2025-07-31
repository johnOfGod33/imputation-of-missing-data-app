import numpy as np
import pandas as pd
import streamlit as st

from .visualizer import Visualizer


class MissingDetector:
    def __init__(self, df):
        self.df = df
        self.default_missing_values = [
            "",
            "N/A",
            "?",
            "Unknown",
            "null",
            "NULL",
            "na",
            "NA",
            "-",
        ]
        self.visualizer = Visualizer()

    def configure_missing_values(self):  # Détection automatique basique
        auto_detected = self._detect_automatic_missing()

        if auto_detected:
            st.write("**Valeurs manquantes détectées automatiquement :**")
            for col, values in auto_detected.items():
                if values:
                    st.write(f"- **{col}** : {values}")

        # Configuration manuelle des valeurs personnalisées
        st.write("**Ajout de valeurs personnalisées manquantes :**")

        config = {}
        columns = list(self.df.columns)

        # Interface simplifiée pour l'ajout de valeurs manquantes
        custom_missing = st.text_input(
            "Valeurs supplémentaires à considérer comme manquantes (séparées par des virgules)",
            help="Ex: -999, 0, Unknown_Value",
        )

        # Parser les valeurs personnalisées
        custom_values = []
        if custom_missing.strip():
            custom_values = [v.strip() for v in custom_missing.split(",")]

        # Appliquer à toutes les colonnes
        for col in columns:
            config[col] = {
                "missing_values": self.default_missing_values + custom_values
            }

        # Gestion des outliers avec boxplots en colonnes
        st.subheader("Gestion des valeurs aberrantes (Outliers)")
        outlier_config = self._configure_outliers()

        # Combiner les configurations
        for col in config:
            if col in outlier_config:
                config[col].update(outlier_config[col])

        # Stocker la configuration pour utilisation ultérieure
        self.config = config

        return config

    def _configure_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_config = {}

        if len(numeric_cols) == 0:
            st.info("Aucune colonne numérique pour la détection d'outliers")
            return outlier_config

        # Afficher tous les boxplots en colonnes
        st.write("**Visualisation des outliers :**")

        # Calculer le nombre de colonnes optimal (max 3 par ligne)
        cols_per_row = min(3, len(numeric_cols))
        num_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row

        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                global_idx = row * cols_per_row + col_idx
                if global_idx < len(numeric_cols):
                    col_name = numeric_cols[global_idx]
                    with cols[col_idx]:
                        st.write(f"**{col_name}**")
                        self.visualizer.plot_boxplot(self.df[col_name])

                        # Calcul des outliers avec IQR
                        outliers_info = self._detect_outliers_iqr(self.df[col_name])

                        if outliers_info["outliers_count"] > 0:
                            st.write(f"Outliers: {outliers_info['outliers_count']}")
                        else:
                            st.write("✅ Aucun outlier")

        # Bouton simple pour traiter les outliers
        if len(numeric_cols) > 0:
            st.write("**Traitement des outliers :**")
            handle_all_outliers = st.button(
                "Traiter les valeurs aberrantes", type="primary"
            )

            if handle_all_outliers:
                for col in numeric_cols:
                    outliers_info = self._detect_outliers_iqr(self.df[col])
                    if outliers_info["outliers_count"] > 0:
                        outlier_config[col] = {
                            "handle_outliers": "Traiter comme valeurs manquantes",
                            "outlier_bounds": (
                                outliers_info["lower_bound"],
                                outliers_info["upper_bound"],
                            ),
                        }
                    else:
                        outlier_config[col] = {"handle_outliers": "Conserver"}

        return outlier_config

    def _detect_outliers_iqr(self, series):
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"outliers_count": 0, "lower_bound": 0, "upper_bound": 0}

        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = clean_series[
            (clean_series < lower_bound) | (clean_series > upper_bound)
        ]

        return {
            "outliers_count": len(outliers),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers": outliers,
        }

    def _detect_automatic_missing(self):
        detected = {}

        for col in self.df.columns:
            col_missing = []

            # Valeurs NaN
            if self.df[col].isnull().any():
                col_missing.append("NaN")

            # Valeurs par défaut
            for val in self.default_missing_values:
                if val in self.df[col].values:
                    col_missing.append(val)

            if col_missing:
                detected[col] = col_missing

        return detected

    def apply_missing_detection(self, config):
        df_processed = self.df.copy()

        for col, col_config in config.items():
            # Remplacer les valeurs manquantes par NaN
            missing_values = col_config["missing_values"]
            df_processed[col] = df_processed[col].replace(missing_values, np.nan)

            # Traitement des outliers
            if (
                "handle_outliers" in col_config
                and col_config["handle_outliers"] != "Conserver"
            ):
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    lower_bound, upper_bound = col_config["outlier_bounds"]

                    if (
                        col_config["handle_outliers"]
                        == "Traiter comme valeurs manquantes"
                    ):
                        # Remplacer les outliers par NaN
                        mask = (df_processed[col] < lower_bound) | (
                            df_processed[col] > upper_bound
                        )
                        df_processed.loc[mask, col] = np.nan

        # Afficher le résumé
        st.subheader("Résumé après traitement")
        missing_summary = df_processed.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(
            ascending=False
        )

        if len(missing_summary) > 0:
            summary_df = pd.DataFrame(
                {
                    "Colonne": missing_summary.index,
                    "Valeurs manquantes": missing_summary.values,
                    "Pourcentage": (
                        missing_summary.values / len(df_processed) * 100
                    ).round(2),
                }
            )
            st.dataframe(summary_df)
        else:
            st.success("Aucune valeur manquante après traitement")

        # Réafficher les boxplots après traitement
        self._show_post_treatment_boxplots(df_processed)

        return df_processed

    def _show_post_treatment_boxplots(self, df_processed):
        """Affiche les boxplots après traitement pour montrer l'effet"""
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return

        # Vérifier s'il y a eu un traitement d'outliers
        had_outlier_treatment = any(
            "handle_outliers" in config and config["handle_outliers"] != "Conserver"
            for config in [self.config.get(col, {}) for col in numeric_cols]
        )

        if had_outlier_treatment:
            st.subheader("📊 Visualisation après traitement des outliers")
            st.write("**Comparaison avant/après traitement :**")

            # Calculer le nombre de colonnes optimal (max 3 par ligne)
            cols_per_row = min(3, len(numeric_cols))
            num_rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row

            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    global_idx = row * cols_per_row + col_idx
                    if global_idx < len(numeric_cols):
                        col_name = numeric_cols[global_idx]
                        with cols[col_idx]:
                            st.write(f"**{col_name}**")

                            # Boxplot après traitement
                            self.visualizer.plot_boxplot(df_processed[col_name])

                            # Comparaison des statistiques
                            original_outliers = self._detect_outliers_iqr(
                                self.df[col_name]
                            )
                            processed_outliers = self._detect_outliers_iqr(
                                df_processed[col_name]
                            )

                            if original_outliers["outliers_count"] > 0:
                                st.write(
                                    f"Outliers avant: {original_outliers['outliers_count']}"
                                )
                                st.write(
                                    f"Outliers après: {processed_outliers['outliers_count']}"
                                )

                                if processed_outliers["outliers_count"] == 0:
                                    st.success("✅ Tous les outliers traités")
                            else:
                                st.write("✅ Aucun outlier détecté")
