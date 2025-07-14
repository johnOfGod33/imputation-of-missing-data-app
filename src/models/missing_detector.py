import numpy as np
import pandas as pd
import streamlit as st


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

    def configure_missing_values(self):
        st.subheader("Configuration des valeurs manquantes")

        # Détection automatique basique
        auto_detected = self._detect_automatic_missing()

        if auto_detected:
            st.write("**Valeurs manquantes détectées automatiquement :**")
            for col, values in auto_detected.items():
                if values:
                    st.write(f"- **{col}** : {values}")

        # Configuration manuelle
        st.write("**Configuration manuelle par colonne :**")

        config = {}
        columns = list(self.df.columns)

        for col in columns:
            with st.expander(f"Colonne: {col}"):
                col_type = str(self.df[col].dtype)
                st.write(f"Type: {col_type}")

                # Valeurs uniques (échantillon) - filtrer les NaN
                unique_values = [v for v in self.df[col].unique() if not pd.isna(v)][
                    :20
                ]
                st.write(f"Valeurs uniques (échantillon): {unique_values}")

                # Configuration des valeurs manquantes
                options_list = [str(v) for v in unique_values]
                default_values = [
                    str(v)
                    for v in unique_values
                    if str(v) in self.default_missing_values
                ]

                missing_values = st.multiselect(
                    f"Valeurs à considérer comme manquantes",
                    options=options_list,
                    default=default_values,
                    key=f"missing_{col}",
                )

                # Plage de valeurs valides pour les colonnes numériques
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    use_range = st.checkbox(
                        f"Définir une plage de valeurs valides", key=f"range_{col}"
                    )
                    if use_range:
                        min_val = (
                            float(self.df[col].min())
                            if not pd.isna(self.df[col].min())
                            else 0.0
                        )
                        max_val = (
                            float(self.df[col].max())
                            if not pd.isna(self.df[col].max())
                            else 100.0
                        )

                        valid_range = st.slider(
                            f"Plage valide pour {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"slider_{col}",
                        )
                        config[col] = {
                            "missing_values": missing_values,
                            "valid_range": valid_range,
                        }
                    else:
                        config[col] = {"missing_values": missing_values}
                else:
                    config[col] = {"missing_values": missing_values}

        return config

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

            # Appliquer la plage de valeurs valides
            if "valid_range" in col_config:
                min_val, max_val = col_config["valid_range"]
                mask = (df_processed[col] < min_val) | (df_processed[col] > max_val)
                df_processed.loc[mask, col] = np.nan

        # Afficher le résumé
        st.subheader("Résumé des valeurs manquantes après configuration")
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
            st.success("Aucune valeur manquante après configuration")

        return df_processed
