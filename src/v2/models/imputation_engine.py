import warnings

import numpy as np
import pandas as pd
import streamlit as st
from miceforest import ImputationKernel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class ImputationEngine:
    def __init__(self, df):
        self.df = df
        self.methods = {
            "Simple - Mean": {"type": "simple", "strategy": "mean"},
            "Simple - Median": {"type": "simple", "strategy": "median"},
            "Simple - Mode": {"type": "simple", "strategy": "most_frequent"},
            "KNN": {"type": "knn", "n_neighbors": 5},
            "MICE Forest": {"type": "miceforest", "iterations": 5},
        }

    def select_methods(self):
        st.subheader("Sélection des méthodes d'imputation")

        selected_methods = st.multiselect(
            "Choisir les méthodes à comparer",
            options=list(self.methods.keys()),
            default=[],
        )

        # Configuration des paramètres
        method_configs = {}
        for method in selected_methods:
            method_configs[method] = self.methods[method].copy()

            if method == "KNN":
                k = st.slider(
                    f"Nombre de voisins (K) pour {method}", 1, 20, 5, key=f"k_{method}"
                )
                method_configs[method]["n_neighbors"] = k

            elif method == "MICE Forest":
                iterations = st.slider(
                    f"Nombre d'itérations pour {method}", 1, 15, 5, key=f"iter_{method}"
                )
                method_configs[method]["iterations"] = iterations

        return method_configs if selected_methods else None

    def execute_imputation(self, methods):
        st.subheader("Exécution des imputations")

        results = {}
        progress_bar = st.progress(0)

        for i, (method_name, config) in enumerate(methods.items()):
            with st.spinner(f"Exécution de {method_name}..."):
                try:
                    imputed_df = self._apply_imputation(config)
                    results[method_name] = imputed_df
                    st.success(f"✓ {method_name} terminé")
                except Exception as e:
                    st.error(f"✗ Erreur avec {method_name}: {str(e)}")

                progress_bar.progress((i + 1) / len(methods))

        return results

    def _apply_imputation(self, config):
        df_imputed = self.df.copy()

        # Séparer les colonnes numériques et catégorielles
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=["object"]).columns

        if config["type"] == "miceforest":
            # Utiliser MICE Forest pour toutes les colonnes
            df_imputed = self._apply_miceforest(df_imputed, config["iterations"])
        else:
            # Traitement des colonnes numériques
            if len(numeric_cols) > 0:
                numeric_data = df_imputed[numeric_cols]

                if config["type"] == "simple":
                    imputer = SimpleImputer(strategy=config["strategy"])
                    numeric_imputed = imputer.fit_transform(numeric_data)

                elif config["type"] == "knn":
                    imputer = KNNImputer(n_neighbors=config["n_neighbors"])
                    numeric_imputed = imputer.fit_transform(numeric_data)

                df_imputed[numeric_cols] = numeric_imputed

            # Traitement des colonnes catégorielles
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    if df_imputed[col].isnull().any():
                        # Pour les variables catégorielles, utiliser le mode
                        mode_value = df_imputed[col].mode()
                        if len(mode_value) > 0:
                            df_imputed[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_imputed[col].fillna("Unknown", inplace=True)

        return df_imputed

    def _apply_miceforest(self, df, iterations):
        # Préparer les données pour MICE Forest
        df_prep = df.copy()

        # Encoder les variables catégorielles
        categorical_cols = df_prep.select_dtypes(include=["object"]).columns
        label_encoders = {}

        for col in categorical_cols:
            if df_prep[col].isnull().any() or len(df_prep[col].unique()) > 1:
                le = LabelEncoder()
                # Fit sur les valeurs non nulles
                non_null_values = df_prep[col].dropna()
                if len(non_null_values) > 0:
                    le.fit(non_null_values)
                    # Transform en gardant les NaN
                    mask = df_prep[col].notnull()
                    df_prep.loc[mask, col] = le.transform(df_prep.loc[mask, col])
                    label_encoders[col] = le

        # Convertir tout en numérique
        df_numeric = df_prep.apply(pd.to_numeric, errors="coerce")

        # Initialiser et exécuter MICE Forest avec la syntaxe correcte
        kernel = ImputationKernel(data=df_numeric, random_state=42)

        # Effectuer l'imputation avec burn-in
        kernel.mice(iterations=iterations, verbose=False)

        # Récupérer les données imputées
        df_imputed = kernel.complete_data(0)

        # Décoder les variables catégorielles
        for col, le in label_encoders.items():
            # Arrondir et convertir en int pour le décodage
            df_imputed[col] = df_imputed[col].round().astype(int)
            # Clip aux valeurs valides
            df_imputed[col] = df_imputed[col].clip(0, len(le.classes_) - 1)
            # Décoder
            df_imputed[col] = le.inverse_transform(df_imputed[col])

        return df_imputed

    def get_imputation_summary(self, original_df, imputed_df, method_name):
        summary = {
            "method": method_name,
            "original_missing": original_df.isnull().sum().sum(),
            "remaining_missing": imputed_df.isnull().sum().sum(),
            "columns_processed": [],
        }

        for col in original_df.columns:
            if original_df[col].isnull().any():
                summary["columns_processed"].append(
                    {
                        "column": col,
                        "original_missing": original_df[col].isnull().sum(),
                        "remaining_missing": imputed_df[col].isnull().sum(),
                    }
                )

        return summary
