import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


class Visualizer:
    def __init__(self):
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_distribution(self, series, bins=30):
        fig, ax = plt.subplots(figsize=(8, 4))

        # Nettoyer les données
        clean_series = series.dropna()

        if len(clean_series) == 0:
            ax.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            # Histogramme
            ax.hist(clean_series, bins=bins, alpha=0.7, edgecolor="black")
            ax.set_ylabel("Fréquence")
            ax.set_xlabel("Valeur")
            ax.set_title(f"Distribution - {series.name}")

            # Statistiques
            mean_val = clean_series.mean()
            median_val = clean_series.median()
            ax.axvline(
                mean_val, color="red", linestyle="--", label=f"Moyenne: {mean_val:.2f}"
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                label=f"Médiane: {median_val:.2f}",
            )
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_categorical_distribution(self, series, max_categories=10):
        fig, ax = plt.subplots(figsize=(8, 4))

        # Nettoyer les données
        clean_series = series.dropna()

        if len(clean_series) == 0:
            ax.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            # Compter les valeurs
            value_counts = clean_series.value_counts().head(max_categories)

            # Graphique en barres
            bars = ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha="right")
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Distribution - {series.name}")

            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, value_counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    str(value),
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_missing_heatmap(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Matrice des valeurs manquantes
        missing_matrix = df.isnull()

        if missing_matrix.any().any():
            # Heatmap
            sns.heatmap(
                missing_matrix, cbar=True, cmap="viridis", yticklabels=False, ax=ax
            )
            ax.set_title("Heatmap des valeurs manquantes")
            ax.set_xlabel("Colonnes")
        else:
            ax.text(
                0.5,
                0.5,
                "Aucune valeur manquante",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Heatmap des valeurs manquantes")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def plot_correlation_matrix(self, df):
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Matrice de corrélation
            corr_matrix = numeric_df.corr()

            # Heatmap
            sns.heatmap(
                corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, ax=ax
            )
            ax.set_title("Matrice de corrélation")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Pas assez de colonnes numériques pour la corrélation")

    def plot_boxplot(self, series):
        fig, ax = plt.subplots(figsize=(8, 4))

        clean_series = series.dropna()

        if len(clean_series) > 0:
            ax.boxplot(clean_series)
            ax.set_ylabel("Valeur")
            ax.set_title(f"Boxplot - {series.name}")
            ax.set_xticklabels([series.name])
        else:
            ax.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
