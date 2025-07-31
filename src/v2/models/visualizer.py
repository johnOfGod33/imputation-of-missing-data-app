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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

        clean_series = series.dropna()

        if len(clean_series) > 0:
            # Boxplot
            ax1.boxplot(clean_series)
            ax1.set_ylabel("Valeur")
            ax1.set_title(f"Boxplot - {series.name}")
            ax1.set_xticklabels([series.name])

            # Histogramme avec outliers mis en évidence
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Séparer les outliers des données normales
            normal_data = clean_series[
                (clean_series >= lower_bound) & (clean_series <= upper_bound)
            ]
            outliers = clean_series[
                (clean_series < lower_bound) | (clean_series > upper_bound)
            ]

            # Histogramme
            ax2.hist(
                normal_data, bins=20, alpha=0.7, color="blue", label="Données normales"
            )
            if len(outliers) > 0:
                ax2.hist(outliers, bins=5, alpha=0.7, color="red", label="Outliers")

            ax2.axvline(
                lower_bound,
                color="orange",
                linestyle="--",
                label=f"Seuil inf: {lower_bound:.2f}",
            )
            ax2.axvline(
                upper_bound,
                color="orange",
                linestyle="--",
                label=f"Seuil sup: {upper_bound:.2f}",
            )
            ax2.set_xlabel("Valeur")
            ax2.set_ylabel("Fréquence")
            ax2.set_title(f"Distribution avec outliers - {series.name}")
            ax2.legend(fontsize=8)

            # Statistiques compactes
            stats_text = f"Q1: {Q1:.2f} | Q3: {Q3:.2f} | IQR: {IQR:.2f}\nOutliers: {len(outliers)} ({(len(outliers)/len(clean_series)*100):.1f}%)"
            fig.text(0.02, 0.02, stats_text, fontsize=7, transform=fig.transFigure)

        else:
            ax1.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax2.text(
                0.5,
                0.5,
                "Aucune donnée",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
