"""
Imputation of missing data app
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from miceforest import ImputationKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer

from models import ImputationMethod

st.title("💧 Water Potability Data Processing")

dataset = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if dataset:
    df = pd.read_csv(dataset)

    # 1. Shape
    st.header("📊 Dataset Shape")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

    # 2. Head
    st.write("head")
    st.write(df)

    st.write("Describe variables")
    st.write(df.describe().T)

    # na values
    st.write("Na values")
    st.write(df.isnull().sum())

    # 3. Potability Distribution Visualization
    if "Potability" in df.columns:
        st.header("📈 Potability Distribution")

        # Create the figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Count plot
        sns.countplot(x="Potability", data=df, ax=ax[0], palette=["#005b96", "#c6e2ff"])
        ax[0].set_title("Count of Potability")

        # Pie chart
        potability_counts = df["Potability"].value_counts()
        ax[1].pie(
            potability_counts.values,
            labels=["Non-potable", "Potable"],
            autopct="%1.2f%%",
            shadow=True,
            explode=(0.05, 0),
            startangle=60,
            colors=["#005b96", "#c6e2ff"],
        )
        ax[1].set_title("Potability Distribution")

        fig.suptitle("Distribution of the Potability", fontsize=16)
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

    # 4. Boxplots for all columns
    st.header("📦 Boxplots of All Variables")

    # Calculate number of rows needed (2 columns)
    n_cols = 2
    n_rows = (len(df.columns) + 1) // 2  # Round up division

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        sns.boxplot(y=col, data=df, ax=axes[i], color="orange")
        axes[i].set_title(f"Box Plot of {col}", fontsize=12)
        axes[i].set_ylabel(col)

    # Hide empty subplots if any
    for i in range(len(df.columns), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # 5. Outlier Handling
    st.header("🔧 Outlier Handling")

    def handling_outlier(data, variable):
        quartile1 = data[variable].quantile(0.2)  # Range (%20-%80)
        quartile3 = data[variable].quantile(0.8)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        data.loc[data[variable] < low_limit, variable] = low_limit
        data.loc[data[variable] > up_limit, variable] = up_limit

    # Create a copy of the dataframe for outlier handling
    df_cleaned = df.copy()

    # Apply outlier handling to all columns except the last one
    for col in df_cleaned.columns[:-1]:
        handling_outlier(df_cleaned, col)

    st.success(f"✅ Outliers handled for {len(df_cleaned.columns[:-1])} variables")

    # Show comparison of statistics before and after
    st.subheader("📊 Statistics Comparison (Before vs After Outlier Handling)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before outlier handling:**")
        st.write(df.describe().T)

    with col2:
        st.write("**After outlier handling:**")
        st.write(df_cleaned.describe().T)

    # 6. Imputation choice
    inputation_cols = st.multiselect("Colonnes à imputer", df_cleaned.columns)

    # 7. Method choice
    method = st.multiselect(
        "Méthode d'imputation",
        [
            ImputationMethod.MEAN.value,
            ImputationMethod.KNN.value,
            ImputationMethod.RF.value,
            ImputationMethod.MF.value,
        ],
    )

    if len(method) >= 0 and inputation_cols:
        df_mean = None
        df_knn = None
        df_mice = None
        df_rf = None
        if ImputationMethod.MEAN.value in method:
            df_mean = df.copy()
            imputer = SimpleImputer(strategy="mean")
            df_mean[inputation_cols] = imputer.fit_transform(df_mean[inputation_cols])

            st.write(df_mean)

        if ImputationMethod.KNN.value in method:
            df_knn = df.copy()
            knn_imp = KNNImputer(n_neighbors=4)
            df_knn = pd.DataFrame(knn_imp.fit_transform(df_knn), columns=df_knn.columns)

            st.write(df_knn)

        if ImputationMethod.MF.value in method:
            df_mice = df.copy()
            mice_kernel = ImputationKernel(data=df_mice, random_state=41)
            mice_kernel.mice(2)
            df_mice = mice_kernel.complete_data()

            st.write(df_mice)

        if ImputationMethod.RF.value in method:
            df_rf = df.copy()
            rf = RandomForestRegressor()

            def fill_missing_numerical(df, col):
                known = df[df[col].notnull()]
                unknown = df[df[col].isnull()]
                X = known.drop(df.columns[df.isnull().any()], axis=1)  # features
                y = known[col]  # target
                rf.fit(X, y)
                unknown[col] = rf.predict(
                    unknown.drop(df.columns[df.isnull().any()], axis=1)
                )
                df[col].fillna(unknown[col], inplace=True)

        for col in inputation_cols:
            plt.figure(figsize=(10, 7))
            if df_mean is not None:
                df_mean.rename(columns={col: "Mean Imputation of " + col})[
                    "Mean Imputation of " + col
                ].plot(kind="kde", color="red")
            if df_knn is not None:
                df_knn.rename(columns={col: "KNN Imputation of " + col})[
                    "KNN Imputation of " + col
                ].plot(kind="kde", color="blue")
            if df_mice is not None:
                df_mice.rename(columns={col: "Multiple Imputation of " + col})[
                    "Multiple Imputation of " + col
                ].plot(kind="kde", color="green")
            if df_rf is not None:
                df_rf.rename(columns={col: "Predictive Imputation of " + col})[
                    "Predictive Imputation of " + col
                ].plot(kind="kde", color="yellow")

            plt.legend()
            plt.title(
                "Distribution of Filled "
                + col.upper()
                + " Values According to Imputation Methods"
            )
            # Display the plot in Streamlit
            st.pyplot(plt)
            plt.close()  # Close the figure to free memory

else:
    st.info("👆 Please upload a CSV file to start the analysis.")
