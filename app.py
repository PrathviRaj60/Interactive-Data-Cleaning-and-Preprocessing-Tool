import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.title("Interactive Data Cleaning & Preprocessing Tool")
st.write("🚀 Upload a CSV file to get started.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # Handle Missing Values
    # -------------------------
    st.subheader("Handle Missing Values")

    strategy = st.selectbox(
        "Select strategy for handling missing values:",
        ("None (do nothing)", "Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode")
    )

    cleaned_df = df.copy()

    if strategy == "Drop rows":
        cleaned_df = cleaned_df.dropna()
    elif strategy == "Fill with Mean":
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif strategy == "Fill with Median":
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif strategy == "Fill with Mode":
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    st.subheader("Data After Handling Missing Values")
    st.dataframe(cleaned_df.head())

    # -------------------------
    # Numeric Columns
    # -------------------------
    numeric_cols = cleaned_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # -------------------------
    # Automatic Outlier Suggestion
    # -------------------------
    st.subheader("Automatic Outlier Suggestion")

    suggested_outlier_cols = []
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).any():
            suggested_outlier_cols.append(col)

    if suggested_outlier_cols:
        st.write("Columns with potential outliers detected:", suggested_outlier_cols)
        selected_col = st.selectbox("Select a numeric column to inspect outliers", suggested_outlier_cols)
        
        # Outlier Detection & Boxplot
        Q1 = cleaned_df[selected_col].quantile(0.25)
        Q3 = cleaned_df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = cleaned_df[
            (cleaned_df[selected_col] < lower_bound) |
            (cleaned_df[selected_col] > upper_bound)
        ]

        st.write(f"Number of outliers in **{selected_col}**: {outliers.shape[0]}")
        st.dataframe(outliers.head())

        st.subheader(f"Boxplot of {selected_col}")
        col_data = cleaned_df[selected_col]
        clipped_data = col_data.clip(lower=col_data.quantile(0.01), upper=col_data.quantile(0.99))
        fig, ax = plt.subplots(figsize=(8, 2))
        sns.boxplot(x=clipped_data, color="skyblue", fliersize=5, ax=ax)
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
    else:
        st.write("No numeric columns with outliers detected automatically.")

    # -------------------------
    # Categorical Encoding
    # -------------------------
    st.subheader("Categorical Encoding")

    cat_cols = cleaned_df.select_dtypes(include=["object"]).columns.tolist()
    encoded_df = cleaned_df.copy()  # always defined

    if cat_cols:
        selected_cat_cols = st.multiselect("Select categorical columns to encode", cat_cols)
        encoding_type = st.radio("Select encoding type", ("Label Encoding", "One-Hot Encoding"))

        if selected_cat_cols:
            if encoding_type == "Label Encoding":
                le = LabelEncoder()
                for col in selected_cat_cols:
                    encoded_df[col] = le.fit_transform(encoded_df[col])
            elif encoding_type == "One-Hot Encoding":
                encoded_df = pd.get_dummies(encoded_df, columns=selected_cat_cols)

    st.subheader("Preview of Data After Encoding")
    st.dataframe(encoded_df.head())

    # -------------------------
    # Normalization / Standardization
    # -------------------------
    st.subheader("Normalization / Standardization")

    if numeric_cols:
        selected_num_cols = st.multiselect("Select numeric columns to scale", numeric_cols)
        scaling_method = st.radio("Select scaling method", ("Min-Max Scaling", "Standardization"))

        scaled_df = encoded_df.copy()  # preserves categorical columns

        if selected_num_cols:
            if scaling_method == "Min-Max Scaling":
                for col in selected_num_cols:
                    min_val = scaled_df[col].min()
                    max_val = scaled_df[col].max()
                    scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)
            elif scaling_method == "Standardization":
                for col in selected_num_cols:
                    mean = scaled_df[col].mean()
                    std = scaled_df[col].std()
                    scaled_df[col] = (scaled_df[col] - mean) / std

            heading = "Preview of Min-Max Scaled Data" if scaling_method=="Min-Max Scaling" else "Preview of Standardized Data"
            st.subheader(heading)
            st.dataframe(scaled_df.head())
    else:
        st.write("No numeric columns available for scaling.")

    # -------------------------
    # Outlier Removal / Capping (Reactive)
    # -------------------------
    st.subheader("Outlier Removal / Capping")

    if numeric_cols:
        selected_outlier_col = st.selectbox("Select a numeric column to handle outliers", numeric_cols, key="outlier_col_reactive")

        method = st.radio("Choose method to handle outliers", ("Remove outliers", "Cap outliers"))

        Q1 = scaled_df[selected_outlier_col].quantile(0.25)
        Q3 = scaled_df[selected_outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        st.write(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        outlier_handled_df = scaled_df.copy()

        if method == "Remove outliers":
            outlier_handled_df = outlier_handled_df[
                (outlier_handled_df[selected_outlier_col] >= lower_bound) &
                (outlier_handled_df[selected_outlier_col] <= upper_bound)
            ]
        elif method == "Cap outliers":
            outlier_handled_df[selected_outlier_col] = outlier_handled_df[selected_outlier_col].clip(lower=lower_bound, upper=upper_bound)

        st.subheader("Data After Outlier Handling")
        st.dataframe(outlier_handled_df.head())

    # -------------------------
    # Export Cleaned Dataset
    # -------------------------
    st.subheader("Export Cleaned Dataset")

    if 'outlier_handled_df' in locals():
        final_df = outlier_handled_df.copy()
    else:
        final_df = scaled_df.copy()  # fallback if outlier handling not applied

    csv = final_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
