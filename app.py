import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

st.set_page_config(page_title="ID Analysis ML Tool", layout="wide")

st.title("ID Analysis ML Tool")
st.write("Upload your CSV file for machine learning analysis")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully loaded!")
        
        # Show data overview
        st.header("Data Overview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display the first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Data info and statistics
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        
        with col1:
            # Display column types
            buffer = []
            for dtype in df.dtypes.unique():
                buffer.append(f"- {dtype}: {sum(df.dtypes == dtype)}")
            st.write("Column Types:")
            st.write("\n".join(buffer))
            
            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.write("Missing Values:")
                st.write(missing[missing > 0])
            else:
                st.write("No missing values")
        
        with col2:
            # Basic statistics for numerical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                st.write("Numerical Statistics:")
                st.write(df[numeric_cols].describe())
        
        # ML Analysis Section
        st.header("Machine Learning Analysis")
        
        # Column selection
        st.subheader("Feature Selection")
        id_column = st.selectbox("Select ID Column", df.columns)
        
        with st.expander("Select Features for Analysis"):
            feature_columns = st.multiselect(
                "Select Feature Columns", 
                [col for col in df.columns if col != id_column],
                default=[col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != id_column][:5]
            )
        
        if len(feature_columns) < 1:
            st.warning("Please select at least one feature column")
        else:
            # Target selection (optional - for supervised learning)
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Unsupervised (Clustering)", "Supervised (Classification)", "Anomaly Detection"]
            )
            
            target_column = None
            if analysis_type == "Supervised (Classification)":
                target_column = st.selectbox(
                    "Select Target Column (for classification)", 
                    [None] + [col for col in df.columns if col != id_column and col not in feature_columns]
                )
            
            # Data preparation
            X = df[feature_columns].copy()
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                st.write("Handling missing values in selected features...")
                for col in X.columns:
                    if X[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(X[col]):
                            X[col].fillna(X[col].median(), inplace=True)
                        else:
                            X[col].fillna(X[col].mode()[0], inplace=True)
            
            # Handle categorical data
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                st.write("Encoding categorical features...")
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply ML analysis
            if st.button("Run Analysis"):
                st.subheader("Analysis Results")
                
                if analysis_type == "Unsupervised (Clustering)":
                    # Determine optimal number of clusters
                    with st.spinner("Finding optimal number of clusters..."):
                        wcss = []
                        silhouette_scores = []
                        max_clusters = min(10, len(df) - 1)
                        for i in range(2, max_clusters + 1):
                            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            wcss.append(kmeans.inertia_)
                            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
                    
                    # Plot elbow method
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(range(2, max_clusters + 1), wcss, 'b-', marker='o')
                    ax1.set_xlabel('Number of Clusters')
                    ax1.set_ylabel('WCSS', color='b')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'r-', marker='s')
                    ax2.set_ylabel('Silhouette Score', color='r')
                    
                    st.pyplot(fig)
                    
                    # Let user select number of clusters
                    n_clusters = st.slider("Select number of clusters", 2, max_clusters, 
                                         value=silhouette_scores.index(max(silhouette_scores)) + 2)
                    
                    # Apply KMeans with selected number of clusters
                    with st.spinner(f"Clustering with {n_clusters} clusters..."):
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Add cluster labels to original data
                        result_df = df.copy()
                        result_df['Cluster'] = clusters
                        
                        # Show cluster statistics
                        st.write("Cluster Sizes:")
                        st.write(pd.Series(clusters).value_counts().sort_index())
                        
                        # Visualize clusters with PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                        pca_df['Cluster'] = clusters
                        pca_df['ID'] = df[id_column]
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', ax=ax)
                        plt.title('Cluster Visualization with PCA')
                        st.pyplot(fig)
                        
                        # Feature importance for each cluster
                        st.subheader("Cluster Characteristics")
                        for i in range(n_clusters):
                            st.write(f"**Cluster {i}**")
                            cluster_data = result_df[result_df['Cluster'] == i]
                            st.write(f"Size: {len(cluster_data)} records")
                            
                            # Calculate mean values for each feature by cluster
                            cluster_means = result_df.groupby('Cluster')[feature_columns].mean()
                            overall_means = result_df[feature_columns].mean()
                            
                            # Calculate the difference from the overall mean
                            diff_from_mean = (cluster_means.loc[i] - overall_means) / overall_means
                            diff_from_mean = diff_from_mean.sort_values(ascending=False)
                            
                            # Display distinguishing features
                            st.write("Distinguishing features (% difference from overall mean):")
                            for feat, val in diff_from_mean.items():
                                if not pd.isna(val):
                                    direction = "higher" if val > 0 else "lower"
                                    st.write(f"- {feat}: {abs(val*100):.1f}% {direction}")
                            
                        # Download option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="id_analysis_clusters.csv",
                            mime="text/csv",
                        )
                
                elif analysis_type == "Supervised (Classification)":
                    if target_column is None:
                        st.error("Please select a target column for classification")
                    else:
                        # Prepare target variable
                        y = df[target_column]
                        
                        # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.3, random_state=42
                        )
                        
                        # Train a Random Forest classifier
                        with st.spinner("Training Random Forest classifier..."):
                            rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train)
                            
                            # Evaluate the model
                            y_pred = rf.predict(X_test)
                            
                            # Classification report
                            st.write("Classification Report:")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())
                            
                            # Confusion matrix
                            st.write("Confusion Matrix:")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            plt.xlabel('Predicted Label')
                            plt.ylabel('True Label')
                            st.pyplot(fig)
                            
                            # Feature importance
                            st.write("Feature Importance:")
                            feature_imp = pd.DataFrame({
                                'Feature': X.columns,
                                'Importance': rf.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(x='Importance', y='Feature', data=feature_imp[:15], ax=ax)
                            plt.title('Feature Importance')
                            st.pyplot(fig)
                            
                            # Predictions on full dataset
                            result_df = df.copy()
                            result_df['Predicted'] = rf.predict(X_scaled)
                            
                            # Download option
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="id_analysis_classification.csv",
                                mime="text/csv",
                            )
                
                elif analysis_type == "Anomaly Detection":
                    with st.spinner("Detecting anomalies..."):
                        # Train an Isolation Forest for anomaly detection
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        anomaly_scores = iso_forest.fit_predict(X_scaled)
                        
                        # Convert predictions to binary: 1 for normal, -1 for anomaly
                        # and then to 0 for normal, 1 for anomaly for easier interpretation
                        anomalies = np.where(anomaly_scores == -1, 1, 0)
                        
                        # Add anomaly flags to original data
                        result_df = df.copy()
                        result_df['Anomaly'] = anomalies
                        result_df['Anomaly_Score'] = iso_forest.score_samples(X_scaled)
                        
                        # Show anomaly statistics
                        st.write(f"Detected {sum(anomalies)} anomalies out of {len(df)} records ({sum(anomalies)/len(df)*100:.2f}%)")
                        
                        # Visualize anomalies with PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                        pca_df['Anomaly'] = anomalies
                        pca_df['ID'] = df[id_column]
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.scatterplot(
                            x='PC1', y='PC2', 
                            hue='Anomaly', 
                            data=pca_df, 
                            palette={0: 'blue', 1: 'red'},
                            ax=ax
                        )
                        plt.title('Anomaly Detection Visualization')
                        st.pyplot(fig)
                        
                        # Display most anomalous records
                        st.write("Top 10 Most Anomalous Records:")
                        top_anomalies = result_df.sort_values('Anomaly_Score').head(10)
                        st.dataframe(top_anomalies)
                        
                        # Download option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="id_analysis_anomalies.csv",
                            mime="text/csv",
                        )
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    # Display instructions when no file is uploaded
    st.info("Please upload a CSV file to begin analysis")
    
    # Example template
    st.subheader("Example CSV Format")
    example_df = pd.DataFrame({
        'ID': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
        'Age': [34, 28, 45, 52, 39],
        'Income': [58000, 72000, 91000, 65000, 82000],
        'Years_Experience': [5, 3, 12, 20, 9],
        'Num_Purchases': [12, 5, 24, 8, 15],
        'Customer_Type': ['Regular', 'New', 'Premium', 'Regular', 'Premium']
    })
    st.dataframe(example_df)
    
    # Usage instructions
    st.subheader("How to Use This Tool")
    st.markdown("""
    1. Upload a CSV file with your data
    2. Review the data overview and statistics
    3. Select your ID column and features for analysis
    4. Choose an analysis type:
       - **Unsupervised (Clustering)**: Group similar records
       - **Supervised (Classification)**: Predict categories
       - **Anomaly Detection**: Find unusual records
    5. Run the analysis and explore results
    6. Download the analyzed data with new insights
    """)

# Add footer with info
st.markdown("---")
st.markdown("ID Analysis ML Tool - Built with Streamlit")