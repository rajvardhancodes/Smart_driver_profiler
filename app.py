"""
Smart Driver Profiler with Fuel Efficiency Analysis
Main Streamlit Application
Tagline: "How You Drive Is What You Pay"
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Driver Profiler",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
    }
    .tagline {
        text-align: center;
        font-size: 1.3rem;
        color: #2c3e50;
        font-weight: 500;
        margin-bottom: 2rem;
        padding: 0.5rem;
    }
    .info-box {
        background-color: #005f73;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00a8cc;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box h3, .info-box h4 {
        color: #ffffff;
    }
    .info-box p, .info-box strong, .info-box ul, .info-box li {
        color: #ffffff;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3
if 'silhouette_avg' not in st.session_state:
    st.session_state.silhouette_avg = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Sidebar navigation with descriptions
st.sidebar.title("🚗 Smart Driver Profiler")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Navigation")

# Page descriptions
page_descriptions = {
    "📊 Dashboard": "Overview & key insights",
    "📁 Data Analysis": "View & explore dataset",
    "🔍 Exploratory Data Analysis": "Visualizations & patterns",
    "🎯 Driver Clustering": "Classify drivers (Safe/Normal/Aggressive)",
    "⛽ Fuel Efficiency Prediction": "Predict fuel consumption",
    "👤 Driver Profile Analyzer": "Analyze your driving profile"
}

page = st.sidebar.radio(
    "Select a page:",
    list(page_descriptions.keys()),
    format_func=lambda x: f"{x} - {page_descriptions[x]}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info("""
**Getting Started:**
1. Start with the Dashboard
2. Explore your data
3. Train models
4. Analyze your profile
""")

# Add help section in sidebar
with st.sidebar.expander("❓ Need Help?"):
    st.markdown("""
    **Common Questions:**
    - Data not loading? Run `python generate_data.py`
    - Models not training? Check data is loaded first
    - Need to restart? Refresh the page (F5)
    """)

# Helper function to load data
@st.cache_data
def load_data(file_path='driver_data.csv'):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' not found. Please run 'generate_data.py' first.")
        return None

# Dashboard Page
if page == "📊 Dashboard":
    st.markdown('<div class="main-header">🚗 Smart Driver Profiler</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">How You Drive Is What You Pay</div>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0; color: #ffffff;">👋 Welcome to Smart Driver Profiler!</h3>
        <p style="color: #ffffff;">Analyze driving behavior, predict fuel efficiency, and get personalized recommendations to improve your driving habits.</p>
        <p style="color: #ffffff;"><strong>Get Started:</strong> Explore the pages using the sidebar to analyze data, train models, and view your driving profile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        st.session_state.df = df
        
        # Overall Statistics with better explanation
        st.header("📈 Overall Statistics")
        st.markdown("**Key metrics from your driver dataset:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Drivers", len(df))
        
        with col2:
            st.metric("Average Fuel Used (L)", f"{df['fuel_used'].mean():.2f}")
        
        with col3:
            avg_fuel_efficiency = (df['trip_distance'] / df['fuel_used']).mean()
            st.metric("Average Fuel Efficiency", f"{avg_fuel_efficiency:.2f} km/L")
        
        with col4:
            st.metric("Total Distance (km)", f"{df['trip_distance'].sum():.0f}")
        
        st.markdown("---")
        
        # Key Insights with better formatting
        st.header("💡 Key Insights & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Overview")
            df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']
            
            info_data = {
                "Total Drivers": len(df),
                "Features": len(df.columns),
                "Missing Values": df.isnull().sum().sum(),
                "Best Efficiency": f"{df['fuel_efficiency'].max():.2f} km/L",
                "Worst Efficiency": f"{df['fuel_efficiency'].min():.2f} km/L",
                "Average Efficiency": f"{df['fuel_efficiency'].mean():.2f} km/L"
            }
            
            for key, value in info_data.items():
                st.markdown(f"**{key}**: {value}")
            
            # Top driver highlight
            best_driver = df.loc[df['fuel_efficiency'].idxmax()]
            st.markdown("---")
            st.success(f"🏆 **Top Performer**: {best_driver['driver_id']} with {best_driver['fuel_efficiency']:.2f} km/L")
        
        with col2:
            st.subheader("🎯 Fuel Efficiency Tips")
            st.info("""
            **💡 Improve Your Fuel Efficiency:**
            
            ✅ **Speed Management**
            - Maintain 60-80 km/h for optimal efficiency
            - Avoid speeds above 100 km/h
            
            ✅ **Smooth Driving**
            - Accelerate gradually
            - Brake smoothly and anticipate stops
            - Minimize sudden speed changes
            
            ✅ **Reduce Idle Time**
            - Turn off engine during long stops
            - Plan routes to avoid traffic
            
            ✅ **Regular Maintenance**
            - Keep tires properly inflated
            - Regular engine service
            """)
        
        st.markdown("---")
        
        # Quick Visualizations
        st.header("📊 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='fuel_efficiency', nbins=30, 
                             title='Fuel Efficiency Distribution',
                             labels={'fuel_efficiency': 'Fuel Efficiency (km/L)', 'count': 'Number of Drivers'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='avg_speed', y='fuel_efficiency',
                           title='Average Speed vs Fuel Efficiency',
                           labels={'avg_speed': 'Average Speed (km/h)', 
                                  'fuel_efficiency': 'Fuel Efficiency (km/L)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations with Fuel Efficiency")
        corr_features = ['trip_distance', 'avg_speed', 'max_speed', 'acceleration_variance', 
                        'harsh_braking', 'idle_time', 'fuel_efficiency']
        corr_matrix = df[corr_features].corr()['fuel_efficiency'].sort_values(ascending=False)
        
        fig = px.bar(x=corr_matrix.index, y=corr_matrix.values,
                    title='Correlation with Fuel Efficiency',
                    labels={'x': 'Features', 'y': 'Correlation Coefficient'})
        st.plotly_chart(fig, use_container_width=True)

# Data Analysis Page
elif page == "📁 Data Analysis":
    st.header("📁 Data Analysis")
    st.markdown("**Upload or view your driver behavior dataset. Explore the data structure and statistics.**")
    
    st.markdown("""
    <div class="info-box">
        <strong style="color: #ffffff;">📖 What is this page?</strong><br>
        <span style="color: #ffffff;">This page allows you to upload your own driver data or view the existing dataset. 
        You can see dataset overview, statistics, and check for any missing values.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload option
    st.subheader("📤 Upload Your Data (Optional)")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with the same structure as driver_data.csv. Required columns: driver_id, trip_distance, avg_speed, max_speed, acceleration_variance, harsh_braking, idle_time, fuel_used"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data loaded successfully!")
    else:
        df = load_data()
        if df is not None:
            st.session_state.df = df
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Dataset Overview with better formatting
        st.subheader("📊 Dataset Overview")
        st.markdown("**Quick summary of your dataset:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Rows", len(df), help="Total number of driver records")
        with col2:
            st.metric("📊 Columns", len(df.columns), help="Number of features in the dataset")
        with col3:
            st.metric("✅ Missing Values", df.isnull().sum().sum(), help="Total missing data points")
        with col4:
            st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB", help="Dataset size in memory")
        
        # Display first few rows with expander
        st.subheader("🔍 Dataset Preview")
        st.markdown("**First 10 records of your dataset:**")
        with st.expander("👁️ View Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show column info
        st.markdown("**💡 Tip:** Scroll horizontally to see all columns")
        
        # Dataset Info
        st.subheader("ℹ️ Dataset Information")
        st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Columns**: {', '.join(df.columns.tolist())}")
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing Value Analysis
        st.subheader("🔍 Missing Value Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(df)) * 100
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            # Visualize missing values
            fig = px.bar(missing_df[missing_df['Missing Count'] > 0], 
                        x='Column', y='Missing Count',
                        title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Types
        st.subheader("📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)

# Exploratory Data Analysis Page
elif page == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    st.markdown("**Explore patterns, correlations, and relationships in your driver data through interactive visualizations.**")
    
    st.markdown("""
    <div class="info-box">
        <strong style="color: #ffffff;">📊 What you'll find here:</strong><br>
        <span style="color: #ffffff;">• Correlation heatmaps showing relationships between features</span><br>
        <span style="color: #ffffff;">• Distribution plots for key metrics</span><br>
        <span style="color: #ffffff;">• Scatter plots revealing trends</span><br>
        <span style="color: #ffffff;">• Box plots comparing different driving patterns</span>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df if st.session_state.df is not None else load_data()
    
    if df is not None:
        st.session_state.df = df
        
        # Calculate fuel efficiency
        df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']
        
        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu',
                       aspect="auto")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution Plots
        st.subheader("📊 Distribution Plots")
        
        selected_feature = st.selectbox("Select feature for distribution", 
                                       numeric_cols.tolist())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=selected_feature, nbins=30,
                             title=f'Distribution of {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_feature,
                        title=f'Box Plot of {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Plots
        st.subheader("📈 Scatter Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='avg_speed', y='fuel_used',
                           title='Average Speed vs Fuel Used',
                           labels={'avg_speed': 'Average Speed (km/h)', 
                                  'fuel_used': 'Fuel Used (L)'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='harsh_braking', y='fuel_used',
                           title='Harsh Braking vs Fuel Used',
                           labels={'harsh_braking': 'Harsh Braking Count', 
                                  'fuel_used': 'Fuel Used (L)'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='acceleration_variance', y='fuel_efficiency',
                           title='Acceleration Variance vs Fuel Efficiency',
                           labels={'acceleration_variance': 'Acceleration Variance', 
                                  'fuel_efficiency': 'Fuel Efficiency (km/L)'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='idle_time', y='fuel_efficiency',
                           title='Idle Time vs Fuel Efficiency',
                           labels={'idle_time': 'Idle Time (%)', 
                                  'fuel_efficiency': 'Fuel Efficiency (km/L)'},
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        # Box Plots
        st.subheader("📦 Box Plots - Fuel Efficiency by Driving Patterns")
        
        # Create categorical bins for better visualization
        df['speed_category'] = pd.cut(df['avg_speed'], 
                                     bins=[0, 60, 80, 120, 200], 
                                     labels=['Slow', 'Optimal', 'Fast', 'Very Fast'])
        
        df['braking_category'] = pd.cut(df['harsh_braking'], 
                                       bins=[-1, 1, 3, 10], 
                                       labels=['Low', 'Medium', 'High'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='speed_category', y='fuel_efficiency',
                        title='Fuel Efficiency by Speed Category',
                        labels={'speed_category': 'Speed Category', 
                               'fuel_efficiency': 'Fuel Efficiency (km/L)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='braking_category', y='fuel_efficiency',
                        title='Fuel Efficiency by Braking Category',
                        labels={'braking_category': 'Braking Category', 
                               'fuel_efficiency': 'Fuel Efficiency (km/L)'})
            st.plotly_chart(fig, use_container_width=True)

# Driver Clustering Page
elif page == "🎯 Driver Clustering":
    st.header("🎯 Driver Clustering")
    
    df = st.session_state.df if st.session_state.df is not None else load_data()
    
    if df is not None:
        st.session_state.df = df
        # Calculate fuel efficiency
        df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']
        
        st.subheader("🔧 Clustering Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=3)
        
        with col2:
            if st.button("🚀 Perform Clustering", type="primary"):
                with st.spinner("Clustering drivers..."):
                    # Select features for clustering
                    features = ['avg_speed', 'acceleration_variance', 'harsh_braking', 'idle_time']
                    X = df[features].values
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Perform K-Means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    st.session_state.clusters = clusters
                    st.session_state.kmeans_model = kmeans
                    st.session_state.n_clusters = n_clusters
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(X_scaled, clusters)
                    st.session_state.silhouette_avg = silhouette_avg
                    
                    st.success(f"✅ Clustering completed! Silhouette Score: {silhouette_avg:.3f}")
        
        if st.session_state.clusters is not None:
            clusters = st.session_state.clusters
            n_clusters = st.session_state.n_clusters
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            
            # Map clusters to labels
            cluster_labels = ['Safe', 'Normal', 'Aggressive']
            if n_clusters == 3:
                # Sort clusters by average fuel efficiency (higher is better)
                cluster_means = df_clustered.groupby('cluster')['fuel_efficiency'].mean().sort_values(ascending=False)
                label_mapping = {cluster_means.index[i]: cluster_labels[i] for i in range(3)}
            else:
                label_mapping = {i: f'Cluster {i+1}' for i in range(n_clusters)}
            
            df_clustered['driver_type'] = df_clustered['cluster'].map(label_mapping)
            
            # Cluster Distribution
            st.subheader("📊 Cluster Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = df_clustered['driver_type'].value_counts()
                colors = {'Safe': 'green', 'Normal': 'orange', 'Aggressive': 'red'}
                if n_clusters > 3:
                    colors = px.colors.qualitative.Set3[:n_clusters]
                
                fig = px.pie(values=cluster_counts.values, 
                           names=cluster_counts.index,
                           title='Driver Category Distribution',
                           color_discrete_map=colors if n_clusters == 3 else None)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                silhouette_avg = st.session_state.silhouette_avg
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                st.write("**Cluster Sizes:**")
                for cluster_type, count in cluster_counts.items():
                    st.write(f"- {cluster_type}: {count} drivers")
            
            # 2D Visualization using PCA
            st.subheader("📈 Cluster Visualization (2D PCA)")
            
            features = ['avg_speed', 'acceleration_variance', 'harsh_braking', 'idle_time']
            X = df[features].values
            X_scaled = st.session_state.scaler.transform(X)
            
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_pca['cluster'] = clusters
            df_pca['driver_type'] = df_pca['cluster'].map(label_mapping)
            
            fig = px.scatter(df_pca, x='PC1', y='PC2', color='driver_type',
                           title='Driver Clusters (PCA Visualization)',
                           labels={'PC1': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                  'PC2': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                           color_discrete_map=colors if n_clusters == 3 else None)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster Characteristics
            st.subheader("📊 Cluster Characteristics Comparison")
            
            comparison_features = ['avg_speed', 'acceleration_variance', 'harsh_braking', 
                                 'idle_time', 'fuel_efficiency', 'fuel_used']
            
            cluster_stats = df_clustered.groupby('driver_type')[comparison_features].mean()
            
            for feature in comparison_features:
                fig = px.bar(x=cluster_stats.index, y=cluster_stats[feature],
                           title=f'Average {feature.replace("_", " ").title()} by Cluster',
                           labels={'x': 'Driver Type', 'y': feature.replace("_", " ").title()},
                           color=cluster_stats.index,
                           color_discrete_map=colors if n_clusters == 3 else None)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster details
            st.subheader("🔍 Cluster Details")
            st.dataframe(cluster_stats, use_container_width=True)

# Fuel Efficiency Prediction Page
elif page == "⛽ Fuel Efficiency Prediction":
    st.header("⛽ Fuel Efficiency Prediction")
    st.markdown("**Train a machine learning model to predict fuel consumption based on driving behavior.**")
    
    st.markdown("""
    <div class="info-box">
        <strong style="color: #ffffff;">🔮 Prediction Model:</strong><br>
        <span style="color: #ffffff;">This page uses Random Forest Regression to predict fuel consumption. 
        The model learns from your data and can predict how much fuel will be used based on driving patterns.</span>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df if st.session_state.df is not None else load_data()
    
    if df is not None:
        st.session_state.df = df
        df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']
        
        st.subheader("🤖 Train Random Forest Model")
        st.markdown("**Click the button below to train the prediction model on your data:**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model..."):
                    # Prepare features and target
                    feature_cols = ['trip_distance', 'avg_speed', 'max_speed', 
                                  'acceleration_variance', 'harsh_braking', 'idle_time']
                    X = df[feature_cols].values
                    y = df['fuel_used'].values
                    
                    # Train-test split
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X_train, y_train)
                    st.session_state.rf_model = rf_model
                    st.session_state.feature_cols = feature_cols
                    
                    # Predictions
                    y_pred_train = rf_model.predict(X_train)
                    y_pred_test = rf_model.predict(X_test)
                    
                    # Calculate metrics
                    r2_train = r2_score(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred_test)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    st.session_state.metrics = {
                        'r2_train': r2_train, 'r2_test': r2_test,
                        'mae_train': mae_train, 'mae_test': mae_test,
                        'rmse_train': rmse_train, 'rmse_test': rmse_test
                    }
                    
                    st.success("✅ Model trained successfully!")
        
        if st.session_state.rf_model is not None:
            metrics = st.session_state.metrics
            
            # Display Metrics
            st.subheader("📊 Model Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score (Test)", f"{metrics['r2_test']:.3f}")
                st.metric("R² Score (Train)", f"{metrics['r2_train']:.3f}")
            
            with col2:
                st.metric("MAE (Test)", f"{metrics['mae_test']:.3f} L")
                st.metric("MAE (Train)", f"{metrics['mae_train']:.3f} L")
            
            with col3:
                st.metric("RMSE (Test)", f"{metrics['rmse_test']:.3f} L")
                st.metric("RMSE (Train)", f"{metrics['rmse_train']:.3f} L")
            
            # Feature Importance
            st.subheader("🎯 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Importance': st.session_state.rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature',
                        orientation='h', title='Feature Importance in Fuel Prediction',
                        labels={'Importance': 'Importance Score'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Actual vs Predicted
            st.subheader("📈 Actual vs Predicted")
            
            # Get test predictions
            feature_cols = st.session_state.feature_cols
            X_test = df[feature_cols].values[int(len(df) * 0.8):]
            y_test = df['fuel_used'].values[int(len(df) * 0.8):]
            y_pred = st.session_state.rf_model.predict(X_test)
            
            fig = px.scatter(x=y_test, y=y_pred,
                           title='Actual vs Predicted Fuel Consumption',
                           labels={'x': 'Actual Fuel Used (L)', 'y': 'Predicted Fuel Used (L)'},
                           trendline="ols")
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals Plot
            st.subheader("📊 Residuals Plot")
            residuals = y_test - y_pred
            
            fig = px.scatter(x=y_pred, y=residuals,
                           title='Residuals Plot',
                           labels={'x': 'Predicted Fuel Used (L)', 'y': 'Residuals (L)'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

# Driver Profile Analyzer Page
elif page == "👤 Driver Profile Analyzer":
    st.header("👤 Driver Profile Analyzer")
    st.markdown("**Enter your driving behavior metrics to get personalized analysis and recommendations.**")
    
    st.markdown("""
    <div class="info-box">
        <strong style="color: #ffffff;">🎯 What does this do?</strong><br>
        <span style="color: #ffffff;">Enter your driving metrics below and get instant feedback on:</span>
        <ul style="color: #ffffff;">
            <li style="color: #ffffff;">Your driver category (Safe, Normal, or Aggressive)</li>
            <li style="color: #ffffff;">Predicted fuel consumption</li>
            <li style="color: #ffffff;">Fuel efficiency calculation</li>
            <li style="color: #ffffff;">Personalized improvement suggestions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df if st.session_state.df is not None else load_data()
    
    if df is not None:
        st.session_state.df = df
        df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']
        
        # Check if models are trained
        if st.session_state.kmeans_model is None or st.session_state.rf_model is None:
            st.warning("⚠️ Please train the clustering and prediction models first from their respective pages.")
            if st.button("🚀 Train Models Now"):
                with st.spinner("Training models..."):
                    # Train clustering
                    features_cluster = ['avg_speed', 'acceleration_variance', 'harsh_braking', 'idle_time']
                    X_cluster = df[features_cluster].values
                    scaler = StandardScaler()
                    X_cluster_scaled = scaler.fit_transform(X_cluster)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_cluster_scaled)
                    st.session_state.scaler = scaler
                    st.session_state.kmeans_model = kmeans
                    st.session_state.clusters = clusters
                    st.session_state.n_clusters = 3
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(X_cluster_scaled, clusters)
                    st.session_state.silhouette_avg = silhouette_avg
                    
                    # Train RF model
                    feature_cols = ['trip_distance', 'avg_speed', 'max_speed', 
                                  'acceleration_variance', 'harsh_braking', 'idle_time']
                    X = df[feature_cols].values
                    y = df['fuel_used'].values
                    split_idx = int(len(X) * 0.8)
                    X_train = X[:split_idx]
                    y_train = y[:split_idx]
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X_train, y_train)
                    st.session_state.rf_model = rf_model
                    st.session_state.feature_cols = feature_cols
                    
                    st.success("✅ Models trained successfully!")
                    st.rerun()
        else:
            # Input form with better labels and help text
            st.subheader("📝 Enter Your Driving Metrics")
            st.markdown("**Adjust the sliders below to match your driving behavior:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚗 Trip Details")
                trip_distance = st.slider(
                    "Trip Distance (km)", 
                    min_value=10.0, max_value=500.0, value=100.0, step=1.0,
                    help="Total distance of your trip in kilometers"
                )
                avg_speed = st.slider(
                    "Average Speed (km/h)", 
                    min_value=30.0, max_value=120.0, value=60.0, step=1.0,
                    help="Your average speed during the trip. Optimal: 60-80 km/h"
                )
                max_speed = st.slider(
                    "Max Speed (km/h)", 
                    min_value=80.0, max_value=150.0, value=100.0, step=1.0,
                    help="Maximum speed reached during the trip"
                )
            
            with col2:
                st.markdown("#### 📊 Driving Behavior")
                acceleration_variance = st.slider(
                    "Acceleration Variance", 
                    min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                    help="How much your acceleration varies. Lower is better (smoother driving)"
                )
                harsh_braking = st.slider(
                    "Harsh Braking Count", 
                    min_value=0, max_value=10, value=2, step=1,
                    help="Number of sudden/harsh braking incidents during the trip"
                )
                idle_time = st.slider(
                    "Idle Time (%)", 
                    min_value=5.0, max_value=25.0, value=10.0, step=0.5,
                    help="Percentage of trip time spent with engine idling"
                )
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button("🔍 Analyze My Driver Profile", type="primary", use_container_width=True)
            
            if analyze_button:
                with st.spinner("Analyzing driver profile..."):
                    # Predict driver category
                    features_cluster = np.array([[avg_speed, acceleration_variance, harsh_braking, idle_time]])
                    features_cluster_scaled = st.session_state.scaler.transform(features_cluster)
                    cluster = st.session_state.kmeans_model.predict(features_cluster_scaled)[0]
                    
                    # Map cluster to label using the same method as clustering page
                    cluster_labels = ['Safe', 'Normal', 'Aggressive']
                    # Use dataset to determine label mapping based on fuel efficiency
                    if 'clusters' in st.session_state and st.session_state.clusters is not None:
                        df_temp = df.copy()
                        df_temp['cluster'] = st.session_state.clusters
                        cluster_means = df_temp.groupby('cluster')['fuel_efficiency'].mean().sort_values(ascending=False)
                        if len(cluster_means) == 3:
                            label_mapping = {cluster_means.index[i]: cluster_labels[i] for i in range(3)}
                            driver_type = label_mapping.get(cluster, f'Cluster {cluster}')
                        else:
                            # Fallback: use cluster centers
                            cluster_centers = st.session_state.kmeans_model.cluster_centers_
                            cluster_scores = []
                            for i, center in enumerate(cluster_centers):
                                score = -(center[1] + center[2] + center[3])
                                cluster_scores.append(score)
                            sorted_indices = np.argsort(cluster_scores)[::-1]
                            label_mapping = {sorted_indices[0]: 'Safe', sorted_indices[1]: 'Normal', sorted_indices[2]: 'Aggressive'}
                            driver_type = label_mapping.get(cluster, f'Cluster {cluster}')
                    else:
                        # Fallback: use cluster centers
                        cluster_centers = st.session_state.kmeans_model.cluster_centers_
                        cluster_scores = []
                        for i, center in enumerate(cluster_centers):
                            score = -(center[1] + center[2] + center[3])
                            cluster_scores.append(score)
                        sorted_indices = np.argsort(cluster_scores)[::-1]
                        label_mapping = {sorted_indices[0]: 'Safe', sorted_indices[1]: 'Normal', sorted_indices[2]: 'Aggressive'}
                        driver_type = label_mapping.get(cluster, f'Cluster {cluster}')
                    
                    # Predict fuel consumption
                    feature_cols = st.session_state.feature_cols
                    features_rf = np.array([[trip_distance, avg_speed, max_speed, 
                                           acceleration_variance, harsh_braking, idle_time]])
                    predicted_fuel = st.session_state.rf_model.predict(features_rf)[0]
                    fuel_efficiency = trip_distance / predicted_fuel
                    
                    # Display Results with better formatting
                    st.markdown("---")
                    st.subheader("📊 Your Analysis Results")
                    st.markdown("**Here's what we found about your driving profile:**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        color_map = {'Safe': ('#28a745', '🟢'), 'Normal': ('#ffc107', '🟡'), 'Aggressive': ('#dc3545', '🔴')}
                        color, emoji = color_map.get(driver_type, ('#007bff', '🔵'))
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, {color}15, {color}05); 
                                    border-radius: 10px; border: 2px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h2 style='color: {color}; margin: 0; font-size: 2rem;'>{emoji}</h2>
                            <h3 style='color: {color}; margin: 0.5rem 0 0 0;'>{driver_type}</h3>
                            <p style='margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;'>Driver Category</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("⛽ Predicted Fuel Used", f"{predicted_fuel:.2f} L", help="Estimated fuel consumption for your trip")
                    
                    with col3:
                        st.metric("📊 Fuel Efficiency", f"{fuel_efficiency:.2f} km/L", help="Distance traveled per liter of fuel")
                    
                    with col4:
                        # Compare with average
                        avg_efficiency = df['fuel_efficiency'].mean()
                        efficiency_diff = fuel_efficiency - avg_efficiency
                        delta_color = "normal" if efficiency_diff >= 0 else "inverse"
                        st.metric("📈 vs Average", f"{fuel_efficiency:.2f} km/L", 
                                delta=f"{efficiency_diff:+.2f} km/L", delta_color=delta_color,
                                help="Comparison with dataset average")
                    
                    # Improvement Suggestions with better formatting
                    st.markdown("---")
                    st.subheader("💡 Personalized Improvement Suggestions")
                    
                    suggestions = []
                    suggestion_icons = {
                        'speed': '⚡',
                        'acceleration': '🚗',
                        'braking': '🛑',
                        'idle': '⏸️',
                        'max_speed': '🏎️',
                        'below_avg': '📉'
                    }
                    
                    if avg_speed < 60 or avg_speed > 80:
                        suggestions.append({
                            'icon': '⚡',
                            'title': 'Speed Optimization',
                            'message': f'Maintain average speed between 60-80 km/h for optimal fuel efficiency. Your current speed: {avg_speed:.1f} km/h',
                            'type': 'warning'
                        })
                    
                    if acceleration_variance > 2.5:
                        suggestions.append({
                            'icon': '🚗',
                            'title': 'Smooth Acceleration',
                            'message': f'Reduce acceleration variance for smoother driving. Current: {acceleration_variance:.2f}',
                            'type': 'info'
                        })
                    
                    if harsh_braking > 3:
                        suggestions.append({
                            'icon': '🛑',
                            'title': 'Reduce Harsh Braking',
                            'message': f'Anticipate stops to brake smoothly. Current harsh braking incidents: {harsh_braking}',
                            'type': 'warning'
                        })
                    
                    if idle_time > 15:
                        suggestions.append({
                            'icon': '⏸️',
                            'title': 'Reduce Idle Time',
                            'message': f'Turn off engine during long stops. Current idle time: {idle_time:.1f}%',
                            'type': 'info'
                        })
                    
                    if max_speed > 120:
                        suggestions.append({
                            'icon': '🏎️',
                            'title': 'Control Maximum Speed',
                            'message': f'Avoid excessive speeds. High speeds significantly increase fuel consumption. Current max: {max_speed:.1f} km/h',
                            'type': 'warning'
                        })
                    
                    if fuel_efficiency < avg_efficiency:
                        suggestions.append({
                            'icon': '📉',
                            'title': 'Below Average Efficiency',
                            'message': f'Your fuel efficiency ({fuel_efficiency:.2f} km/L) is below the average ({avg_efficiency:.2f} km/L). Consider implementing the suggestions above.',
                            'type': 'error'
                        })
                    
                    if len(suggestions) == 0:
                        st.markdown("""
                        <div class="success-box">
                            <h4 style="margin-top: 0;">✅ Excellent Driving!</h4>
                            <p>Your driving behavior is optimal for fuel efficiency. Keep up the good work!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        for suggestion in suggestions:
                            icon = suggestion['icon']
                            title = suggestion['title']
                            message = suggestion['message']
                            stype = suggestion['type']
                            
                            if stype == 'error':
                                st.error(f"**{icon} {title}**\n\n{message}")
                            elif stype == 'warning':
                                st.warning(f"**{icon} {title}**\n\n{message}")
                            else:
                                st.info(f"**{icon} {title}**\n\n{message}")
                    
                    # Visualizations
                    st.subheader("📈 Your Profile vs Dataset")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Compare with cluster averages
                        df_temp = df.copy()
                        df_temp['cluster'] = st.session_state.clusters
                        cluster_avg = df_temp.groupby('cluster')['fuel_efficiency'].mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=['Your Efficiency', 'Dataset Average'],
                                           y=[fuel_efficiency, avg_efficiency],
                                           marker_color=['blue', 'gray']))
                        fig.update_layout(title='Your Fuel Efficiency vs Dataset Average',
                                        yaxis_title='Fuel Efficiency (km/L)')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Speed comparison
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=['Your Avg Speed', 'Optimal Range'],
                                           y=[avg_speed, 70],
                                           marker_color=['orange', 'green']))
                        fig.update_layout(title='Your Average Speed vs Optimal Range',
                                        yaxis_title='Speed (km/h)')
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    pass

