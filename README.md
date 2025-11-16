# 🚗 Smart Driver Profiler with Fuel Efficiency Analysis

**Tagline: "How You Drive Is What You Pay"**

A comprehensive data science web application that analyzes driving behavior, classifies drivers into categories (Safe, Normal, Aggressive), and estimates fuel efficiency impact based on driving patterns.

## Features

- 📊 **Data Analysis**: Upload and analyze driver behavior datasets
- 🔍 **Exploratory Data Analysis**: Comprehensive visualizations and correlation analysis
- 🎯 **Driver Clustering**: K-Means clustering to classify drivers (Safe/Normal/Aggressive)
- ⛽ **Fuel Efficiency Prediction**: Random Forest model to predict fuel consumption
- 👤 **Driver Profile Analyzer**: Interactive tool to analyze individual driver profiles
- 📈 **Dashboard**: Overall statistics and key insights

## Tech Stack

- **Frontend & Backend**: Streamlit (Python)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (KMeans clustering, Random Forest Regression)
- **Visualization**: matplotlib, seaborn, plotly

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic driver data:
```bash
python generate_data.py
```

This will create `driver_data.csv` with 150 driver records.

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Application Pages

1. **📊 Dashboard**: Overview of the dataset with key statistics and insights
2. **📁 Data Analysis**: Upload data, view dataset overview, and analyze missing values
3. **🔍 Exploratory Data Analysis**: Correlation heatmaps, distributions, scatter plots, and box plots
4. **🎯 Driver Clustering**: Perform K-Means clustering to classify drivers
5. **⛽ Fuel Efficiency Prediction**: Train Random Forest model and view predictions
6. **👤 Driver Profile Analyzer**: Interactive tool to analyze individual driving profiles

## Dataset Features

- `driver_id`: Unique driver identifier
- `trip_distance`: Trip distance in kilometers
- `avg_speed`: Average speed in km/h
- `max_speed`: Maximum speed in km/h
- `acceleration_variance`: Variance in acceleration
- `harsh_braking`: Number of harsh braking incidents
- `idle_time`: Percentage of time spent idle
- `fuel_used`: Fuel consumed in liters

## Machine Learning Models

- **K-Means Clustering**: Classifies drivers into 3 categories (Safe, Normal, Aggressive)
- **Random Forest Regression**: Predicts fuel consumption based on driving behavior

## Notes

- The application uses synthetic data by default
- You can upload your own CSV file with the same column structure
- Models are trained on-the-fly and stored in session state
- All visualizations are interactive using Plotly

## License

This project is open source and available for educational purposes.

