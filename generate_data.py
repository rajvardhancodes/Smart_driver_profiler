"""
Data Generation Script for Smart Driver Profiler
Generates synthetic driver behavior dataset with 100+ records
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of drivers
n_drivers = 150

# List of random Indian names (mix of common first names)
indian_names = [
    "Arjun", "Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Karan", "Meera",
    "Aditya", "Kavya", "Rohan", "Divya", "Aman", "Pooja", "Siddharth", "Neha",
    "Varun", "Shreya", "Ayush", "Riya", "Krishna", "Anjali", "Rishi", "Tanvi",
    "Surya", "Isha", "Dhruv", "Aditi", "Vivek", "Nisha", "Abhishek", "Simran",
    "Nikhil", "Ritika", "Ravi", "Sakshi", "Kunal", "Preeti", "Akash", "Muskan",
    "Sahil", "Jyoti", "Harsh", "Swati", "Manish", "Deepika", "Gaurav", "Kiran",
    "Ankit", "Manisha", "Ritesh", "Vidya", "Yash", "Poonam", "Rajesh", "Suman",
    "Sandeep", "Lakshmi", "Amit", "Rekha", "Vinod", "Geeta", "Naveen", "Sunita",
    "Manoj", "Asha", "Pradeep", "Rashmi", "Sanjay", "Kalpana", "Dinesh", "Madhuri",
    "Avinash", "Sushma", "Bharat", "Radha", "Chandan", "Meena", "Deepak", "Sarita",
    "Gopal", "Kamala", "Jitendra", "Uma", "Mahesh", "Leela", "Pankaj", "Indira",
    "Ramesh", "Saroj", "Suresh", "Lata", "Mukesh", "Shanti", "Ashok", "Padma",
    "Harish", "Mala", "Sunil", "Kamala", "Bharat", "Shobha", "Raj", "Veena",
    "Kiran", "Seema", "Ajay", "Nirmala", "Vijay", "Usha", "Jagdish", "Pushpa",
    "Pramod", "Sarla", "Satish", "Sulochana", "Sudhir", "Kamini", "Brijesh", "Kavita",
    "Nitin", "Manju", "Vikas", "Rajni", "Alok", "Smita", "Anil", "Archana",
    "Tarun", "Charu", "Hemant", "Dolly", "Sachin", "Nidhi", "Nilesh", "Monika"
]

# Generate random driver names
import random
random.seed(42)
np.random.seed(42)

# Create a list of names (ensure we have enough unique names)
all_names = indian_names * ((n_drivers // len(indian_names)) + 1)
random.shuffle(all_names)
driver_names = all_names[:n_drivers]

# Generate synthetic data (we'll assign names after calculating fuel efficiency)
data = {
    
    # Trip distance (km) - range: 10-500 km
    'trip_distance': np.random.uniform(10, 500, n_drivers).round(2),
    
    # Average speed (km/h) - range: 30-120 km/h
    'avg_speed': np.random.uniform(30, 120, n_drivers).round(2),
    
    # Max speed (km/h) - correlated with avg_speed but higher
    'max_speed': np.random.uniform(80, 150, n_drivers).round(2),
    
    # Acceleration variance - higher for aggressive drivers
    'acceleration_variance': np.random.uniform(0.5, 5.0, n_drivers).round(2),
    
    # Harsh braking count - number of harsh braking incidents per trip
    'harsh_braking': np.random.poisson(2, n_drivers),
    
    # Idle time percentage - percentage of trip time spent idle
    'idle_time': np.random.uniform(5, 25, n_drivers).round(2),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate fuel_used based on driving patterns
# Realistic fuel efficiency for Indian vehicles:
# - Bikes: 40-80 km/L (average ~55 km/L)
# - Small Cars: 15-25 km/L (average ~20 km/L)
# - Sedans: 10-20 km/L (average ~15 km/L)
# - SUVs: 8-15 km/L (average ~12 km/L)

# Assign vehicle types (mix of bikes and cars)
vehicle_types = np.random.choice(['bike', 'small_car', 'sedan', 'suv'], 
                                 size=n_drivers, 
                                 p=[0.4, 0.3, 0.2, 0.1])  # 40% bikes, 30% small cars, 20% sedans, 10% SUVs

# Base fuel efficiency by vehicle type (km/L)
base_efficiency = np.where(
    vehicle_types == 'bike', np.random.uniform(45, 75, n_drivers),
    np.where(
        vehicle_types == 'small_car', np.random.uniform(18, 28, n_drivers),
        np.where(
            vehicle_types == 'sedan', np.random.uniform(12, 22, n_drivers),
            np.random.uniform(10, 16, n_drivers)  # SUV
        )
    )
)

# Adjust efficiency based on driving behavior
# Optimal speed range (60-80 km/h) gives best efficiency
speed_efficiency_factor = np.where(
    (df['avg_speed'] >= 60) & (df['avg_speed'] <= 80),
    1.0,  # Optimal - no penalty
    np.where(
        (df['avg_speed'] >= 40) & (df['avg_speed'] < 60),
        0.95,  # Slight penalty for slower speeds
        np.where(
            (df['avg_speed'] > 80) & (df['avg_speed'] <= 100),
            0.90,  # Penalty for higher speeds
            0.85   # Significant penalty for very high speeds
        )
    )
)

# Behavior penalty: aggressive driving reduces efficiency
# Higher acceleration variance, harsh braking, and idle time reduce efficiency
behavior_efficiency_factor = 1.0 - (
    (df['acceleration_variance'] - 1.0) * 0.03 +  # Each unit above 1.0 reduces efficiency by 3%
    df['harsh_braking'] * 0.02 +  # Each harsh brake reduces efficiency by 2%
    (df['idle_time'] - 5) * 0.005  # Idle time above 5% reduces efficiency
)

# Max speed penalty: very high speeds significantly reduce efficiency
max_speed_penalty = np.where(
    df['max_speed'] > 120,
    1.0 - ((df['max_speed'] - 120) * 0.002),  # Penalty for speeds above 120 km/h
    1.0
)

# Clamp behavior factors to reasonable ranges
behavior_efficiency_factor = np.clip(behavior_efficiency_factor, 0.70, 1.0)  # Between 70% and 100%
max_speed_penalty = np.clip(max_speed_penalty, 0.80, 1.0)  # Between 80% and 100%

# Calculate actual fuel efficiency (km/L)
actual_efficiency = base_efficiency * speed_efficiency_factor * behavior_efficiency_factor * max_speed_penalty

# Add some random variation
actual_efficiency = actual_efficiency * np.random.uniform(0.95, 1.05, n_drivers)

# Calculate fuel used (L) = distance (km) / efficiency (km/L)
df['fuel_used'] = (df['trip_distance'] / actual_efficiency).round(2)

# Ensure fuel_used is positive and reasonable (minimum 0.1L for very short trips)
df['fuel_used'] = np.maximum(df['fuel_used'], 0.1)

# Add some realistic correlations
# Aggressive drivers tend to have higher max_speed and acceleration_variance
aggressive_mask = (df['acceleration_variance'] > 3.5) | (df['harsh_braking'] > 3)
df.loc[aggressive_mask, 'max_speed'] = np.random.uniform(110, 150, aggressive_mask.sum()).round(2)

# Safe drivers tend to have lower values
safe_mask = (df['acceleration_variance'] < 1.5) & (df['harsh_braking'] < 2) & (df['max_speed'] < 100)
df.loc[safe_mask, 'acceleration_variance'] = np.random.uniform(0.5, 1.5, safe_mask.sum()).round(2)

# Calculate fuel efficiency to identify the best driver
df['fuel_efficiency'] = df['trip_distance'] / df['fuel_used']

# Sort by fuel efficiency (descending) to find the best driver
df_sorted = df.sort_values('fuel_efficiency', ascending=False).reset_index(drop=True)

# Assign names: "Rajvardhan" to the best driver, random names to others
driver_names_final = driver_names.copy()
# Remove one name from the list to make room for "Rajvardhan"
driver_names_final = driver_names_final[:n_drivers-1]

# Assign "Rajvardhan" to the top driver (best fuel efficiency)
df_sorted.loc[0, 'driver_id'] = "Rajvardhan"

# Assign random names to the rest
for i in range(1, n_drivers):
    df_sorted.loc[i, 'driver_id'] = driver_names_final[i-1]

# Shuffle the dataframe to randomize the order (but keep Rajvardhan as best)
# Actually, let's keep them sorted by efficiency so Rajvardhan is clearly at the top
# But we can also randomize if needed - let's randomize for more realistic data
df = df_sorted.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder columns (remove fuel_efficiency as it's a calculated field)
df = df[['driver_id', 'trip_distance', 'avg_speed', 'max_speed', 
         'acceleration_variance', 'harsh_braking', 'idle_time', 'fuel_used']]

# Save to CSV
df.to_csv('driver_data.csv', index=False)

print(f"Generated {n_drivers} driver records")
print(f"Saved to 'driver_data.csv'")
print(f"\nDataset Preview:")
print(df.head())
print(f"\nDataset Statistics:")
print(df.describe())

