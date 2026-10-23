import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv("ai4i2020.csv")

# Display the first 5 rows
print(df.head())
# Check dataset size
print("Dataset shape:", df.shape)

# Check column names
print("Column names:")
print(df.columns.tolist())
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
# Check for duplicate rows
duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)
# Check data types
print("\nData types of each column:")
print(df.dtypes)
# Select the required columns for the ML model
selected_columns = [
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Machine failure'
]

df_cleaned = df[selected_columns].copy()

# Display the selected columns
print("\nSelected columns:")
print(df_cleaned.columns.tolist())

print("\nCleaned dataset shape:")
print(df_cleaned.shape)
# Encode the Type column using one-hot encoding
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['Type'],
    dtype=int
)

# Display the new columns
print("\nColumns after encoding:")
print(df_cleaned.columns.tolist())

print("\nDataset shape after encoding:")
print(df_cleaned.shape)
# Check numerical values
print("\nStatistical summary:")
print(df_cleaned.describe())
# Check minimum and maximum values of sensor features
sensor_columns = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

print("\nMinimum values:")
print(df_cleaned[sensor_columns].min())

print("\nMaximum values:")
print(df_cleaned[sensor_columns].max())
# Check the distribution of the target variable
print("\nMachine failure distribution:")
print(df_cleaned['Machine failure'].value_counts())

print("\nMachine failure percentage:")
print(df_cleaned['Machine failure'].value_counts(normalize=True) * 100)
# Save the cleaned dataset
output_file = "ai4i2020_cleaned.csv"

df_cleaned.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully as:", output_file)