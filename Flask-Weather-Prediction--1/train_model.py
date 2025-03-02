import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv("bombay.csv")

# Convert 'datetime' to DateTime format
data["datetime"] = pd.to_datetime(data["datetime"], format="%d-%m-%Y", errors="coerce")

# Drop invalid dates
data.dropna(subset=["datetime"], inplace=True)

# Sort data by date (important for time series)
data.sort_values(by="datetime", inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy="median")
data[["tavg", "tmin", "tmax", "prcp"]] = imputer.fit_transform(data[["tavg", "tmin", "tmax", "prcp"]])

# Extract date-related features
data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day

# ✅ Add Previous Year's Temperature Feature (prev_year_tavg)
data["prev_year_tavg"] = data.apply(
    lambda row: data.loc[data["datetime"] == row["datetime"] - pd.DateOffset(years=1), "tavg"].mean(),
    axis=1
)

# Handle NaN values in prev_year_tavg (for first-year records)
data["prev_year_tavg"].fillna(data["tavg"].median(), inplace=True)  # Replace missing values with median

# Define Features & Target
X = data[["tmin", "tmax", "prcp", "year", "month", "day", "prev_year_tavg"]]  # Now includes prev_year_tavg
y = data["tavg"]  # Target: Predicting tavg

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model & Features
joblib.dump(model, "temp_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("✅ Model trained and saved successfully for future prediction!")
