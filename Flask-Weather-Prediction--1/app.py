from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import joblib
import datetime
import sqlite3
import pandas as pd
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ✅ OpenWeatherMap API Key (Replace with your actual API key)
API_KEY = "4ed1d04b3a2278df1de2d159e7a30666"

# ✅ Load trained ML model and expected feature names
model = joblib.load("temp_model.pkl")
model_features = joblib.load("model_features.pkl")

# ✅ Load historical dataset
historical_data = pd.read_csv("bombay.csv")
historical_data["datetime"] = pd.to_datetime(historical_data["datetime"], format="%d-%m-%Y")

# ✅ Database Setup
def init_db():
    """Create database and ensure predictions table exists."""
    conn = sqlite3.connect("weather.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            date TEXT,
            tmin REAL,
            tmax REAL,
            prcp REAL,
            prev_year_tavg REAL,
            prediction REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize database
init_db()

# ✅ Fetch real-time weather data
def get_current_weather(city):
    """Fetch current weather data from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "tmin": data["main"]["temp_min"],
            "tmax": data["main"]["temp_max"],
            "prcp": data.get("rain", {}).get("1h", 0)  # Rainfall in last 1 hour, default 0
        }
    else:
        return None

# ✅ Homepage Route
@app.route("/")
def home():
    return render_template("main.html")

# ✅ Predict Future Weather
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "city" not in data or "date" not in data:
            return jsonify({"error": "City and date must be provided"}), 400

        city = data["city"]
        future_date = datetime.datetime.strptime(data["date"], "%Y-%m-%d")

        # ✅ Fetch real-time weather data
        weather_data = get_current_weather(city)
        if not weather_data:
            return jsonify({"error": "Could not fetch real-time weather data"}), 500

        tmin, tmax, prcp = weather_data["tmin"], weather_data["tmax"], weather_data["prcp"]

        # ✅ Get previous year's average temperature
        prev_year_date = future_date.replace(year=future_date.year - 1)
        prev_year_data = historical_data[historical_data["datetime"] == prev_year_date]["tavg"]
        prev_year_tavg = prev_year_data.mean() if not prev_year_data.empty else (tmin + tmax) / 2

        # ✅ Prepare input features
        input_data = np.array([[tmin, tmax, prcp, future_date.year, future_date.month, future_date.day, prev_year_tavg]])
        input_df = dict(zip(model_features, input_data[0]))
        input_array = np.array([list(input_df.values())])

        # ✅ Make prediction
        prediction = model.predict(input_array)[0]

        # ✅ Save Prediction to Database
        conn = sqlite3.connect("weather.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (city, date, tmin, tmax, prcp, prev_year_tavg, prediction) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (city, future_date.strftime("%Y-%m-%d"), tmin, tmax, prcp, prev_year_tavg, prediction))
        conn.commit()
        conn.close()

        return jsonify({
            "city": city,
            "date": future_date.strftime("%Y-%m-%d"),
            "tmin": round(tmin, 2),
            "tmax": round(tmax, 2),
            "prcp": round(prcp, 2),
            "prev_year_tavg": round(prev_year_tavg, 2),
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Fetch Past Predictions
@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Fetch past predictions from the database."""
    try:
        conn = sqlite3.connect("weather.db")
        cursor = conn.cursor()
        cursor.execute("SELECT city, date, tmin, tmax, prcp, prev_year_tavg, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        predictions = [
            {"city": row[0], "date": row[1], "tmin": row[2], "tmax": row[3], "prcp": row[4], "prev_year_tavg": row[5], "prediction": row[6], "timestamp": row[7]}
            for row in rows
        ]

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)


