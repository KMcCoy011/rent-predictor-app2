# app.py
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

# Load and preprocess data
raw_df = pd.read_csv('TrueFinalData.csv')
df = raw_df.drop(columns=['Address', '0', 'Latitude', 'Longitude', 'Census Tract', 'Traffic', 'SoundScore'], errors='ignore')
df = df[df['City'].notna() & df['Home Type'].notna()]

# Encode categories
df = pd.get_dummies(df, columns=["City", "Home Type"], prefix=["City", "Home Type"], drop_first=True)
X = df.drop(columns=["Minimum Price"])
y = df["Minimum Price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save feature order for prediction
feature_order = X.columns.tolist()
joblib.dump(model, "model.pkl")
pd.Series(feature_order).to_csv("feature_order.csv", index=False)

@app.route('/')
def form():
    cities = sorted(raw_df['City'].dropna().unique())
    home_types = ['Apartment', 'Single-Family', 'Townhouse', 'Multi-Family', 'Condo']
    return render_template('form.html', cities=cities, home_types=home_types)

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load("model.pkl")
    feature_order = pd.read_csv("feature_order.csv", header=None)[0].tolist()

    input_dict = {
        'Minimum Beds': int(request.form['beds']),
        'Minimum Baths': int(request.form['baths']),
        'Sqft': float(request.form['sqft']),
        'Units': int(request.form['units']),
        'Noise Pollution': int(request.form['noise']),
        'PM2.5': int(request.form['pm25']),
        'Poverty': int(request.form['poverty']),
        'Distance to School': float(request.form['school']),
        'Distance to Hospital': float(request.form['hospital']),
        'Distance to Grocery Store': float(request.form['grocery'])
    }

    selected_city = request.form['city']
    selected_type = request.form['home_type']
    city_key = f"City_{selected_city}"
    type_key = f"Home Type_{'MULTIUNIT' if selected_type == 'Multi-Family' else selected_type.upper().replace('-', '_')}"

    for col in feature_order:
        if col.startswith("City_"):
            input_dict[col] = 1 if col == city_key else 0
        elif col.startswith("Home Type_"):
            input_dict[col] = 1 if col == type_key else 0
        elif col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])[feature_order]
    predicted_rent = model.predict(input_df)[0]
    return render_template("result.html", price=int(predicted_rent))

if __name__ == '__main__':
    app.run(debug=True)
