import pandas as pd
import numpy as np
import folium
from flask import Flask, render_template, request
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Initialize Flask app
app = Flask(__name__)

# Load data
df = pd.read_csv("TrueFinalData.csv")
df = df.drop(columns=['Address', '0', 'Latitude', 'Longitude', 'Census Tract', 'SoundScore'], errors='ignore')

# Save a version with addresses and coordinates for map generation
df_map = pd.read_csv("TrueFinalData.csv")
df_map = df_map.dropna(subset=["Latitude", "Longitude", "Minimum Price"])

# Preprocess for ML model
df_model = pd.get_dummies(df, columns=["City", "Home Type"], prefix=["City", "Home Type"], drop_first=True)
X = df_model.drop(columns=["Minimum Price"])
y = df_model["Minimum Price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Save order of columns to match in prediction
feature_order = X.columns.tolist()
pd.Series(feature_order).to_csv("feature_order.csv", index=False)

# Generate map
def create_interactive_map():
    base_map = folium.Map(
        location=[df_map["Latitude"].mean(), df_map["Longitude"].mean()],
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True
    )

    def add_gradient_layer(df, col, layer_name, colormap='inferno', radius=2):
        layer = folium.FeatureGroup(name=layer_name, show=False)
        norm = mcolors.Normalize(vmin=df[col].min(), vmax=df[col].max())
        cmap = cm.get_cmap(colormap)
        for _, row in df.iterrows():
            color = mcolors.to_hex(cmap(norm(row[col])))
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=f"{row['Address']}<br>{col}: {row[col]}"
            ).add_to(layer)
        return layer

    # Base layer: rental prices
    price_layer = add_gradient_layer(df_map, "Minimum Price", "Rental Prices", "plasma")
    price_layer.show = True
    price_layer.add_to(base_map)

    # Additional layers
    layers_to_add = [
        ("PM2.5", "Air Pollution", "Oranges"),
        ("Poverty", "Poverty Rate", "Reds"),
        ("Traffic", "Traffic Level", "Greys"),
        ("Noise Pollution", "Noise Pollution", "Purples"),
        ("Distance to School", "Distance to School", "Blues"),
        ("Distance to Hospital", "Distance to Hospital", "BuGn"),
        ("Distance to Grocery Store", "Distance to Grocery Store", "YlGnBu"),
        ("Sqft", "Square Footage", "PuRd")
    ]
    for col, title, cmap in layers_to_add:
        if col in df_map.columns:
            add_gradient_layer(df_map, col, title, colormap=cmap).add_to(base_map)

    # Add city grouping layer
    city_layer = folium.FeatureGroup(name="City Grouping", show=False)
    cities = df_map["City"].unique()
    cmap_city = cm.get_cmap("Paired", len(cities))
    city_color_map = {city: mcolors.to_hex(cmap_city(i / len(cities))) for i, city in enumerate(sorted(cities))}
    for _, row in df_map.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2,
            color=city_color_map[row["City"]],
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['Address']}<br>{row['City']}"
        ).add_to(city_layer)
    for city in cities:
        coords = df_map[df_map["City"] == city][["Latitude", "Longitude"]].mean()
        folium.Marker(
            location=[coords["Latitude"], coords["Longitude"]],
            icon=folium.DivIcon(html=f"<div style='font-size:10px;color:white;text-shadow:0 0 3px black'>{city}</div>")
        ).add_to(city_layer)
    city_layer.add_to(base_map)

    folium.LayerControl().add_to(base_map)
    base_map.save("static/combined_map.html")

create_interactive_map()

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    cities = sorted(df['City'].dropna().unique())
    home_types = sorted(df['Home Type'].dropna().unique())
    prediction = None

    if request.method == 'POST':
        user_input = {
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
        selected_home = request.form['home_type']

        # Prepare dummy vector
        input_vector = {feat: 0 for feat in feature_order}
        input_vector.update(user_input)
        input_vector[f"City_{selected_city}"] = 1
        input_vector[f"Home Type_{selected_home}"] = 1

        X_pred = pd.DataFrame([input_vector])
        prediction = int(model.predict(X_pred)[0])

    return render_template('form.html', cities=cities, home_types=home_types, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
