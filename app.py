from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import folium
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("TrueFinalData.csv")
df.drop(columns=['Address', '0', 'Latitude', 'Longitude', 'Census Tract', 'SoundScore'], errors='ignore', inplace=True)

# Save address and coordinates for mapping
df_coords = pd.read_csv("TrueFinalData.csv")[["Address", "Latitude", "Longitude", "City", "Minimum Price", "Units"]].dropna()

# One-hot encode
df = pd.get_dummies(df, columns=["City", "Home Type"], prefix=["City", "Home Type"], drop_first=True)
X = df.drop(columns=["Minimum Price"])
y = df["Minimum Price"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression().fit(X_train, y_train)
feature_order = X.columns.tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        form = request.form

        # Base numerical features
        input_data = {
            "Minimum Beds": int(form["beds"]),
            "Minimum Baths": int(form["baths"]),
            "Sqft": int(form["sqft"]),
            "Units": int(form["units"]),
            "Noise Pollution": int(form["noise"]),
            "PM2.5": int(form["pm25"]),
            "Poverty": int(form["poverty"]),
            "Distance to School": float(form["school"]),
            "Distance to Hospital": float(form["hospital"]),
            "Distance to Grocery Store": float(form["grocery"]),
        }

        # One-hot for city and home type
        for col in feature_order:
            if col.startswith("City_"):
                input_data[col] = 1 if col == f"City_{form['city']}" else 0
            elif col.startswith("Home Type_"):
                selected_type = form["home_type"]
                selected_type = "MULTI_FAMILY" if selected_type == "MULTIUNIT" else selected_type
                input_data[col] = 1 if col == f"Home Type_{selected_type}" else 0

        # Fill missing columns
        for col in feature_order:
            input_data.setdefault(col, 0)

        input_df = pd.DataFrame([input_data])[feature_order]
        prediction = model.predict(input_df)[0]

        return render_template("results.html", rent=int(prediction), map_file="combined_map.html")

    # For GET request
    cities = sorted(df_coords["City"].dropna().unique())
    home_types = ["Apartment", "Single-Family", "Townhouse", "Multi-Family", "Condo"]
    return render_template("form.html", cities=cities, home_types=home_types)

@app.before_request
def create_interactive_map():
    df_map = df_coords.copy()

    base_map = folium.Map(
        location=[df_map["Latitude"].mean(), df_map["Longitude"].mean()],
        zoom_start=10, tiles="CartoDB positron"
    )

    # --- Rental Price Layer ---
    price_layer = folium.FeatureGroup(name="Rental Prices")
    norm = mcolors.Normalize(vmin=1000, vmax=6000)
    cmap = cm.get_cmap("plasma")
    for _, row in df_map.iterrows():
        color = mcolors.to_hex(cmap(norm(np.clip(row["Minimum Price"], 1000, 6000))))
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2, color=color, fill=True, fill_opacity=0.9,
            popup=f"{row['Address']}<br>Price: ${int(row['Minimum Price'])}"
        ).add_to(price_layer)
    price_layer.add_to(base_map)

    # --- Property Density Layer ---
    density_layer = folium.FeatureGroup(name="Property Density")
    coords_rad = np.radians(df_map[["Latitude", "Longitude"]])
    tree = BallTree(coords_rad, metric='haversine')
    radius = 1 / 3958.8
    df_map["NearbyCount"] = tree.query_radius(coords_rad, r=radius, count_only=True)
    df_map["DensityScore"] = df_map["NearbyCount"] * df_map["Units"].fillna(1)
    norm_density = mcolors.Normalize(vmin=df_map["DensityScore"].min(), vmax=df_map["DensityScore"].max())
    cmap_density = cm.get_cmap("viridis")
    for _, row in df_map.iterrows():
        color = mcolors.to_hex(cmap_density(norm_density(row["DensityScore"])))
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2, color=color, fill=True, fill_opacity=1.0,
            popup=f"{row['Address']}<br>Density Score: {int(row['DensityScore'])}"
        ).add_to(density_layer)
    density_layer.add_to(base_map)

    # --- City Grouping Layer ---
    city_layer = folium.FeatureGroup(name="City Grouping")
    cities = df_map["City"].unique()
    cmap_city = cm.get_cmap("Paired", len(cities))
    city_color_map = {city: mcolors.to_hex(cmap_city(i / len(cities))) for i, city in enumerate(sorted(cities))}
    for _, row in df_map.iterrows():
        color = city_color_map[row["City"]]
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2, color=color, fill=True, fill_opacity=0.8,
            popup=f"{row['Address']}<br>{row['City']}"
        ).add_to(city_layer)
    for city in cities:
        coords = df_map[df_map["City"] == city][["Latitude", "Longitude"]].mean()
        folium.Marker(
            location=[coords["Latitude"], coords["Longitude"]],
            icon=folium.DivIcon(html=f"""
                <div style="font-size:10px; color:white; font-weight:bold;
                            text-shadow: 0 0 3px black; text-align:center;">
                    {city}
                </div>""")
        ).add_to(city_layer)
    city_layer.add_to(base_map)

    folium.LayerControl().add_to(base_map)

    # Ensure static dir exists and save
    os.makedirs("static", exist_ok=True)
    base_map.save("static/combined_map.html")
    create_interactive_map()
