import streamlit as st
import requests
import json
import math
import folium
from streamlit_folium import st_folium
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Nearby Places Finder",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.distance-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
}
.info-box {
    background: #e7f3ff;
    border-left: 4px solid #2196F3;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Haversine Distance Calculation ────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in kilometers"""
    R = 6371  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

# ─── Geolocation JavaScript ────────────────────────────────────────────
def get_geolocation():
    """Get user's location using browser geolocation API"""
    geolocation_script = """
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                window.parent.document.getElementById("user_location_data").value =
                    JSON.stringify({
                        "latitude": position.coords.latitude,
                        "longitude": position.coords.longitude,
                        "accuracy": position.coords.accuracy,
                        "timestamp": new Date().toISOString()
                    });
            }, function(error) {
                alert("Error getting location: " + error.message);
            }, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            });
        } else {
            alert("Geolocation is not supported by this browser.");
        }
    }
    window.onload = getLocation;
    </script>
    """
    return geolocation_script

# ─── Fetch Places from Overpass API ────────────────────────────────────
def fetch_nearby_places(lat, lon, place_type, radius=2000):
    """Fetch nearby places using Overpass API (OpenStreetMap)"""

    # Overpass API query
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Build query based on place type
    query_map = {
        "dentist": "amenity=dentist",
        "restaurant": "amenity=restaurant",
        "hospital": "amenity=hospital",
        "pharmacy": "amenity=pharmacy",
        "hotel": "tourism=hotel",
        "cafe": "amenity=cafe",
        "bank": "amenity=bank",
        "atm": "amenity=atm",
        "police": "amenity=police",
        "fire_station": "amenity=fire_station",
        "parking": "amenity=parking",
        "gas_station": "amenity=fuel",
        "supermarket": "shop=supermarket",
        "grocery": "shop=grocery",
        "clinic": "amenity=clinic",
        "doctor": "amenity=doctors",
    }

    query_tag = query_map.get(place_type.lower(), f"amenity={place_type.lower()}")

    # Calculate bounding box (in degrees, rough conversion)
    lat_offset = radius / 111000  # 1 degree latitude ≈ 111 km
    lon_offset = radius / (111000 * math.cos(math.radians(lat)))  # Adjust for latitude

    south = lat - lat_offset
    west = lon - lon_offset
    north = lat + lat_offset
    east = lon + lon_offset

    # Overpass QL query with simpler format
    query = f"""[out:json];
(
  node["{query_tag.split('=')[0]}"{query_tag.split('=')[1]}]({south},{west},{north},{east});
  way["{query_tag.split('=')[0]}"{query_tag.split('=')[1]}]({south},{west},{north},{east});
);
out center;"""

    try:
        with st.spinner("📍 Searching OpenStreetMap..."):
            response = requests.post(overpass_url, data=query, timeout=15)

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    st.warning("⚠️ OpenStreetMap API returned invalid data. Please try again in a moment.")
                    return None
            elif response.status_code == 429:
                st.warning("⚠️ OpenStreetMap API is busy. Please wait a moment and try again.")
                return None
            elif response.status_code == 400:
                st.error("❌ Invalid query. Please check your search parameters.")
                return None
            else:
                st.error(f"❌ API Error {response.status_code}. Please try again.")
                return None
    except requests.Timeout:
        st.error("❌ Request timed out. OpenStreetMap API is slow. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ─── Parse Overpass Response ───────────────────────────────────────────
def parse_overpass_places(data, user_lat, user_lon):
    """Parse Overpass API response and calculate distances"""
    places = []

    if not data or "elements" not in data:
        return places

    for element in data["elements"]:
        lat, lon = None, None
        name = element.get("tags", {}).get("name", "Unnamed Place")

        # Get coordinates
        if "lat" in element and "lon" in element:
            lat, lon = element["lat"], element["lon"]
        elif "center" in element:
            lat = element["center"]["lat"]
            lon = element["center"]["lon"]
        elif "geometry" in element and len(element["geometry"]) > 0:
            # For ways/relations, use first point
            lat = element["geometry"][0]["lat"]
            lon = element["geometry"][0]["lon"]

        if lat and lon:
            distance = haversine_distance(user_lat, user_lon, lat, lon)
            places.append({
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "distance_km": distance,
                "distance_m": distance * 1000,
                "type": element.get("type", "unknown")
            })

    # Sort by distance
    places.sort(key=lambda x: x["distance_km"])
    return places

# ─── Main App ──────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>📍 Nearby Places Finder</h1><p>Find places near you sorted by proximity</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="info-box"><strong>ℹ️ How it works:</strong><br>1. Allow location access<br>2. Enter place type (dentist, restaurant, etc.)<br>3. See results sorted by distance</div>', unsafe_allow_html=True)

# Initialize session state
if "user_location" not in st.session_state:
    st.session_state.user_location = None
if "places" not in st.session_state:
    st.session_state.places = []

# Location request button
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Get My Location", key="location_btn"):
        st.markdown(get_geolocation(), unsafe_allow_html=True)
        st.info("📍 Please allow location access when prompted by your browser")

# Manual location input (fallback)
with col2:
    st.subheader("Or enter location manually:")
    manual_lat = st.number_input("Latitude", value=0.0, format="%.6f")
    manual_lon = st.number_input("Longitude", value=0.0, format="%.6f")

    if st.button("Use Manual Location"):
        st.session_state.user_location = {
            "latitude": manual_lat,
            "longitude": manual_lon,
            "accuracy": None,
            "timestamp": datetime.now().isoformat()
        }

# Place type search
st.subheader("🔍 Search for Places")

place_categories = [
    "dentist", "restaurant", "hospital", "pharmacy", "hotel",
    "cafe", "bank", "atm", "police", "fire_station",
    "parking", "gas_station", "supermarket", "grocery", "clinic", "doctor"
]

col1, col2 = st.columns([2, 1])

with col1:
    place_type = st.selectbox(
        "What are you looking for?",
        place_categories,
        index=0,
        key="place_type"
    )

with col2:
    search_radius = st.slider(
        "Search radius (meters)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500
    )

# Search button
if st.button("🔎 Find Nearby Places", key="search_btn"):
    if st.session_state.user_location:
        lat = st.session_state.user_location["latitude"]
        lon = st.session_state.user_location["longitude"]

        with st.spinner(f"Searching for nearby {place_type}s..."):
            data = fetch_nearby_places(lat, lon, place_type, search_radius)
            if data:
                st.session_state.places = parse_overpass_places(data, lat, lon)
    else:
        st.warning("⚠️ Please get your location first (allow browser access or enter manually)")

# Display results
if st.session_state.places:
    places = st.session_state.places

    st.subheader(f"📌 Found {len(places)} nearby {place_type}s")

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nearest", f"{places[0]['distance_m']:.0f}m" if places[0]['distance_m'] < 1000 else f"{places[0]['distance_km']:.2f}km")
    with col2:
        avg_distance = sum(p["distance_km"] for p in places) / len(places)
        st.metric("Average Distance", f"{avg_distance:.2f}km")
    with col3:
        st.metric("Total Found", len(places))

    # Create map
    st.subheader("🗺️ Map View")
    user_lat = st.session_state.user_location["latitude"]
    user_lon = st.session_state.user_location["longitude"]

    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Add user location
    folium.Marker(
        [user_lat, user_lon],
        popup="📍 Your Location",
        icon=folium.Icon(color="blue", icon="person"),
        tooltip="You are here"
    ).add_to(m)

    # Add nearby places
    colors = ["red", "orange", "green", "purple", "darkred", "lightred", "gray", "black"]
    for idx, place in enumerate(places[:20]):  # Show top 20 on map
        color = colors[idx % len(colors)]
        folium.Marker(
            [place["latitude"], place["longitude"]],
            popup=f"{place['name']}<br>Distance: {place['distance_m']:.0f}m",
            icon=folium.Icon(color=color, icon="info-sign"),
            tooltip=f"#{idx+1}: {place['name']} ({place['distance_m']:.0f}m)"
        ).add_to(m)

    st_folium(m, width=1200, height=500)

    # Results table
    st.subheader("📋 Results (Sorted by Distance)")

    # Display top results
    for idx, place in enumerate(places[:30], 1):
        distance_str = f"{place['distance_m']:.0f}m" if place['distance_m'] < 1000 else f"{place['distance_km']:.2f}km"

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**#{idx} {place['name']}**")
        with col2:
            st.markdown(f"**{distance_str}**")
        with col3:
            st.markdown(f"[📍 View on Map](https://maps.google.com/?q={place['latitude']},{place['longitude']})")

    # Expandable details
    with st.expander(f"Show all {len(places)} results"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Name": p["name"],
                "Distance (km)": f"{p['distance_km']:.3f}",
                "Distance (m)": f"{p['distance_m']:.0f}"
            }
            for idx, p in enumerate(places)
        ])
        st.dataframe(df, use_container_width=True)
else:
    if st.session_state.user_location:
        st.info("🔍 Click 'Find Nearby Places' to search")
    else:
        st.warning("⚠️ Please allow location access first or enter your location manually")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>Nearby Places Finder - Pure proximity-based search, no algorithm bias</p>
    <p>Data powered by OpenStreetMap | Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
</div>
""", unsafe_allow_html=True)
