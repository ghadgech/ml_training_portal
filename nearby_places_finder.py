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

# ─── Demo Data for Testing ────────────────────────────────────────────
def get_demo_places(lat, lon, place_type):
    """Return category-specific demo data for testing the app"""
    demo_data = {
        "dentist": [
            {"name": "Smile Dental Clinic", "lat": lat + 0.002, "lon": lon + 0.002},
            {"name": "Perfect Teeth Center", "lat": lat - 0.001, "lon": lon + 0.003},
            {"name": "Bright Smile Dentistry", "lat": lat + 0.004, "lon": lon - 0.002},
            {"name": "Dr. Smith Dental", "lat": lat - 0.003, "lon": lon - 0.001},
            {"name": "Modern Dental Clinic", "lat": lat + 0.005, "lon": lon + 0.004},
        ],
        "restaurant": [
            {"name": "Burger Palace", "lat": lat + 0.001, "lon": lon + 0.001},
            {"name": "Italian Bistro", "lat": lat - 0.002, "lon": lon + 0.002},
            {"name": "Asian Fusion", "lat": lat + 0.003, "lon": lon - 0.001},
            {"name": "Pizza House", "lat": lat - 0.001, "lon": lon - 0.003},
            {"name": "Sushi Bar", "lat": lat + 0.004, "lon": lon + 0.002},
        ],
        "hotel": [
            {"name": "Grand Hotel", "lat": lat + 0.005, "lon": lon + 0.003},
            {"name": "Sunset Inn", "lat": lat - 0.004, "lon": lon + 0.002},
            {"name": "Central Plaza Hotel", "lat": lat + 0.001, "lon": lon - 0.004},
            {"name": "Luxury Suites", "lat": lat - 0.002, "lon": lon - 0.001},
            {"name": "Budget Motel", "lat": lat + 0.003, "lon": lon + 0.001},
        ],
        "hospital": [
            {"name": "City General Hospital", "lat": lat + 0.005, "lon": lon + 0.003},
            {"name": "St. Mary's Medical Center", "lat": lat - 0.004, "lon": lon + 0.001},
            {"name": "Emergency Care Hospital", "lat": lat + 0.002, "lon": lon - 0.004},
            {"name": "Riverside Medical", "lat": lat + 0.001, "lon": lon + 0.005},
            {"name": "Veterans Hospital", "lat": lat - 0.003, "lon": lon - 0.003},
        ],
        "pharmacy": [
            {"name": "MediCare Pharmacy", "lat": lat + 0.002, "lon": lon + 0.001},
            {"name": "Quick Pharmacy", "lat": lat - 0.001, "lon": lon + 0.003},
            {"name": "Health Plus Pharmacy", "lat": lat + 0.003, "lon": lon - 0.002},
            {"name": "24hr Pharmacy", "lat": lat - 0.002, "lon": lon - 0.001},
            {"name": "Community Drug Store", "lat": lat + 0.004, "lon": lon + 0.002},
        ],
        "cafe": [
            {"name": "Coffee Corner", "lat": lat + 0.001, "lon": lon + 0.002},
            {"name": "Brew Haven", "lat": lat - 0.001, "lon": lon + 0.001},
            {"name": "The Daily Cafe", "lat": lat + 0.002, "lon": lon - 0.001},
            {"name": "Morning Glory", "lat": lat - 0.003, "lon": lon + 0.002},
            {"name": "Bean There Coffee", "lat": lat + 0.003, "lon": lon + 0.003},
        ],
        "bank": [
            {"name": "First National Bank", "lat": lat + 0.002, "lon": lon + 0.001},
            {"name": "City Bank", "lat": lat - 0.001, "lon": lon + 0.002},
            {"name": "Trust Bank", "lat": lat + 0.003, "lon": lon - 0.001},
            {"name": "Capital Bank", "lat": lat - 0.002, "lon": lon - 0.002},
            {"name": "Community Bank", "lat": lat + 0.001, "lon": lon + 0.003},
        ],
        "atm": [
            {"name": "ATM - First National Bank", "lat": lat + 0.001, "lon": lon + 0.001},
            {"name": "ATM - City Center", "lat": lat - 0.002, "lon": lon + 0.002},
            {"name": "ATM - Mall Entrance", "lat": lat + 0.003, "lon": lon - 0.001},
            {"name": "ATM - Station Square", "lat": lat - 0.001, "lon": lon - 0.003},
            {"name": "ATM - Shopping Complex", "lat": lat + 0.004, "lon": lon + 0.002},
        ],
        "police": [
            {"name": "Police Station - Downtown", "lat": lat + 0.005, "lon": lon + 0.003},
            {"name": "Police Outpost - North", "lat": lat - 0.004, "lon": lon + 0.001},
            {"name": "Traffic Police - Main Road", "lat": lat + 0.002, "lon": lon - 0.004},
            {"name": "Police Station - East", "lat": lat - 0.001, "lon": lon + 0.005},
            {"name": "Police Beat - Market Area", "lat": lat + 0.003, "lon": lon - 0.002},
        ],
        "fire_station": [
            {"name": "Fire Station - Central", "lat": lat + 0.004, "lon": lon + 0.002},
            {"name": "Fire Department - North", "lat": lat - 0.003, "lon": lon + 0.003},
            {"name": "Fire Rescue - South", "lat": lat + 0.001, "lon": lon - 0.003},
            {"name": "Fire Station - East", "lat": lat - 0.002, "lon": lon - 0.001},
            {"name": "Emergency Fire - West", "lat": lat + 0.005, "lon": lon + 0.001},
        ],
        "supermarket": [
            {"name": "Super Mart - Main", "lat": lat + 0.002, "lon": lon + 0.002},
            {"name": "Great Groceries", "lat": lat - 0.001, "lon": lon + 0.001},
            {"name": "Mega Supermarket", "lat": lat + 0.003, "lon": lon - 0.002},
            {"name": "Price World", "lat": lat - 0.002, "lon": lon - 0.001},
            {"name": "Family Mart", "lat": lat + 0.001, "lon": lon + 0.003},
        ],
    }

    return demo_data.get(place_type.lower(), demo_data["restaurant"])

# ─── Reverse Geocoding - Get place name from coordinates ────────────────
def get_place_name(lat, lon):
    """Get place name from coordinates using Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'NearbyPlacesFinder/1.0'}
        response = requests.get(url, timeout=5, headers=headers)

        if response.status_code == 200:
            data = response.json()
            # Try to get address name
            if "address" in data:
                address = data["address"]
                # Priority: city > town > village > county
                for key in ["city", "town", "village", "county"]:
                    if key in address:
                        return address[key]
            elif "name" in data:
                return data["name"]
    except:
        pass

    return None

# ─── Fetch Places from Google Places API ──────────────────────────────
def fetch_google_places(lat, lon, place_type, radius=2000, api_key=None):
    """Fetch nearby places from Google Places API"""

    if not api_key:
        return None

    # Map place types to Google Places types
    google_type_map = {
        "dentist": "dentist",
        "restaurant": "restaurant",
        "hospital": "hospital",
        "pharmacy": "pharmacy",
        "hotel": "lodging",
        "cafe": "cafe",
        "bank": "bank",
        "atm": "atm",
        "police": "police",
        "fire_station": "fire_station",
        "parking": "parking_lot",
        "gas_station": "gas_station",
        "supermarket": "supermarket",
        "grocery": "grocery_or_supermarket",
        "clinic": "doctor",
        "doctor": "doctor",
    }

    google_type = google_type_map.get(place_type.lower(), "restaurant")

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": google_type,
        "key": api_key
    }

    try:
        with st.spinner("📍 Searching Google Places..."):
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data["status"] == "OK":
                    st.success("✅ Using real data from Google Places")
                    return data
                elif data["status"] == "ZERO_RESULTS":
                    st.info("ℹ️ No places found in this area")
                    return {"results": []}
                elif data["status"] == "REQUEST_DENIED":
                    st.error("❌ Invalid API key")
                    return None
                elif data["status"] == "OVER_QUERY_LIMIT":
                    st.warning("⚠️ API quota exceeded. Try again later.")
                    return None
                else:
                    st.warning(f"⚠️ API Status: {data['status']}")
                    return None
            else:
                st.error(f"❌ API Error {response.status_code}")
                return None

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ─── Parse Google Places Response ──────────────────────────────────────
def parse_google_places(data, user_lat, user_lon):
    """Parse Google Places API response and sort by distance"""
    places = []

    if not data or "results" not in data:
        return places

    for place in data["results"]:
        location = place.get("geometry", {}).get("location", {})
        lat = location.get("lat")
        lon = location.get("lng")
        name = place.get("name", "Unnamed Place")

        if lat and lon:
            distance = haversine_distance(user_lat, user_lon, lat, lon)
            places.append({
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "distance_km": distance,
                "distance_m": distance * 1000,
                "type": "google_place",
                "rating": place.get("rating", 0),
                "open_now": place.get("opening_hours", {}).get("open_now"),
            })

    # Sort by distance (proximity first)
    places.sort(key=lambda x: x["distance_km"])
    return places
    """Fetch nearby places using Overpass API with demo fallback"""

    overpass_url = "https://overpass-api.de/api/interpreter"

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

    key, value = query_map.get(place_type.lower(), "amenity=" + place_type.lower()).split("=")

    # Calculate bounding box
    lat_offset = radius / 111000
    lon_offset = radius / (111000 * max(math.cos(math.radians(lat)), 0.1))

    bbox = f"{lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset}"

    # Simple Overpass query
    query = f"""
    [bbox:{bbox}];
    (
      node[{key}={value}];
      way[{key}={value}];
      relation[{key}={value}];
    );
    out center;
    """

    try:
        with st.spinner("📍 Searching for nearby places..."):
            headers = {'User-Agent': 'NearbyPlacesFinder/1.0'}
            response = requests.post(
                overpass_url,
                data=query,
                timeout=20,
                headers=headers
            )

            if response.status_code == 200 and response.text.strip():
                try:
                    data = response.json()
                    if data and "elements" in data and len(data["elements"]) > 0:
                        st.success("✅ Found places from OpenStreetMap")
                        return data
                except json.JSONDecodeError:
                    pass

        # Fallback to demo data
        st.warning("⚠️ Using demo data (OpenStreetMap API unavailable). To use real data, add your Google Places API key.")
        demo_places = get_demo_places(lat, lon, place_type)
        return {
            "elements": [
                {
                    "type": "node",
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "tags": {"name": p["name"]}
                }
                for p in demo_places
            ]
        }

    except Exception as e:
        st.warning(f"⚠️ Using demo data. Error: {str(e)[:50]}")
        demo_places = get_demo_places(lat, lon, place_type)
        return {
            "elements": [
                {
                    "type": "node",
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "tags": {"name": p["name"]}
                }
                for p in demo_places
            ]
        }

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

# ─── Sidebar - Configuration ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### 📍 Data Source")
    st.info("""
    **Current:** Demo data (sample places)

    To use **real places from Google Maps**, you need a Google Places API key.
    """)

    google_api_key = st.text_input(
        "Google Places API Key (optional)",
        type="password",
        help="Get a free key from https://cloud.google.com/docs/authentication/api-keys"
    )

    if google_api_key:
        st.success("✅ Google Places API key configured")
    else:
        st.markdown("""
        **How to get a free Google Places API key:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project
        3. Enable "Places API"
        4. Create an API key
        5. Paste it above

        **Cost:** Free tier includes $200/month credit
        """)

    st.markdown("---")
    st.markdown("### 📌 About This App")
    st.markdown("""
    - Finds nearby places sorted by **distance only**
    - No algorithm bias or sponsored results
    - Works when driving or stationary
    - Updates as you move
    """)

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
    st.subheader("🌍 Quick Location Setup:")
    st.markdown("""
    **Option 1: Google Maps**
    1. Open Google Maps on your phone
    2. Tap your location (blue dot)
    3. Copy the coordinates
    4. Paste below

    **Option 2: What3Words**
    - Go to what3words.com
    - Enter your location
    - Convert to lat/lon
    """)

    if st.button("❓ What's my location?", key="help_loc_btn"):
        st.info("""
        **Find your current location:**
        1. Google Maps → Tap blue dot → Copy coordinates
        2. Or visit: https://www.latlong.net/ (shows your location)
        3. Or ask: "What is my location?" in Google Maps
        """)

# Manual location input (fallback)
with col2:
    st.subheader("📍 Or enter location manually:")
    col_lat, col_lon = st.columns(2)

    with col_lat:
        manual_lat = st.number_input("Latitude", value=0.0, format="%.6f", key="lat_input")

    with col_lon:
        manual_lon = st.number_input("Longitude", value=0.0, format="%.6f", key="lon_input")

    if st.button("✅ Set Location", key="manual_loc_btn"):
        st.session_state.user_location = {
            "latitude": manual_lat,
            "longitude": manual_lon,
            "accuracy": None,
            "timestamp": datetime.now().isoformat()
        }
        st.success(f"📍 Location set to: {manual_lat}, {manual_lon}")

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
        value=5000,
        step=500
    )

col3, _ = st.columns([1, 1])
with col3:
    results_count = st.number_input(
        "Results to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

# Search button
if st.button("🔎 Find Nearby Places", key="search_btn"):
    if st.session_state.user_location:
        lat = st.session_state.user_location["latitude"]
        lon = st.session_state.user_location["longitude"]

        # Try Google Places API first if key is available
        if google_api_key:
            data = fetch_google_places(lat, lon, place_type, search_radius, google_api_key)
            if data:
                st.session_state.places = parse_google_places(data, lat, lon)
        else:
            # Fallback to OpenStreetMap or demo data
            with st.spinner(f"Searching for nearby {place_type}s..."):
                data = fetch_nearby_places(lat, lon, place_type, search_radius)
                if data:
                    st.session_state.places = parse_overpass_places(data, lat, lon)
    else:
        st.warning("⚠️ Please enter your location first (Latitude & Longitude)")

# Display results
if st.session_state.places:
    places = st.session_state.places
    user_lat = st.session_state.user_location["latitude"]
    user_lon = st.session_state.user_location["longitude"]

    # Display search summary
    st.markdown("---")

    # Get place name from coordinates
    place_name = get_place_name(user_lat, user_lon)
    location_display = place_name if place_name else f"{user_lat:.4f}, {user_lon:.4f}"

    # Check if using demo data
    is_demo = False
    if len(places) > 0 and "Demo" in places[0].get('_source', ''):
        is_demo = True

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Your Location", location_display)
    with col2:
        st.metric("🔍 Searching For", place_type.title())
    with col3:
        st.metric("📌 Total Found", len(places))
    with col4:
        st.metric("⏮️ Closest", f"{places[0]['distance_m']:.0f}m" if places[0]['distance_m'] < 1000 else f"{places[0]['distance_km']:.2f}km")

    # Show data source indicator
    if is_demo:
        st.warning("⚠️ **DEMO DATA MODE** - Using sample places for demonstration. To see real places, add your Google Places API key in the sidebar.")
    else:
        st.success("✅ Using real OpenStreetMap data")

    st.markdown("---")

    # ─── TOP N RESULTS (Prominent Display) ───────────────────────
    st.subheader(f"🏆 Top {min(results_count, len(places))} Nearest {place_type.title()}s")

    # Display results in a scrollable container
    for idx, place in enumerate(places[:results_count], 1):
        distance_m = place['distance_m']
        distance_str = f"{distance_m:.0f}m" if distance_m < 1000 else f"{distance_m/1000:.2f}km"

        # Color code by distance
        if distance_m < 500:
            color = "🟢"  # Green - very close
        elif distance_m < 1000:
            color = "🟡"  # Yellow - close
        else:
            color = "🔴"  # Red - far

        st.markdown(f"""
        <div class="distance-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{color} #{idx} {place['name']}</h4>
                    <p style="margin: 0; color: #666;">Distance: <strong>{distance_str}</strong></p>
                </div>
                <div style="text-align: right;">
                    <a href="https://maps.google.com/?q={place['latitude']},{place['longitude']}" target="_blank" style="text-decoration: none; padding: 8px 16px; background: #667eea; color: white; border-radius: 5px;">View Map</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Calculate map limit
    map_limit = min(results_count + 10, len(places))  # Show a few more than selected

    # Create map
    st.subheader(f"🗺️ Map View (Top {map_limit} Results)")

    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # Add user location (center, blue)
    folium.Marker(
        [user_lat, user_lon],
        popup="📍 Your Location",
        icon=folium.Icon(color="blue", icon="person", prefix="fa"),
        tooltip="You are here"
    ).add_to(m)

    # Add nearby places (color coded by rank)
    colors = ["red", "orange", "green", "purple", "darkred", "lightred", "gray", "black", "blue", "pink"]

    for idx, place in enumerate(places[:map_limit]):
        color = colors[idx % len(colors)]
        distance_m = place['distance_m']
        distance_str = f"{distance_m:.0f}m" if distance_m < 1000 else f"{distance_m/1000:.2f}km"

        # Create simple text popup (HTML has issues in streamlit-folium)
        popup_text = f"""
#{idx+1} {place['name']}
Distance: {distance_str}
Rating: {place.get('rating', 'N/A')}
Lat: {place['latitude']:.4f}, Lon: {place['longitude']:.4f}
        """

        folium.Marker(
            [place["latitude"], place["longitude"]],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="utensils", prefix="fa"),
            tooltip=f"{place['name']} ({distance_str})"
        ).add_to(m)

    st_folium(m, width=1200, height=500)

    # All results in expandable section
    st.markdown("---")
    with st.expander(f"📋 View All {len(places)} Results"):
        st.subheader("Complete Results List (Sorted by Distance)")
        for idx, place in enumerate(places, 1):
            distance_m = place['distance_m']
            distance_str = f"{distance_m:.0f}m" if distance_m < 1000 else f"{distance_m/1000:.2f}km"
            st.write(f"**#{idx}. {place['name']}** - {distance_str}")


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
