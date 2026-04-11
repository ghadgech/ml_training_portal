# 📍 Nearby Places Finder

A proximity-based place search app that finds nearby services sorted **purely by distance**, without algorithm bias, sponsorship influence, or rating manipulation.

## Problem Solved

Google's place search shows results based on:
- Sponsorship/ads
- Star ratings
- Review count
- Algorithm preferences

**Result:** A dentist 200m away might be buried behind a dentist 20km away with more reviews.

This app shows you the **closest places first**, period.

## Features

✅ **Real-Time Location Tracking**
- Browser geolocation API integration
- GPS accuracy display
- Manual location input fallback

✅ **Distance-Based Ranking**
- Pure proximity sorting (nearest first)
- Haversine formula for accurate distance calculation
- Shows distance in meters or kilometers

✅ **Interactive Map**
- OpenStreetMap visualization
- Color-coded markers for top results
- Direct links to Google Maps

✅ **Multiple Categories**
- Dentist
- Restaurant
- Hospital
- Pharmacy
- Hotel
- Cafe
- Bank
- ATM
- Police
- Fire Station
- Parking
- Gas Station
- Supermarket
- Grocery
- Clinic
- Doctor

✅ **Customizable Search Radius**
- Adjustable from 500m to 5km
- Responsive to user needs

## How It Works

### Architecture
1. **Frontend:** Streamlit UI with real-time geolocation
2. **Location API:** Browser native Geolocation API (no external service)
3. **Places Data:** OpenStreetMap (Overpass API)
4. **Distance Calculation:** Haversine formula for accuracy
5. **Sorting:** Pure distance-based ranking

### Data Flow
```
User Location (GPS) 
    ↓
Place Type Selection
    ↓
Overpass API Query (OpenStreetMap)
    ↓
Distance Calculation
    ↓
Sort by Distance (Ascending)
    ↓
Display Results + Map
```

## Setup & Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Required Packages
- streamlit
- folium
- streamlit-folium
- requests
- geopy

## Running the App

```bash
streamlit run nearby_places_finder.py
```

The app will open at `http://localhost:8501`

## Usage

### Step 1: Allow Location Access
Click **"🔄 Get My Location"** and allow browser location access when prompted.

**OR**

Manually enter your latitude and longitude, then click **"Use Manual Location"**

### Step 2: Select Place Type
Choose from the dropdown menu (dentist, restaurant, hotel, etc.)

### Step 3: Adjust Search Radius
Use the slider to set search area (default: 2km)

### Step 4: Search
Click **"🔎 Find Nearby Places"** to see results

### Results Display
- **Statistics:** Nearest distance, average distance, total count
- **Interactive Map:** Visual representation of your location and nearby places
- **Sorted Table:** All results ranked by distance (closest first)

## Business Strategy

### Target Market: Small Local Businesses
- **Problem:** Local businesses get buried by Google's algorithm
- **Solution:** Proximity-first app shows them to nearby users
- **Revenue:** Premium listings, analytics dashboard, commission model

### Go-to-Market
1. Identify local businesses in specific area
2. Send outreach emails: "You rank #1 on our app but #15 on Google"
3. Show proof with data
4. Offer free/premium listing tiers
5. Scale to other cities

## Technical Details

### Geolocation
- Uses browser's Geolocation API (W3C standard)
- Requires HTTPS in production
- High accuracy mode enabled
- Timeout: 10 seconds

### Distance Calculation
- Formula: Haversine formula
- Accuracy: ±0.5% for typical distances
- Earth radius: 6371 km

### Place Data
- Source: OpenStreetMap (Overpass API)
- No API key required (free)
- Updated regularly by community
- No sponsored results

### Search Radius
- Minimum: 500 meters
- Maximum: 5 kilometers
- Default: 2 kilometers
- User adjustable

## Privacy & Data

✅ **No Tracking:** Only uses browser geolocation
✅ **No Storage:** Doesn't store user location
✅ **No Ads:** No sponsored results
✅ **No Cookies:** Minimal tracking
✅ **Open Data:** Uses OpenStreetMap (community data)

## Future Enhancements

- [ ] Real-time updates as user drives
- [ ] Category filters (rating, opening hours, wheelchair accessible)
- [ ] Business listing dashboard
- [ ] Email outreach system
- [ ] User reviews (organic, non-manipulated)
- [ ] Offline mode
- [ ] Mobile app version
- [ ] Multiple location comparison

## API Documentation

### Overpass API
- **Endpoint:** https://overpass-api.de/api/interpreter
- **Query Language:** Overpass QL
- **Timeout:** 10 seconds
- **Rate Limit:** Generally free, but high-volume queries may be throttled

### Browser Geolocation API
- **Method:** navigator.geolocation.getCurrentPosition()
- **Permission:** Required (user must allow)
- **Accuracy:** Device dependent (5-50m typical)
- **Privacy:** Processed locally, not stored

## Troubleshooting

### Issue: "Please allow location access"
**Solution:** 
- Check browser privacy settings
- Try manual entry (latitude/longitude)
- Use HTTPS in production

### Issue: No results found
**Solution:**
- Increase search radius
- Check place type spelling
- Verify location is correct
- Try different category

### Issue: Results are inaccurate
**Solution:**
- OpenStreetMap data depends on community contributions
- Some areas may have limited data
- Report missing places on openstreetmap.org

## License

MIT License - Free to use and modify

## Support

For issues or feature requests, contact development team.

---

**Built with:** Streamlit, OpenStreetMap, Folium
**Status:** MVP - Production Ready
**Last Updated:** 2026-04-11
