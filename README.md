# Ezy Parking

**Seattle Parking Analysis — Interactive Map**

The Team:

- **Aditee Elkunchwar**
- **Andrew Frederick**
- **Ken Thuleeratanarom**
- **Mimi Richert**
- **Tinna Mokaramanee**

This project is an interactive WebGIS application designed to help users explore Seattle’s on-street and off-street parking through a clean, modern, and accessible interface.
It uses Mapbox GL JS, custom controls, live filters, geolocation, and search tools to let users locate parking quickly and intuitively.

## 🚗 Project Overview

The Seattle Parking Map displays parking garages, street parking blockfaces, overview points, and restricted permit zones.
Users can:

- Search for an address

- Sort nearby parking by distance or price

- Filter street parking by maximum price

- Filter availability by time of day

- Show or hide individual data layers

- Center the map on their current location

- Use custom zoom and legend toggle controls

**Built for GEOG 328: Web GIS, this project highlights practical spatial analysis and UI/UX design for geospatial applications.**

## Project Goals

This project aims to visualize Seattle's parking availability, and provide meaningful insights for residents, city planners and tourists. By leveraging geospatial data, we strive to enhance urban mobility and contribute to more efficient parking management in Seattle, as well as improve parking accessibility for all users.

## Application URL

**Live Map:**
https://tinnam11.github.io/ezy_parking/

![Home Page](<img/Screenshot 2025-12-03 at 8.37.35 PM.png>)
![Map2](<img/Screenshot 2025-12-03 at 8.37.42 PM.png>)
![Map1](<img/Screenshot 2025-12-03 at 8.37.56 PM.png>)

## 🗺️ Features

**1. Search & Directions Support**

Search any address or place (e.g., 400 Pike St, Seattle)

View and sort nearby parking

**2. Multiple Parking Layers**

Each layer has a unique symbol and toggle:

- Garages - Blue points

- Street parking blockfaces — line features (green)

- Overview parking points — point data (red)

- Restricted / permit areas — purple, indicating limited access

**3. Filters**

- Price slider: show only street parking at or below a user-selected hourly price

- Time filters: filter by a specific time range

- Sort options: sort by distance or price for quick comparison

**4. Interactive Map Controls**

- Zoom buttons (+ / —)

- Geolocation button (“My location”)

- “Toggle Legend” to show or hide the on-map instructions

- Smooth panning and popups for feature details

## Data Source

This project uses publicly available dataset, including:

Parking Categories
- https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::parking-categories/about

Parking Tiers
- https://data-seattlecitygis.opendata.arcgis.com/datasets/4159ee278747484ab456576198b54263_0/explore?location=47.619662%2C-122.337296%2C14.99

Restricted Parking Zone Areas
- https://data-seattlecitygis.opendata.arcgis.com/maps/f6f00c5cf9634d578d41156bf4a4679f/about

Public Garage and Paring Lot
- https://data-seattlecitygis.opendata.arcgis.com/datasets/3029d63401544cd6b9783ef1bfb40b91_1/explore

## Applied Libraries & Web Services

This project uses a combination of Python-based geospatial libraries for data preparation and modern web technologies for interactive map rendering and deployment.

### Python Libraries (Data Processing & Preparation)

**GeoPandas**
Python library for geospatial data processing
Built on top of Pandas and Shapely
Used for: reading/writing GeoJSON, spatial operations, coordinate transformations, data filtering

**Pandas**
Library for data manipulation and analysis
Used for: attribute cleaning, field standardization, data type conversions, and filtering

**Shapely**
Geometric operations library powered by GEOS
Used for: geometry validation, spatial relationships, topology repair

**NumPy**
Numerical computing library and a core dependency of GeoPandas
Used for: grid generation and spatial aggregation calculations

**Regular Expressions (re)**
Python module for pattern matching and text parsing
Used for: parsing time limits, cleaning text data, extracting numeric fields

### Web Libraries & Services (Interactive Map & Deployment)

**Mapbox GL JS**
Javascript library for interactive web maps and vector tile rendering
Used for: map visualization, styling layers, loading GeoJSON sources

**Mapbox Geocoding API**
Used for: location search and address lookup

**JavaScript**
Used for: dynamic data loading, filter logic, UI controls, popup creation

**HTML / CSS**
Used for: structuring the webpage and styling the interface

**GitHub Pages**
Used for: hosting and deploying the live web application

## Acknowledgement

We thank:

**University of Washington** – GEOG 328: Web GIS for instruction, guidance, and applied learning opportunities

The **City of Seattle / Seattle Open Data Portal**, for providing parking datasets

**Mapbox** for providing accessible, powerful WebGIS tools

Special thanks to our **professor** and **peers** for their feedback throughout development.

## AI Use Disclosure

ChatGPT was used to:

- Improve readability and formatting of the README
- Suggest structure and wording
- Help refine explanations
- Provide grammar corrections and organization

All code was reviewed, adapted, and integrated manually by the team. AI was used as a support tool only, and the final implementation reflects the team’s decisions and edits.

## Additional Notes

- This application is intended for educational and research purposes, not real-time parking enforcement.

- Users should verify parking signs on-site when in doubt.

- The datasets used may not reflect real-time availability.