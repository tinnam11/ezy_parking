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