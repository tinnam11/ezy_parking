mapboxgl.accessToken = 'pk.eyJ1IjoibXJpY2hlcnQiLCJhIjoiY21oNTl4NnlxMDVqdTJqb205aDFmcWU2ZSJ9.0TGc-97aSBJ8xmNjTjjbuw';

const map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/satellite-streets-v11',
  center: [-122.3321, 47.6062],
  zoom: 12
});

const pendingVisibility = {};
function applyPendingVisibility(layerId) {
  if (!(layerId in pendingVisibility)) return;
  const visible = pendingVisibility[layerId];
  if (map.getLayer(layerId)) {
    map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
    delete pendingVisibility[layerId];
  }
}
function setLayerVisibility(layerId, visible) {
  if (map.getLayer(layerId)) {
    map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
  } else {
    pendingVisibility[layerId] = visible;
  }
}

/**
 * Add a GeoJSON source and either a line layer or a circle layer depending
 * on the first feature geometry type.
 *
 * options: {
 *   sourceId, layerId, geojsonPath,
 *   circlePaint (for points), linePaint (for lines),
 *   minzoom, maxzoom
 * }
 */
async function addGeoJSONLayer(options) {
  const {
    sourceId,
    layerId,
    geojsonPath,
    circlePaint = {},
    linePaint = {},
    minzoom = 0,
    maxzoom = 24
  } = options;

  try {
    const resp = await fetch(geojsonPath);
    if (!resp.ok) throw new Error(`Failed to fetch ${geojsonPath}: ${resp.statusText}`);
    const data = await resp.json();
    // create or update source
    if (map.getSource(sourceId)) {
      map.getSource(sourceId).setData(data);
    } else {
      map.addSource(sourceId, { type: 'geojson', data });
    }

    // detect geometry type safely
    let geomType = 'Point';
    if (data.features && data.features.length > 0 && data.features[0].geometry && data.features[0].geometry.type) {
      geomType = data.features[0].geometry.type;
    }

    // If layer already exists, apply pending visibility and return
    if (map.getLayer(layerId)) {
      applyPendingVisibility(layerId);
      console.log(`Layer ${layerId} already exists.`);
      return data;
    }

    if (geomType === 'LineString' || geomType === 'MultiLineString') {
      // Default zoom-dependent line paint (can be overridden by passed linePaint)
      const defaultLinePaint = {
        // Thinner and more transparent at city zooms (e.g. zoom 10-12),
        // thicker and fully opaque when zoomed in (e.g. zoom 15-17).
        'line-color': '#10b981',
        'line-width': [
          'interpolate',
          ['linear'],
          ['zoom'],
          // at zoom 10 -> width 1
          10, 1,
          // at zoom 13 -> width 2.5
          13, 2.5,
          // at zoom 16 -> width 6
          16, 6
        ],
        'line-opacity': [
          'interpolate',
          ['linear'],
          ['zoom'],
          // at zoom 10 -> faint
          10, 0.35,
          // at zoom 13 -> medium
          13, 0.6,
          // at zoom 16 -> near fully opaque
          16, 0.95
        ],
        'line-gap-width': 0
      };

      map.addLayer({
        id: layerId,
        type: 'line',
        source: sourceId,
        minzoom,
        maxzoom,
        paint: Object.assign({}, defaultLinePaint, linePaint),
        layout: { visibility: 'visible', 'line-join': 'round', 'line-cap': 'round' }
      });
    } else {
      // Default circle paint for points
      const defaultCirclePaint = {
        'circle-radius': 5,
        'circle-color': '#3b82f6',
        'circle-stroke-width': 1.5,
        'circle-stroke-color': '#ffffff',
        'circle-opacity': 0.95
      };

      map.addLayer({
        id: layerId,
        type: 'circle',
        source: sourceId,
        minzoom,
        maxzoom,
        paint: Object.assign({}, defaultCirclePaint, circlePaint),
        layout: { visibility: 'visible' }
      });
    }

    applyPendingVisibility(layerId);
    console.log(`Added layer ${layerId} from ${geojsonPath} as ${geomType}`);
    return data;
  } catch (err) {
    console.error('Error adding layer', layerId, err);
    throw err;
  }
}

// price filter function 
function applyPriceRangeFilter() {
  const priceMin = parseFloat(document.getElementById('priceMin')?.value || 0);
  const priceMax = parseFloat(document.getElementById('priceMax')?.value || 10);
  
  const display = document.getElementById('priceDisplay');
  if (display) {
    display.textContent = `${priceMin} - ${priceMax}`;
  }

  const filterExpression = [
    'any',
    //include free parking (rate is 0 or doesn't exist)
    ['!', ['has', 'weekday_rate']],
    ['==', ['get', 'weekday_rate'], 0],
    //include parking within price range
    ['all',
      ['>=', ['get', 'weekday_rate'], priceMin],
      ['<=', ['get', 'weekday_rate'], priceMax]
    ]
  ];
  
  ['garage-points', 'street-parking-layer', 'parking-all-points', 'parking-restricted'].forEach(layerId => {
    if (map.getLayer(layerId)) {
      map.setFilter(layerId, filterExpression);
    }
  });
}

//search marker (global so we can remove it later)
let searchMarker = null;

//search and show nearby parking
async function searchAndFilterByDistance() {
  const searchInput = document.getElementById('searchInput').value.trim();
  if (!searchInput) {
    alert('Please enter an address or place');
    return;
  }
  
  const resultsDiv = document.getElementById('searchResults');
  resultsDiv.innerHTML = '<p>🔍 Searching...</p>';
  
  //use Mapbox Geocoding API
  const geocodeUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(searchInput)}.json?access_token=${mapboxgl.accessToken}&proximity=-122.3321,47.6062&limit=1`;
  
  try {
    const response = await fetch(geocodeUrl);
    const data = await response.json();
    
    if (data.features && data.features.length > 0) {
      const [lng, lat] = data.features[0].center;
      const placeName = data.features[0].place_name;
      
      // Fly to the location
      map.flyTo({ center: [lng, lat], zoom: 15, duration: 1500 });
      
      // Remove old marker if exists
      if (searchMarker) {
        searchMarker.remove();
      }
      
      //add a marker for the searched location
      searchMarker = new mapboxgl.Marker({ color: '#ef4444' })
        .setLngLat([lng, lat])
        .setPopup(new mapboxgl.Popup().setHTML(`<strong>Search Location</strong><br>${placeName}`))
        .addTo(map);
      
      calculateNearbyParking(lng, lat);
    } else {
      resultsDiv.innerHTML = '<p style="color:#ef4444">❌ Location not found. Try a different address.</p>';
    }
  } catch (error) {
    console.error('Geocoding error:', error);
    resultsDiv.innerHTML = '<p style="color:#ef4444">❌ Error searching for location. Please try again.</p>';
  }
}

//calculate distance 
function calculateDistance(lon1, lat1, lon2, lat2) {
  const R = 3958.8; // Earth's radius in miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c; 
}

//display nearby parking
function calculateNearbyParking(searchLng, searchLat) {
  const results = [];
  
  //all parking features from all sources
  const sources = [
    { id: 'garages-src', type: 'Garage' },
    { id: 'parkingpoints-src', type: 'Street' },
    { id: 'restricted-src', type: 'Restricted' }
  ];
  
  sources.forEach(({ id, type }) => {
    const source = map.getSource(id);
    if (source && source._data && source._data.features) {
      source._data.features.forEach(feature => {
        let coords;
        if (feature.geometry.type === 'Point') {
          coords = feature.geometry.coordinates;
        } else if (feature.geometry.type === 'LineString') {
          coords = feature.geometry.coordinates[0];
        }
        
        if (coords) {
          const distance = calculateDistance(searchLng, searchLat, coords[0], coords[1]);
          
          //only include parking within 1.5 miles
          if (distance <= 1.5) {
            results.push({
              name: feature.properties.name || feature.properties.facility || feature.properties.category || type + ' Parking',
              distance: distance,
              rate: feature.properties.weekday_rate || null,
              spaces: feature.properties.total_spaces || feature.properties.capacity || '?',
              coordinates: coords,
              type: type,
              properties: feature.properties
            });
          }
        }
      });
    }
  });
  
  //sort by distance or price based on sortSelect
  const sortBy = document.getElementById('sortSelect')?.value || 'distance';
  results.sort((a, b) => {
    if (sortBy === 'distance') {
      return a.distance - b.distance;
    } else {
      const priceA = typeof a.rate === 'number' ? a.rate : 999;
      const priceB = typeof b.rate === 'number' ? b.rate : 999;
      return priceA - priceB;
    }
  });
  
  displaySearchResults(results.slice(0, 15)); //show top 15
}

function displaySearchResults(results) {
  const resultsDiv = document.getElementById('searchResults');
  
  if (results.length === 0) {
    resultsDiv.innerHTML = '<p style="color:#64748b">No parking found within 2miles of this location.</p>';
    return;
  }
  
  const html = `
    <div style="margin-bottom:0.5rem;font-weight:600;color:#0f172a">
      Found ${results.length} parking options nearby:
    </div>
  ` + results.map((r, i) => {
    const rateDisplay = typeof r.rate === 'number' ? `${r.rate.toFixed(2)}/hr` : 'Free';
    const typeColor = r.type === 'Garage' ? '#3b82f6' : r.type === 'Restricted' ? '#7c3aed' : '#f97316';
    
    return `
      <div class="result-item" style="padding:0.6rem;border-bottom:1px solid #e2e8f0;cursor:pointer;border-left:3px solid ${typeColor}" 
           onclick="map.flyTo({center:[${r.coordinates[0]},${r.coordinates[1]}],zoom:17}); new mapboxgl.Popup().setLngLat([${r.coordinates[0]},${r.coordinates[1]}]).setHTML('<strong>${r.name}</strong><div>Distance: ${r.distance.toFixed(2)} miles<br>Rate: ${rateDisplay}<br>Spaces: ${r.spaces}</div>').addTo(map);">
        <div style="display:flex;justify-content:space-between;align-items:start">
          <strong style="font-size:0.9rem">${i + 1}. ${r.name}</strong>
          <span style="font-size:0.8rem;color:#10b981;font-weight:600">${rateDisplay}</span>
        </div>
        <div style="font-size:0.85rem;color:#64748b;margin-top:0.2rem">
          📍 ${r.distance.toFixed(2)} miles away • ${r.type} • ${r.spaces} spaces
        </div>
      </div>
    `;
  }).join('');
  
  resultsDiv.innerHTML = html;
}

map.on('load', async () => {
  // Garages (points)
  await addGeoJSONLayer({
    sourceId: 'garages-src',
    layerId: 'garage-points',
    geojsonPath: 'assets/garages_clean.geojson',
    circlePaint: {
      'circle-color': '#3b82f6',
      'circle-radius': 5,
      'circle-stroke-width': 2,
      'circle-stroke-color': '#ffffff'
    }
  });

  // street_parking_detailed: will render as line if blockfaces are lines
  await addGeoJSONLayer({
    sourceId: 'streetparking-src',
    layerId: 'street-parking-layer',
    geojsonPath: 'assets/street_parking_detailed.geojson',
    linePaint: {
      // you can override color here if you want different color than default
      // 'line-color': '#10b981'
    }
  });

  // parking_all_points (overview points - all types)
  await addGeoJSONLayer({
    sourceId: 'parkingpoints-src',
    layerId: 'parking-all-points',
    geojsonPath: 'assets/parking_all_points.geojson',
    circlePaint: {
      'circle-color': '#f97316',
      'circle-radius': 4,
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff'
    }
  });

  // parking_restricted (restricted/permit zones only)
  await addGeoJSONLayer({
    sourceId: 'restricted-src',
    layerId: 'parking-restricted',
    geojsonPath: 'assets/parking_restricted.geojson',
    circlePaint: {
      'circle-color': '#7c3aed',
      'circle-radius': 5,
      'circle-stroke-width': 1.5,
      'circle-stroke-color': '#ffffff'
    }
  });

  
  const initVisibility = [
    { checkboxId: 'toggleGarages', layerId: 'garage-points' },
    { checkboxId: 'toggleBlockface', layerId: 'street-parking-layer' },
    { checkboxId: 'togglePoints', layerId: 'parking-all-points' },
    { checkboxId: 'toggleRestricted', layerId: 'parking-restricted' }
  ];
  initVisibility.forEach(({ checkboxId, layerId }) => {
    const el = document.getElementById(checkboxId);
    const checked = el ? el.checked : true;
    setLayerVisibility(layerId, checked);
  });

  // Wire up toggles
  initVisibility.forEach(({ checkboxId, layerId }) => {
    const el = document.getElementById(checkboxId);
    if (!el) return;
    el.addEventListener('change', (evt) => {
      setLayerVisibility(layerId, evt.target.checked);
    });
  });

  // Wire up search functionality
  const btnSearch = document.getElementById('btnSearch');
  const searchInput = document.getElementById('searchInput');
  
  if (btnSearch) {
    btnSearch.addEventListener('click', searchAndFilterByDistance);
  }
  
  if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        searchAndFilterByDistance();
      }
    });
  }
  
  // Wire up sort selector
  const sortSelect = document.getElementById('sortSelect');
  if (sortSelect) {
    sortSelect.addEventListener('change', () => {
      // Re-run search if we have a marker (meaning search was performed)
      if (searchMarker) {
        const center = searchMarker.getLngLat();
        calculateNearbyParking(center.lng, center.lat);
      }
    });
  }

  // wire for price range filter inputs
  const priceMin = document.getElementById('priceMin');
  const priceMax = document.getElementById('priceMax');
  const priceRange = document.getElementById('priceRange');

  if (priceMin && priceMax) {
    // Min/max inputs
    priceMin.addEventListener('input', applyPriceRangeFilter);
    priceMax.addEventListener('input', applyPriceRangeFilter);
  }

  if (priceRange) {
    // Simple slider (controls max price)
    priceRange.addEventListener('input', (e) => {
      const maxPrice = parseFloat(e.target.value);
      const display = document.getElementById('priceDisplay');
      if (display) display.textContent = maxPrice.toFixed(1);
      
      // Filter: show everything from $0 to selected max
      const filterExpression = [
        'any',
        ['!', ['has', 'weekday_rate']],
        ['==', ['get', 'weekday_rate'], 0],
        ['<=', ['get', 'weekday_rate'], maxPrice]
      ];
      
      ['garage-points', 'street-parking-layer', 'parking-all-points', 'parking-restricted'].forEach(layerId => {
        if (map.getLayer(layerId)) {
          map.setFilter(layerId, filterExpression);
        }
      });
    });
  }

  // Zoom buttons
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  if (btnZoomIn) btnZoomIn.addEventListener('click', () => map.zoomIn());
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => map.zoomOut());

  // Geolocate button
  const btnGeolocate = document.getElementById('btnGeolocate');
  if (btnGeolocate) {
    btnGeolocate.addEventListener('click', () => {
      if (!navigator.geolocation) return alert('Geolocation not supported');
      navigator.geolocation.getCurrentPosition((pos) => {
        map.flyTo({ center: [pos.coords.longitude, pos.coords.latitude], zoom: 15 });
      }, (err) => {
        console.warn('Geolocation error', err); alert('Unable to get current location.');
      }, { enableHighAccuracy: true });
    });
  }

  // Click popup: handle both point and line features 
  map.on('click', (e) => {
    const features = map.queryRenderedFeatures(e.point, {
      layers: ['garage-points', 'street-parking-layer', 'parking-all-points', 'parking-restricted']
    });
    if (!features.length) return;
    const f = features[0];
    const props = f.properties || {};
    const title = props.name || props.label || props.facility || props.TYPE || 'Parking';
    const popupHtml = `<strong>${title}</strong>
      <div style="font-size:0.9rem;margin-top:6px">
        ${Object.entries(props).slice(0,6).map(([k,v]) => `<div><em>${k}</em>: ${v}</div>`).join('')}
      </div>`;
    new mapboxgl.Popup().setLngLat(e.lngLat).setHTML(popupHtml).addTo(map);
  });

  map.getCanvas().style.cursor = 'default';
});