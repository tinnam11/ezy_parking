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
    // keep a deep copy of the original data for this source (used by time filter)
    try {
      originalSourceData[sourceId] = JSON.parse(JSON.stringify(data));
    } catch (err) {
      originalSourceData[sourceId] = data;
    }
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

// --- Time filter helpers ---
function timeStrToMinutes(t) {
  if (!t && t !== '00:00') return null;
  if (typeof t !== 'string') return null;
  const parts = t.split(':');
  if (parts.length < 2) return null;
  const hh = parseInt(parts[0], 10);
  const mm = parseInt(parts[1], 10);
  if (Number.isNaN(hh) || Number.isNaN(mm)) return null;
  return hh * 60 + mm;
}

function rangesFromInterval(startMin, endMin) {
  // return array of [s,e] intervals in 0..1439 that represent the interval,
  // splitting wrap-around intervals into two parts
  if (startMin == null || endMin == null) return [[0, 24 * 60 - 1]];
  if (startMin <= endMin) return [[startMin, endMin]];
  // wrap-around
  return [[startMin, 24 * 60 - 1], [0, endMin]];
}

function intervalsOverlap(aStart, aEnd, bStart, bEnd) {
  const aRanges = rangesFromInterval(aStart, aEnd);
  const bRanges = rangesFromInterval(bStart, bEnd);
  for (const [as, ae] of aRanges) {
    for (const [bs, be] of bRanges) {
      if (as <= be && bs <= ae) return true;
    }
  }
  return false;
}

function restoreSourceToOriginal(sourceId) {
  const orig = originalSourceData[sourceId];
  if (!orig) return;
  const src = map.getSource(sourceId);
  if (src) src.setData(JSON.parse(JSON.stringify(orig)));
}

function applyTimeFilter() {
  // sources to apply time filtering to (do not filter garages)
  const sourceIds = ['streetparking-src', 'parkingpoints-src', 'restricted-src'];

  const useNow = document.getElementById('timeNow')?.checked;
  let selStart = document.getElementById('timeFrom')?.value || '';
  let selEnd = document.getElementById('timeTo')?.value || '';

  if (useNow) {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    selStart = `${hh}:${mm}`;
    selEnd = selStart;
  }

  // if neither start nor end provided and not using now -> restore originals
  if ((!selStart || selStart === '') && (!selEnd || selEnd === '') && !useNow) {
    sourceIds.forEach(sid => restoreSourceToOriginal(sid));
    return;
  }

  const selStartMin = timeStrToMinutes(selStart);
  const selEndMin = timeStrToMinutes(selEnd != null && selEnd !== '' ? selEnd : selStart);

  sourceIds.forEach((sid) => {
    const orig = originalSourceData[sid];
    if (!orig) return;
    // filter features by overlap between selected interval and feature's weekday_start/weekday_end
    const filtered = orig.features.filter((f) => {
      const p = f.properties || {};
      const fStart = p.weekday_start || p.start || null;
      const fEnd = p.weekday_end || p.end || null;
      // if no time restrictions on feature -> include
      if (!fStart && !fEnd) return true;
      const fStartMin = timeStrToMinutes(fStart);
      const fEndMin = timeStrToMinutes(fEnd);
      // If feature times are invalid, include conservatively
      if (fStartMin == null || fEndMin == null) return true;
      return intervalsOverlap(fStartMin, fEndMin, selStartMin, selEndMin);
    });

    const newGeo = { type: 'FeatureCollection', features: filtered };
    const src = map.getSource(sid);
    if (src) src.setData(newGeo);
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
      resultsDiv.innerHTML = '<p style="color:#ef4444"> Location not found. Try a different address.</p>';
    }
  } catch (error) {
    console.error('Geocoding error:', error);
    resultsDiv.innerHTML = '<p style="color:#ef4444"> Error searching for location. Please try again.</p>';
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
    // include blockface/line source so time filters on street blockfaces
    // (which use weekday_start/weekday_end) affect search results
    { id: 'streetparking-src', type: 'Blockface' },
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
  
  // If travel-time filtering is enabled, compute durations for nearby candidates
  const useTravel = document.getElementById('useTravelTime')?.checked;
  const travelModeVal = document.getElementById('travelMode')?.value || 'walking';
  const maxMin = parseFloat(document.getElementById('travelTimeRange')?.value || 0);

  // add an estimated travel time (seconds) for each result so we can sort
  // by travel time visually even before full travel-time filtering is implemented
  results.forEach(r => {
    r.estimatedSec = (r.distance / milesPerMinuteForMode(travelModeVal)) * 60; // seconds
    r.actualSec = null; // will be filled with matrix durations when available
  });

  async function finalizeResults() {
    let final = results;
    if (useTravel && maxMin > 0) {
      // take up to matrixLimit closest candidates to compute accurate durations
      const candidates = results.sort((a,b)=>a.distance-b.distance).slice(0, TRAVEL.matrixLimit);
      const destinations = candidates.map(c => c.coordinates);
      const origin = [searchLng, searchLat];
      const durations = await fetchMatrixDurations(origin, destinations, travelModeVal);
      const cutoff = maxMin * 60;
      // filter candidates by duration (fallback to distance estimate when duration null)
      const allowedSet = new Set();
      candidates.forEach((c, i) => {
        const d = durations[i];
        if (d != null) {
          c.actualSec = d; // store exact duration (seconds)
          if (d <= cutoff) allowedSet.add(c);
        } else {
          const estSec = (c.distance / milesPerMinuteForMode(travelModeVal)) * 60;
          c.actualSec = null;
          if (estSec <= cutoff) allowedSet.add(c);
        }
      });
      // final = those in allowedSet, keep original ordering by distance
      final = results.filter(r => allowedSet.has(r));
    }

    //sort by distance or price based on sortSelect
    const sortBy = document.getElementById('sortSelect')?.value || 'distance';
    final.sort((a, b) => {
      if (sortBy === 'distance') {
        return a.distance - b.distance;
      } else {
        const priceA = typeof a.rate === 'number' ? a.rate : 999;
        const priceB = typeof b.rate === 'number' ? b.rate : 999;
        return priceA - priceB;
      }
    });

    // If the user has enabled the "Use travel time" checkbox, sort by travel
    // time (prefer exact matrix durations when available, otherwise use the
    // estimate) and show the full list so users can inspect ordering by travel
    // time. If not using travel-time mode, show the top 15 as before.
    const showByTravel = document.getElementById('useTravelTime')?.checked;
    if (showByTravel) {
      final.sort((a, b) => ( (a.actualSec ?? a.estimatedSec) - (b.actualSec ?? b.estimatedSec) ));
      displaySearchResults(final);
    } else {
      displaySearchResults(final.slice(0, 15)); //show top 15
    }
  }

  // run the async finalize
  finalizeResults().catch(err => {
    console.warn('Error finalizing search results', err);
    const showByTravel = document.getElementById('useTravelTime')?.checked;
    if (showByTravel) displaySearchResults(results);
    else displaySearchResults(results.slice(0,15));
  });
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

  // my location button
const btnGeo = document.getElementById('btnGeolocate');

if (btnGeo) {
  btnGeo.addEventListener('click', () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const userLng = pos.coords.longitude;
        const userLat = pos.coords.latitude;

        map.flyTo({
          center: [userLng, userLat],
          zoom: 15,
          essential: true
        });

        if (window.userLocationMarker) {
          const addressCoords = data.features[0].geometry.coordinates;

          window.userLocationMarker.setLngLat(addressCoords);
        } else {
          window.userLocationMarker = new mapboxgl.Marker({ color: '#0ea5e9' })
            .setLngLat([userLng, userLat])
            .setPopup(new mapboxgl.Popup().setHTML(`<strong>You are here</strong>`))
            .addTo(map);
        }

        const geocodeUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${userLng},${userLat}.json?access_token=${mapboxgl.accessToken}`;

        try {
          const response = await fetch(geocodeUrl);
          const data = await response.json();

          if (data.features && data.features.length > 0) {
            const address = data.features[0].place_name; 

            const searchInput = document.getElementById("searchInput");
            if (searchInput) {
              searchInput.value = address;
            }
          }
        } catch (err) {
          console.error("Reverse geocoding failed:", err);
        }
      },

      (err) => {
        alert('Unable to retrieve your location: ' + err.message);
      }
    );

  });
}

  
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

  // Wire up time filter inputs
  const timeFrom = document.getElementById('timeFrom');
  const timeTo = document.getElementById('timeTo');
  const timeNow = document.getElementById('timeNow');
  if (timeFrom) timeFrom.addEventListener('input', applyTimeFilter);
  if (timeTo) timeTo.addEventListener('input', applyTimeFilter);
  if (timeNow) timeNow.addEventListener('change', applyTimeFilter);

  // apply initial time filter if any values present
  applyTimeFilter();

  // Wire travel-time UI controls
  const travelRange = document.getElementById('travelTimeRange');
  const travelDisplay = document.getElementById('travelTimeDisplay');
  const travelMode = document.getElementById('travelMode');
  const useTravel = document.getElementById('useTravelTime');
  if (travelRange && travelDisplay) {
    travelDisplay.textContent = travelRange.value;
    travelRange.addEventListener('input', (e) => {
      travelDisplay.textContent = e.target.value;
    });
    travelRange.addEventListener('change', applyTravelTimeFilter);
  }
  if (travelMode) travelMode.addEventListener('change', applyTravelTimeFilter);
  if (useTravel) useTravel.addEventListener('change', applyTravelTimeFilter);


  // Zoom buttons
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  if (btnZoomIn) btnZoomIn.addEventListener('click', () => map.zoomIn());
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => map.zoomOut());

  // Home button
  const btnHome = document.getElementById('btnHome');
  if (btnHome) {
    btnHome.addEventListener('click', () => {
    window.location.href = 'index.html';
    });
  }


  // Sidebar minimize / maximize
  const appMain = document.querySelector('.app-main');
  const btnToggleSidebar = document.getElementById('btnToggleSidebar');

  if (appMain && btnToggleSidebar) {
    btnToggleSidebar.addEventListener('click', () => {
      const collapsed = appMain.classList.toggle('sidebar-collapsed');

      // ARIA state
      btnToggleSidebar.setAttribute('aria-expanded', String(!collapsed));

      // Button text
      btnToggleSidebar.textContent = collapsed ? 'Maximize' : 'Minimize';

      map.resize()
    });
  }

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

// store original fetched GeoJSON data so we can restore and filter safely
const originalSourceData = {};

// travel time settings and Mapbox Matrix integration
const TRAVEL = {
  speeds: { // mph
    walking: 3.0,
    driving: 25.0
  },
  // limit destinations in single Matrix request to avoid hitting API limits
  matrixLimit: 25
};

function getFeatureCoords(feature) {
  if (!feature || !feature.geometry) return null;
  if (feature.geometry.type === 'Point') return feature.geometry.coordinates;
  if (feature.geometry.type === 'LineString') return feature.geometry.coordinates[0];
  if (feature.geometry.type === 'MultiLineString' && feature.geometry.coordinates.length) return feature.geometry.coordinates[0][0];
  return null;
}

function milesPerMinuteForMode(mode) {
  const mph = TRAVEL.speeds[mode] || TRAVEL.speeds.walking;
  return mph / 60.0;
}

async function fetchMatrixDurations(origin, destinations, mode) {
  // origin: [lng, lat]
  // destinations: array of [lng, lat]
  // returns array of durations in seconds (null if unknown) same order as destinations
  if (!origin || !destinations || destinations.length === 0) return [];

  const coords = [origin].concat(destinations).map(c => `${c[0]},${c[1]}`).join(';');
  const destIndexes = destinations.map((_, i) => i + 1).join(';');
  const profile = mode === 'driving' ? 'driving' : 'walking';
  const url = `https://api.mapbox.com/directions-matrix/v1/mapbox/${profile}/${coords}?sources=0&destinations=${destIndexes}&annotations=duration&access_token=${mapboxgl.accessToken}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Matrix request failed: ${resp.statusText}`);
    const data = await resp.json();
    if (!data || !data.durations || !Array.isArray(data.durations) || data.durations.length === 0) return destinations.map(() => null);
    const row = data.durations[0] || [];
    // Mapbox may return nulls for unreachable
    return row.map(d => (d == null ? null : d));
  } catch (err) {
    console.warn('Matrix error', err);
    return destinations.map(() => null);
  }
}

function getOriginFallback() {
  // return [lng, lat] from searchMarker, else current geolocation (if available), else map center
  if (searchMarker) return [searchMarker.getLngLat().lng, searchMarker.getLngLat().lat];
  // prefer last known geolocation if available via navigator (synchronous fallback)
  if (map && map.getCenter) {
    const c = map.getCenter();
    return [c.lng, c.lat];
  }
  return null;
}

async function applyTravelTimeFilter() {
  const use = document.getElementById('useTravelTime')?.checked;
  const maxMin = parseFloat(document.getElementById('travelTimeRange')?.value || 0);
  const mode = document.getElementById('travelMode')?.value || 'walking';

  if (!use || !maxMin || maxMin <= 0) {
    // restore original data for the sources we filtered earlier
    ['streetparking-src', 'parkingpoints-src', 'restricted-src'].forEach(restoreSourceToOriginal);
    return;
  }

  const origin = getOriginFallback();
  if (!origin) return;

  // gather candidates from originals
  const sourceIds = ['streetparking-src', 'parkingpoints-src', 'restricted-src'];
  const allCandidates = [];
  const featureSourceMap = {};

  sourceIds.forEach((sid) => {
    const orig = originalSourceData[sid];
    if (!orig || !orig.features) return;
    orig.features.forEach((f, idx) => {
      const coords = getFeatureCoords(f);
      if (!coords) return;
      const distance = calculateDistance(origin[0], origin[1], coords[0], coords[1]);
      allCandidates.push({ sid, idx, feature: f, coords, distance });
    });
  });

  // estimate a loose distance threshold to prefilter candidates: maxMin * speed_miles_per_min * buffer
  const mpmin = milesPerMinuteForMode(mode);
  const maxDistance = Math.max(0.5, maxMin * mpmin * 1.5); // at least 0.5 miles

  // keep only those within maxDistance, sort by distance
  let candidates = allCandidates.filter(c => c.distance <= maxDistance).sort((a,b) => a.distance - b.distance);

  // if no candidates by distance, expand a bit to include some for routing
  if (candidates.length === 0) {
    candidates = allCandidates.sort((a,b)=>a.distance - b.distance).slice(0, TRAVEL.matrixLimit);
  }

  // split into two groups: those we will query via Matrix (limited) and remaining will be approximated
  const matrixTargets = candidates.slice(0, TRAVEL.matrixLimit);
  const approxTargets = candidates.slice(TRAVEL.matrixLimit);

  const destinations = matrixTargets.map(t => t.coords);
  const durations = await fetchMatrixDurations(origin, destinations, mode);

  // build set of allowed features
  const allowed = new Set();
  const cutoffSec = maxMin * 60;

  matrixTargets.forEach((t, i) => {
    const dur = durations[i];
    if (dur != null) {
      if (dur <= cutoffSec) allowed.add(`${t.sid}::${t.idx}`);
    } else {
      // if unreachable in matrix, fall back to distance estimate
      const estimateSec = (t.distance / milesPerMinuteForMode(mode)) * 60; // distance (miles) / (miles/min) => minutes * 60 -> sec
      if (estimateSec <= cutoffSec) allowed.add(`${t.sid}::${t.idx}`);
    }
  });

  // process approxTargets using a simple estimate
  approxTargets.forEach(t => {
    const estMin = t.distance / milesPerMinuteForMode(mode);
    if (estMin <= maxMin * 1.05) allowed.add(`${t.sid}::${t.idx}`);
  });

  // For features not evaluated (outside initial candidates), default to excluded
  // Now create filtered GeoJSON per source and setData
  sourceIds.forEach((sid) => {
    const orig = originalSourceData[sid];
    if (!orig) return;
    const filtered = orig.features.filter((f, idx) => allowed.has(`${sid}::${idx}`));
    const newGeo = { type: 'FeatureCollection', features: filtered };
    const src = map.getSource(sid);
    if (src) src.setData(newGeo);
  });
}
