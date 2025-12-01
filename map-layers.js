
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
  }
    );

  // street_parking_detailed: will render as line if blockfaces are lines; we still pass a linePaint option
  await addGeoJSONLayer({
    sourceId: 'streetparking-src',
    layerId: 'street-parking-layer',
    geojsonPath: 'assets/street_parking_detailed.geojson',
    // The default linePaint inside addGeoJSONLayer already includes zoom-based expressions,
    // but allow overrides here if desired.
    linePaint: {
      // you can override color here if you want different color than default
      // 'line-color': '#10b981'
    }
  });

  // parking_all_points (points)
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

  // Initialize visibility from checkboxes
  const initVisibility = [
    { checkboxId: 'toggleGarages', layerId: 'garage-points' },
    { checkboxId: 'toggleBlockface', layerId: 'street-parking-layer' },
    { checkboxId: 'toggleRestricted', layerId: 'parking-all-points' }
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
      layers: ['garage-points', 'street-parking-layer', 'parking-all-points']
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