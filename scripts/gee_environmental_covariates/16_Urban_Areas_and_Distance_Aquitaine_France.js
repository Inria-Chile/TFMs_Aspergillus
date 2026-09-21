// ============================================================================
// 16. URBAN AREAS AND DISTANCE — AQUITAINE, FRANCE
// ============================================================================
// Urban areas were derived from the ESRI 10-m Annual Land Use/Land Cover
// product for 2024 using the Built class (class value = 7).
//
// The categorical built-area layer was harmonized to the 30-m working grid.
// Connected built areas were converted to polygons and only patches with an
// area >= 1 km² were retained.
//
// Distance to these urban areas was calculated directly on the 30-m working
// grid using ee.Image.cumulativeCost() with geodeticDistance = true.
//
// Outputs:
//   1. Urban areas >= 1 km²
//   2. Distance to urban areas >= 1 km²
//
// No bilinear resampling is used for the categorical urban mask.
// The distance surface is calculated directly at 30 m.
// ============================================================================


// --------------------------------------------------------------------------
// 1. Settings and study-area extent
// --------------------------------------------------------------------------
var SCALE = 30;
var CRS_UTM = 'EPSG:32630';

var MIN_URBAN_AREA = 1000000;  // 1 km²
var MAX_DISTANCE = 40000;       // 40 km

var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

// Context allows urban areas outside the final ROI to contribute to distance.
var urbanContext = roi.buffer(MAX_DISTANCE);

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. ESRI Annual Land Cover 2024
// Native resolution: 10 m
// Built class = 7
// --------------------------------------------------------------------------
var esri2024 = ee.ImageCollection(
  'projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS'
)
  .filterDate('2024-01-01', '2025-01-01')
  .filterBounds(urbanContext)
  .mosaic()
  .clip(urbanContext);


// --------------------------------------------------------------------------
// 3. Binary built-area mask at 30 m
// Categorical variable -> nearest-neighbor behavior
// --------------------------------------------------------------------------
var built30m = esri2024
  .eq(7)
  .rename('built')
  .toByte()
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  });


// --------------------------------------------------------------------------
// 4. Retain connected urban areas >= 1 km²
// --------------------------------------------------------------------------
var builtMasked = built30m.selfMask();

var urbanPolygons = builtMasked.reduceToVectors({
  geometry: urbanContext,
  scale: SCALE,
  crs: CRS_UTM,
  geometryType: 'polygon',
  eightConnected: true,
  labelProperty: 'built',
  reducer: ee.Reducer.countEvery(),
  maxPixels: 1e13,
  tileScale: 4,
  geometryInNativeProjection: true
});


// Calculate patch area and retain only urban areas >= 1 km².
var urbanPolygons1km2 = urbanPolygons
  .map(function(feature) {
    return feature.set(
      'area_m2',
      feature.geometry().area(1)
    );
  })
  .filter(
    ee.Filter.gte('area_m2', MIN_URBAN_AREA)
  );


// Rasterize retained urban areas directly on the 30-m grid.
var urban1km2 = ee.Image.constant(0)
  .byte()
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .paint({
    featureCollection: urbanPolygons1km2,
    color: 1
  })
  .selfMask()
  .rename('Urban_1km2');


// --------------------------------------------------------------------------
// 5. Distance to urban areas >= 1 km²
// --------------------------------------------------------------------------
var costImage = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  });

var roiMask = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .clip(roi)
  .selfMask();


var urbanDistance = costImage
  .cumulativeCost({
    source: urban1km2,
    maxDistance: MAX_DISTANCE,
    geodeticDistance: true
  })
  .updateMask(roiMask)
  .rename('dist_urban_m');


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------

// Urban areas used as distance sources.
Map.addLayer(
  urban1km2,
  {
    palette: ['8B0000']
  },
  'Urban areas >= 1 km²',
  true
);


// Final distance covariate.
Map.addLayer(
  urbanDistance,
  {
    min: 0,
    max: 20000,
    palette: [
      '0000FF',
      '00AAFF',
      '00FFAA',
      'FFFF00',
      'FF8800',
      'FF0000'
    ]
  },
  'Distance to urban areas >= 1 km²',
  true
);


// --------------------------------------------------------------------------
// 7. Export urban areas >= 1 km²
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: urban1km2.clip(roi),
  description: 'Urban_1km2_2024_Aq',
  assetId:
    'users/omarorellanahn/INRIA/Urban_1km2_2024_Aq',
  region: roi,
  scale: SCALE,
  crs: CRS_UTM,
  maxPixels: 1e13,
  pyramidingPolicy: {
    '.default': 'mode'
  }
});


// --------------------------------------------------------------------------
// 8. Export distance to urban areas >= 1 km²
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: urbanDistance,
  description: 'Urban1km2_Distance_Aq',
  assetId:
    'users/omarorellanahn/INRIA/Urban1km2_Distance_Aq',
  region: roi,
  scale: SCALE,
  crs: CRS_UTM,
  maxPixels: 1e13,
  pyramidingPolicy: {
    '.default': 'mean'
  }
});
