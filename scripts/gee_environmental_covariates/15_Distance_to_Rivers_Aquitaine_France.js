// ============================================================================
// 15. DISTANCE TO RIVERS — AQUITAINE, FRANCE
// ============================================================================
// Distance to the nearest river was calculated using the WWF HydroSHEDS
// Free Flowing Rivers Network v1.
//
// River reaches with RIV_ORD <= 7 were retained. In this dataset, RIV_ORD is
// based on long-term average discharge; classes 1–7 correspond to river
// reaches with estimated discharge >= 0.1 m3/s.
//
// River vectors were rasterized directly on the common 30-m working grid.
// A constant-cost raster (cost = 1) was then used with
// ee.Image.cumulativeCost() and geodeticDistance = true to calculate distance
// in meters from each pixel to the nearest river-source pixel.
//
// A 40-km contextual buffer was used when selecting river segments so that
// rivers immediately outside the final study extent could still contribute
// to the distance calculation.
//
// The distance covariate was generated directly at 30 m in WGS 84 / UTM
// zone 30N (EPSG:32630). No bilinear resampling was required.
//
// Source:
// WWF HydroSHEDS Free Flowing Rivers Network v1
// GEE: WWF/HydroSHEDS/v1/FreeFlowingRivers
//
// References:
// Lehner, B., Verdin, K., & Jarvis, A. (2008).
// New global hydrography derived from spaceborne elevation data.
// Eos, Transactions AGU, 89(10), 93–94.
//
// Grill, G. et al. (2019).
// Mapping the world's free-flowing rivers.
// Nature, 569, 215–221.
// ============================================================================


// --------------------------------------------------------------------------
// 1. Settings and study-area processing extent
// --------------------------------------------------------------------------
var SCALE = 30;
var CRS_UTM = 'EPSG:32630';
var MAX_DISTANCE = 40000;  // meters

var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

// Context used only to include rivers outside the final ROI.
var riverContext = roi.buffer(MAX_DISTANCE);

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. River network
// --------------------------------------------------------------------------
// RIV_ORD <= 7 retains reaches with long-term average discharge >= 0.1 m3/s.

var rivers = ee.FeatureCollection(
  'WWF/HydroSHEDS/v1/FreeFlowingRivers'
)
  .filterBounds(riverContext)
  .filter(
    ee.Filter.lte('RIV_ORD', 7)
  );

print(
  'River segments within 40-km context:',
  rivers.size()
);


// --------------------------------------------------------------------------
// 3. Explicit 30-m working grid
// --------------------------------------------------------------------------
// Constant traversal cost.

var costImage = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  });


// --------------------------------------------------------------------------
// 4. River-source raster
// --------------------------------------------------------------------------
// Rivers are painted directly on the 30-m target grid.
//
// A width of 2 pixels reproduces the original workflow and helps maintain
// continuous source lines after rasterization.

var riverSource = ee.Image.constant(0)
  .byte()
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .paint({
    featureCollection: rivers,
    color: 1,
    width: 2
  })
  .selfMask()
  .rename('river');


// Final ROI mask.

var roiMask = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .clip(roi)
  .selfMask();


// --------------------------------------------------------------------------
// 5. Geodetic distance to rivers
// --------------------------------------------------------------------------
var distRiverRaw = costImage
  .cumulativeCost({
    source: riverSource,
    maxDistance: MAX_DISTANCE,
    geodeticDistance: true
  })
  .rename('dist_river_m');


// Restrict final covariate to the study extent.

var distRiver = distRiverRaw
  .updateMask(roiMask)
  .rename('dist_river_m');


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------
var distanceVis = {
  min: 0,
  max: 10000,
  palette: [
    '0000FF',
    '00AAFF',
    '00FFAA',
    'FFFF00',
    'FF8800',
    'FF0000'
  ]
};


// Original vector river network.
Map.addLayer(
  rivers,
  {color: '0000FF'},
  'River network',
  false
);


// Rasterized river-source pixels.
Map.addLayer(
  riverSource,
  {
    palette: ['00FFFF']
  },
  'River source raster',
  false
);


// Full distance calculation before final ROI masking.
Map.addLayer(
  distRiverRaw,
  distanceVis,
  'Distance to rivers - full calculation',
  false
);


// Final distance covariate.
Map.addLayer(
  distRiver,
  distanceVis,
  'Distance to rivers',
  true
);


// ROI boundary.
var roiOutline = ee.Image()
  .byte()
  .paint({
    featureCollection: region,
    color: 1,
    width: 2
  });

Map.addLayer(
  roiOutline,
  {palette: ['000000']},
  'ROI boundary',
  false
);


// --------------------------------------------------------------------------
// 7. Export to Earth Engine Asset
// --------------------------------------------------------------------------
// Distance is calculated directly at 30 m.
// No additional bilinear resampling is applied.

Export.image.toAsset({
  image: distRiver,
  description: 'Distance_to_Rivers_30m_Aquitaine',
  assetId:
    'users/omarorellanahn/INRIA/Distance_to_Rivers_30m_Aquitaine',
  region: roi,
  scale: SCALE,
  crs: CRS_UTM,
  maxPixels: 1e13,
  pyramidingPolicy: {
    '.default': 'mean'
  }
});
