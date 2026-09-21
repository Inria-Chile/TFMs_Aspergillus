// ============================================================================
// 17. GLOBAL WIND ATLAS — WIND SPEED AT 10 m
// AQUITAINE, FRANCE
// ============================================================================
// Mean annual wind speed at 10 m above ground was obtained from the Global
// Wind Atlas (GWA) v3.
//
// The GWA provides wind-climate estimates on a native 250-m grid.
// Only wind speed at 10 m above ground was retained for the analytical
// workflow.
//
// Wind speed is a continuous variable. Bilinear resampling was therefore
// applied to the original source images before mosaicking and subsequent
// harmonization to the common 30-m working grid.
//
// Export at 30 m does not increase the native spatial information content.
//
// Source:
// Global Wind Atlas v3 — DTU Wind Energy / World Bank Group
// https://globalwindatlas.info
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var SCALE = 30;
var CRS_UTM = 'EPSG:32630';

var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. Global Wind Atlas v3 — 10-m wind speed
// Native spatial resolution: 250 m
// Units: m/s
// --------------------------------------------------------------------------
var gwaCollection = ee.ImageCollection(
  'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/global_wind_atlas/wind-speed'
);

var wind10Collection = gwaCollection
  .filter(ee.Filter.eq('height', 10))
  .select('b1');


// Use the native projection of the original GWA image.
var nativeProjection = ee.Image(
  wind10Collection.first()
).projection();


// --------------------------------------------------------------------------
// 3. Native 250-m wind-speed surface
// --------------------------------------------------------------------------
// Preserve the native projection explicitly after mosaicking.

var windSpeed10mNative = wind10Collection
  .mosaic()
  .setDefaultProjection(nativeProjection)
  .rename('wind_speed_10m')
  .clip(roi);


// --------------------------------------------------------------------------
// 4. Bilinear resampling
// --------------------------------------------------------------------------
// Earth Engine recommends applying resampling to the source images BEFORE
// creating a mosaic/composite.

var wind10BilinearCollection = wind10Collection.map(function(image) {
  return image.resample('bilinear');
});

var windSpeed10mBilinear = wind10BilinearCollection
  .mosaic()
  .setDefaultProjection(nativeProjection)
  .rename('wind_speed_10m')
  .clip(roi);


// --------------------------------------------------------------------------
// 5. Visualization
// --------------------------------------------------------------------------
var windVis = {
  min: 2,
  max: 7,
  palette: [
    '081d58',
    '225ea8',
    '41b6c4',
    'a1dab4',
    'ffffcc',
    'feb24c',
    'f03b20',
    'bd0026'
  ]
};


// Native 250-m GWA surface.
Map.addLayer(
  windSpeed10mNative,
  windVis,
  'GWA wind speed 10 m - native 250 m',
  true
);


// Bilinear version used for final 30-m export.
Map.addLayer(
  windSpeed10mBilinear,
  windVis,
  'GWA wind speed 10 m - bilinear 30 m',
  false
);


// --------------------------------------------------------------------------
// 6. Export final 30-m wind-speed covariate
// --------------------------------------------------------------------------
// Continuous variable: bilinear resampling.
// Native source: 250 m
// Final working grid: 30 m, EPSG:32630

Export.image.toAsset({
  image: windSpeed10mBilinear,
  description: 'GWA_WindSpeed_10m_Aq',
  assetId: 'users/omarorellanahn/INRIA/GWA_WindSpeed_10m_Aq',
  region: roi,
  scale: SCALE,
  crs: CRS_UTM,
  maxPixels: 1e13
});
