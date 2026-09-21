// ============================================================================
// 18. VIIRS NIGHTTIME LIGHTS — AQUITAINE, FRANCE
// ============================================================================
// Nighttime-light intensity for 2024 was derived from the annual VIIRS
// Day/Night Band (DNB) nighttime-lights product.
//
// The annual average-radiance band was transformed using log(x + 1).
//
// The VIIRS annual product has a native pixel size of approximately 463.83 m.
// Because nighttime-light radiance is continuous, bilinear resampling was
// applied from the native VIIRS image when harmonizing the layer to the
// common 30-m working grid.
//
// IMPORTANT:
// Bilinear resampling is applied to the original annual VIIRS image, not to
// an ImageCollection composite, so the native VIIRS projection is preserved.
//
// Export at 30 m does not increase the native spatial information content.
//
// Source:
// NOAA VIIRS DNB Annual Nighttime Lights V2.2
// GEE: NOAA/VIIRS/DNB/ANNUAL_V22
//
// Reference:
// Elvidge, C.D., Zhizhin, M., Ghosh, T., Hsu, F.C., & Taneja, J. (2021).
// Annual time series of global VIIRS nighttime lights derived from monthly
// averages: 2012 to 2019. Remote Sensing, 13(5), 922.
// https://doi.org/10.3390/rs13050922
// ============================================================================


// --------------------------------------------------------------------------
// 1. Settings and study-area processing extent
// --------------------------------------------------------------------------
var SCALE = 30;
var CRS = 'EPSG:4326';

var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. VIIRS Annual Nighttime Lights, 2024
// Native pixel size: approximately 463.83 m
// Band: average radiance
// --------------------------------------------------------------------------
var viirs2024 = ee.Image(
  ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22')
    .filterDate('2024-01-01', '2025-01-01')
    .first()
)
  .select('average');


// Preserve the true native VIIRS projection.
var nativeProjection = viirs2024.projection();


// --------------------------------------------------------------------------
// 3. Log(x + 1) transformation at native VIIRS support
// --------------------------------------------------------------------------
var ntlNative = viirs2024
  .add(1)
  .log()
  .rename('NTL_mean_log1p')
  .setDefaultProjection(nativeProjection);


// --------------------------------------------------------------------------
// 4. Bilinear harmonization to the 30-m working grid
// --------------------------------------------------------------------------
// Continuous variable -> bilinear interpolation from the native VIIRS grid.

var ntl30m = ntlNative
  .resample('bilinear')
  .clip(roi);


// --------------------------------------------------------------------------
// 5. Visualization
// --------------------------------------------------------------------------
var ntlVis = {
  min: 0,
  max: 4,
  palette: [
    '440154',
    '3B528B',
    '21918C',
    '5DC863',
    'FDE725'
  ]
};


// Native VIIRS log-transformed surface.
Map.addLayer(
  ntlNative.clip(roi),
  ntlVis,
  'VIIRS NTL 2024 - native ~464 m',
  true
);


// Bilinear surface that will be exported at 30 m.
Map.addLayer(
  ntl30m,
  ntlVis,
  'VIIRS NTL 2024 - bilinear 30 m',
  false
);


// --------------------------------------------------------------------------
// 6. Export to Earth Engine Asset
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: ntl30m,
  description: 'VIIRS_ntl_mean_log1p_2024_Aq',
  assetId:
    'users/omarorellanahn/INRIA/VIIRS_ntl_mean_log1p_2024_Aq',
  region: roi,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13
});
