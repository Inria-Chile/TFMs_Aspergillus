// ============================================================================
// 05. SENTINEL-5P ATMOSPHERIC COVARIATES — AQUITAINE, FRANCE
// ============================================================================
// Annual atmospheric covariates for 2023 were derived from Sentinel-5P
// OFFL Level-3 products available in Google Earth Engine.
//
// For each product, all observations within the study-area processing extent
// and calendar year 2023 were summarized using the temporal median. When the
// product contained a "cloud_fraction" band, observations with cloud fraction
// > 0.3 were masked before temporal aggregation.
//
// Five atmospheric products were generated:
//   - Tropospheric NO2 column density
//   - SO2 column density
//   - CO column density
//   - Tropospheric HCHO column density
//   - Absorbing Aerosol Index
//
// Sentinel-5P products have a much coarser native spatial resolution than
// the 30-m working grid used in this study. Therefore, bilinear resampling
// was applied during export to harmonize these continuous variables with the
// common spatial framework used for the environmental covariates.
//
// Native spatial resolution of the source products:
//   - NO2, SO2, HCHO, AER_AI: 3.5 × 7 km
//   - CO: 7 × 7 km
//
// The 6-km processing extent defined in Script 01 was used throughout.
// Outputs were exported in WGS 84 / UTM zone 30N (EPSG:32630).
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var roiFC = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = roiFC.geometry();
Map.centerObject(roi, 8);


// --------------------------------------------------------------------------
// 2. Temporal and export settings
// --------------------------------------------------------------------------
var startDate = '2023-01-01';
var endDate   = '2024-01-01';

var SCALE_EXPORT = 30;
var CRS_UTM = 'EPSG:32630';
var exportRegion = roi.bounds(1);

var ASSET_DIR = 'users/omarorellanahn/INRIA';


// --------------------------------------------------------------------------
// 3. Function to generate annual Sentinel-5P median
// --------------------------------------------------------------------------
function buildS5P(collectionId, bandName, outName, maxCloudFrac) {

  var col = ee.ImageCollection(collectionId)
    .filterDate(startDate, endDate)
    .filterBounds(roi);

  var firstImage = ee.Image(col.first());
  var hasCloudFraction = firstImage.bandNames().contains('cloud_fraction');

  col = ee.ImageCollection(
    ee.Algorithms.If(
      hasCloudFraction,
      col.map(function(image) {
        return image.updateMask(
          image.select('cloud_fraction').lte(maxCloudFrac)
        );
      }),
      col
    )
  );

  return col
    .select(bandName)
    .median()
    .clip(roi)
    .rename(outName);
}


// --------------------------------------------------------------------------
// 4. Atmospheric covariates
// --------------------------------------------------------------------------
var no2 = buildS5P(
  'COPERNICUS/S5P/OFFL/L3_NO2',
  'tropospheric_NO2_column_number_density',
  'NO2_mean',
  0.3
);

var so2 = buildS5P(
  'COPERNICUS/S5P/OFFL/L3_SO2',
  'SO2_column_number_density',
  'SO2_mean',
  0.3
);

var co = buildS5P(
  'COPERNICUS/S5P/OFFL/L3_CO',
  'CO_column_number_density',
  'CO_mean',
  0.3
);

var hcho = buildS5P(
  'COPERNICUS/S5P/OFFL/L3_HCHO',
  'tropospheric_HCHO_column_number_density',
  'HCHO_mean',
  0.3
);

var ai = buildS5P(
  'COPERNICUS/S5P/OFFL/L3_AER_AI',
  'absorbing_aerosol_index',
  'AER_AI_mean',
  0.3
);


// --------------------------------------------------------------------------
// 5. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  no2,
  {
    min: 0.0000125,
    max: 0.000037,
    palette: ['081d58','225ea8','41b6c4','a1dab4',
              'ffffcc','fdae61','f46d43','d73027']
  },
  'NO2 annual median',
  false
);

Map.addLayer(
  so2,
  {
    min: -0.000050,
    max: 0.000227,
    palette: ['081d58','225ea8','41b6c4','a1dab4',
              'ffffcc','fdae61','f46d43','d73027']
  },
  'SO2 annual median',
  false
);

Map.addLayer(
  co,
  {
    min: 0.024,
    max: 0.033,
    palette: ['081d58','225ea8','41b6c4','a1dab4',
              'ffffcc','fdae61','f46d43','d73027']
  },
  'CO annual median',
  false
);

Map.addLayer(
  hcho,
  {
    min: 0.00005,
    max: 0.00013,
    palette: ['081d58','225ea8','41b6c4','a1dab4',
              'ffffcc','fdae61','f46d43','d73027']
  },
  'HCHO annual median',
  false
);

Map.addLayer(
  ai,
  {
    min: -0.50,
    max: -0.16,
    palette: ['081d58','225ea8','41b6c4','a1dab4',
              'ffffcc','fdae61','f46d43','d73027']
  },
  'Absorbing Aerosol Index annual median',
  true
);


// --------------------------------------------------------------------------
// 6. Export to Earth Engine Assets
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: no2.resample('bilinear'),
  description: 'NO2_mean_30m_Aquitaine',
  assetId: ASSET_DIR + '/NO2_mean_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: so2.resample('bilinear'),
  description: 'SO2_mean_30m_Aquitaine',
  assetId: ASSET_DIR + '/SO2_mean_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: co.resample('bilinear'),
  description: 'CO_mean_30m_Aquitaine',
  assetId: ASSET_DIR + '/CO_mean_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: hcho.resample('bilinear'),
  description: 'HCHO_mean_30m_Aquitaine',
  assetId: ASSET_DIR + '/HCHO_mean_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

Export.image.toAsset({
  image: ai.resample('bilinear'),
  description: 'AER_AI_mean_30m_Aquitaine',
  assetId: ASSET_DIR + '/AER_AI_mean_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});
