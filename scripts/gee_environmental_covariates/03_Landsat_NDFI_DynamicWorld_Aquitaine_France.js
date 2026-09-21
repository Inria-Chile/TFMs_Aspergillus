// ============================================================================
// 03. LANDSAT, NDFI AND DYNAMIC WORLD COVARIATES — AQUITAINE, FRANCE
// ============================================================================
// Scientific and technical basis:
//
// - Souza et al. (2005): Spectral Mixture Analysis (SMA), endmember fractions,
//   shade normalization, and formulation of the Normalized Difference Fraction
//   Index (NDFI).
//
// - Small (2004): Physical basis and general applicability of linear spectral
//   mixture models using generic vegetation, substrate, and dark endmembers
//   in Landsat ETM+ reflectance space.
//
// - Zhu & Woodcock (2012): Original Fmask algorithm for automated cloud and
//   cloud-shadow detection in Landsat imagery.
//
// - Zhu et al. (2015): Extension and improvement of Fmask for Landsat 8,
//   including use of the cirrus band.
//
// - Foga et al. (2017): Validation and comparison of operational Landsat cloud
//   detection algorithms, supporting the use of CFMask.
//
// - Skakun et al. (2022): CMIX intercomparison of cloud-masking algorithms for
//   Landsat 8 and Sentinel-2.
//
// - Wulder et al. (2019): Landsat program, calibrated measurements, sensor
//   continuity, and analysis-ready data context.
//
// - USGS Landsat Collection 2 documentation: QA_PIXEL and QA_RADSAT bit
//   definitions and Level-2 surface-reflectance scaling
//   (DN × 0.0000275 - 0.2).
//
// Landsat 8 and Landsat 9 Collection 2 Level-2 imagery from 2024 was used to
// generate a cloud-masked annual median surface-reflectance composite and NDFI.
// Dynamic World probabilities for trees, cropland, and built-up areas were
// summarized using the annual median for 2024.
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
// 2. Landsat 8/9 cloud and saturation masking
// --------------------------------------------------------------------------
function maskL89sr(image) {

  var qaMask = image.select('QA_PIXEL')
    .bitwiseAnd(parseInt('11111', 2))
    .eq(0);

  var satMask = image.select('QA_RADSAT').eq(0);

  var optical = image.select('SR_B.')
    .multiply(0.0000275)
    .add(-0.2);

  return image
    .addBands(optical, null, true)
    .updateMask(qaMask)
    .updateMask(satMask);
}


// --------------------------------------------------------------------------
// 3. Landsat 8 and 9 Collection 2 Level-2 — 2024
// --------------------------------------------------------------------------
var bands = [
  'SR_B2', 'SR_B3', 'SR_B4',
  'SR_B5', 'SR_B6', 'SR_B7',
  'QA_PIXEL', 'QA_RADSAT'
];

var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 70))
  .select(bands);

var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roi)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 70))
  .select(bands);


// --------------------------------------------------------------------------
// 4. Annual median Landsat composite
// --------------------------------------------------------------------------
var landsatComposite = L8.merge(L9)
  .map(maskL89sr)
  .median()
  .clip(roi)
  .select(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
  );

Map.addLayer(
  landsatComposite,
  {bands: ['red', 'green', 'blue'], min: 0.003, max: 0.11},
  'Landsat 8/9 composite 2024'
);


// --------------------------------------------------------------------------
// 5. NDFI from Spectral Mixture Analysis
// --------------------------------------------------------------------------
// Band order: Blue, Green, Red, NIR, SWIR1, SWIR2
// Endmembers from Souza et al. (2005).
var endmembers = [
  [0.0119, 0.0475, 0.0169, 0.6250, 0.2399, 0.0675], // GV
  [0.1514, 0.1597, 0.1421, 0.3053, 0.7707, 0.1975], // NPV
  [0.1799, 0.2479, 0.3158, 0.5437, 0.7707, 0.6646], // Soil
  [0.4031, 0.8714, 0.7900, 0.8989, 0.7002, 0.6607]  // Cloud
];

var sma = landsatComposite
  .select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
  .unmix(endmembers)
  .max(0)
  .rename(['GV', 'NPV', 'Soil', 'Cloud']);

var shade = sma.reduce(ee.Reducer.sum())
  .subtract(1.0)
  .abs()
  .rename('Shade');

var GVs = sma.select('GV')
  .divide(shade.subtract(1.0).abs())
  .rename('GVs');

var NDFI = sma.addBands([shade, GVs]).expression(
  '(GVs - (NPV + Soil)) / (GVs + NPV + Soil)',
  {
    GVs: GVs,
    NPV: sma.select('NPV'),
    Soil: sma.select('Soil')
  }
).rename('NDFI');

Map.addLayer(
  NDFI,
  {
    min: -0.5,
    max: 1,
    palette: [
      '1916d6', 'd6350c', 'dd8e35', 'f7ff2a',
      'acff36', 'f9ed14', 'aaffaa', '73aa24', '27593b'
    ]
  },
  'NDFI 2024'
);


// --------------------------------------------------------------------------
// 6. Dynamic World probabilities — 2024
// --------------------------------------------------------------------------
var DW = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(roi)
  .filterDate('2024-01-01', '2024-12-31')
  .select(['trees', 'crops', 'built'])
  .median()
  .clip(roi);

var trees = DW.select('trees');
var crops = DW.select('crops');
var built = DW.select('built');

Map.addLayer(
  trees,
  {min: 0, max: 0.7, palette: ['red', 'yellow', 'limegreen', 'darkgreen']},
  'Tree probability',
  false
);

Map.addLayer(
  crops,
  {min: 0, max: 0.6, palette: ['fff2b2', 'd9a441', '795548']},
  'Cropland probability',
  false
);

Map.addLayer(
  built,
  {min: 0, max: 0.8, palette: ['d8cfb7', 'a89489', '746766', '4b4544']},
  'Built-up probability',
  false
);


// --------------------------------------------------------------------------
// 7. Export settings
// --------------------------------------------------------------------------
var CRS_UTM = 'EPSG:32630';
var ASSET_DIR = 'users/omarorellanahn/INRIA';


// --------------------------------------------------------------------------
// 8. Export to Earth Engine Assets
// --------------------------------------------------------------------------

// Six-band Landsat surface-reflectance composite
Export.image.toAsset({
  image: landsatComposite.toFloat(),
  description: 'L89_2024_Mosaico',
  assetId: ASSET_DIR + '/L89_2024_Mosaico',
  region: roi,
  scale: 30,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// NDFI
Export.image.toAsset({
  image: NDFI.toFloat(),
  description: 'L89_2024_NDFI_30m',
  assetId: ASSET_DIR + '/L89_2024_NDFI_30m',
  region: roi,
  scale: 30,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// Dynamic World — tree probability
Export.image.toAsset({
  image: trees.toFloat(),
  description: 'DW_2024_Prob_Bosque_Trees_10m',
  assetId: ASSET_DIR + '/DW_2024_Prob_Bosque_Trees_10m',
  region: roi,
  scale: 30,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// Dynamic World — cropland probability
Export.image.toAsset({
  image: crops.toFloat(),
  description: 'DW_2024_Prob_Cultivos_Crops_10m',
  assetId: ASSET_DIR + '/DW_2024_Prob_Cultivos_Crops_10m',
  region: roi,
  scale: 30,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// Dynamic World — built-up probability
Export.image.toAsset({
  image: built.toFloat(),
  description: 'DW_2024_Prob_Built_10m',
  assetId: ASSET_DIR + '/DW_2024_Prob_Built_10m',
  region: roi,
  scale: 30,
  crs: CRS_UTM,
  maxPixels: 1e13
});
