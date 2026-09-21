// ============================================================================
// 07. HUMAN FOOTPRINT COVARIATE (HFI) — AQUITAINE, FRANCE
// ============================================================================
// Human footprint was represented using the 2020 Human Impact Index (HII)
// raster distributed by the Wildlife Conservation Society (WCS) Human
// Footprint project.
//
// IMPORTANT TERMINOLOGICAL NOTE:
// The official WCS source raster is named Human Impact Index (HII) and is
// distributed in Google Earth Engine as:
//   projects/HII/v1/hii
//
// In this study, "HFI" is retained as the abbreviation for the human-footprint
// covariate used throughout the analytical workflow. Therefore:
//   - HFI = study covariate abbreviation
//   - HII = official WCS source raster name
//
// WCS presents these HII products within its broader Human Footprint framework.
// The source HII product is generated at a 300-m spatial scale. Because this
// variable is continuous, bilinear resampling was used to harmonize it with
// the common 30-m working grid employed for the environmental covariates.
//
// The 2020 HII layer was transformed using log(x + 1). Water areas were
// assigned a value of 0.0001 after transformation.
//
// References:
// Wildlife Conservation Society (WCS), Human Footprint Project:
// https://www.wcshumanfootprint.org/data-access
//
// Venter, O. et al. (2016). Sixteen years of change in the global terrestrial
// human footprint and implications for biodiversity conservation.
// Nature Communications, 7, 12558.
// https://doi.org/10.1038/ncomms12558
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. Land-water mask from GHSL SMOD 2020
// Water class = 10
// --------------------------------------------------------------------------
var smod = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020')
  .select('smod_code')
  .clip(region);

var landMask = smod
  .neq(10)
  .unmask(0)
  .clip(region);


// --------------------------------------------------------------------------
// 3. Load WCS Human Impact Index (HII), 2020
// Official source raster used to represent the HFI study covariate.
// --------------------------------------------------------------------------
var hfiCol = ee.ImageCollection('projects/HII/v1/hii');

var hfi2020 = ee.Image(
  hfiCol
    .filterDate('2020-01-01', '2021-01-01')
    .first()
)
  .select('hii')
  .clip(region);


// --------------------------------------------------------------------------
// 4. Log(x + 1) transformation
// --------------------------------------------------------------------------
var hfiLog = hfi2020
  .add(1)
  .log()
  .rename('HFI_2020_log1p');


// --------------------------------------------------------------------------
// 5. Harmonization and water fill
// Source scale: 300 m
// Working/export scale: 30 m
// Resampling: bilinear
// --------------------------------------------------------------------------
var hfiFinal = hfiLog
  .resample('bilinear')
  .unmask(0.0001)
  .where(landMask.eq(0), 0.0001)
  .float()
  .clip(region);


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  hfi2020,
  {min: 500, max: 6000},
  'WCS HII 2020 raw',
  false
);

Map.addLayer(
  hfiFinal,
  {min: 0, max: 8},
  'Human Footprint (HFI) 2020 log1p',
  true
);


// --------------------------------------------------------------------------
// 7. Export to Earth Engine Asset
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: hfiFinal,
  description: 'HFI_2020_log1p_Aq',
  assetId: 'users/omarorellanahn/INRIA/HFI_2020_log1p_Aq',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e13
});
