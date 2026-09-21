// ============================================================================
// 08. GLOBAL HUMAN MODIFICATION (gHM) — AQUITAINE, FRANCE
// ============================================================================
// Global Human Modification (gHM) circa 2016 was obtained from the
// Conservation Science Partners (CSP) global dataset.
//
// gHM is a continuous variable ranging from 0 to 1 that represents cumulative
// human modification from major anthropogenic stressors.
//
// Processing followed two distinct spatial stages:
//
//   1. Native-resolution preparation:
//      - gHM was maintained at its native ~1-km spatial support.
//      - The land-water mask was aligned to the native gHM grid.
//      - Water and remaining missing pixels were assigned 0.0001.
//
//   2. Spatial harmonization:
//      - After obtaining a complete native-resolution raster, bilinear
//        resampling was used to harmonize the continuous gHM layer with the
//        common 30-m working grid.
//
// Exporting at 30 m does not increase the native spatial information content;
// it only provides a common grid for integration with the other covariates.
//
// Reference:
// Kennedy, C.M., Oakleaf, J.R., Theobald, D.M., Baruch-Mordo, S.,
// & Kiesecker, J. (2019). Managing the middle: A shift in conservation
// priorities based on the global human modification gradient.
// Global Change Biology, 25, 811–826.
// https://doi.org/10.1111/gcb.14549
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

Map.centerObject(region, 9);


// --------------------------------------------------------------------------
// 2. Load native Global Human Modification (gHM)
// Native pixel size: ~1,000 m
// --------------------------------------------------------------------------
var ghmRaw = ee.ImageCollection(
  'CSP/HM/GlobalHumanModification'
)
  .first()
  .select('gHM')
  .clip(region);

var ghmProjection = ghmRaw.projection();

print('Native gHM projection:', ghmProjection);
print('Native gHM nominal scale (m):', ghmProjection.nominalScale());


// --------------------------------------------------------------------------
// 3. Land-water mask
// GHSL SMOD class 10 = water.
//
// The categorical mask is aligned to the native gHM grid before being used.
// Nearest-neighbor behavior is retained for this categorical layer.
// --------------------------------------------------------------------------
var smod = ee.Image(
  'JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020'
)
  .select('smod_code');

var waterMaskNative = smod
  .eq(10)
  .unmask(0)
  .reproject({
    crs: ghmProjection
  })
  .clip(region);


// --------------------------------------------------------------------------
// 4. Prepare complete raster at native gHM spatial support
// Water and remaining missing values = 0.0001
// --------------------------------------------------------------------------
var ghmNative = ghmRaw
  .unmask(0.0001)
  .where(waterMaskNative.eq(1), 0.0001)
  .rename('gHM_2016')
  .float()
  .clip(region);


// --------------------------------------------------------------------------
// 5. Harmonization to the common 30-m working grid
// Continuous variable -> bilinear resampling
// --------------------------------------------------------------------------
var ghmFinal = ghmNative
  .resample('bilinear')
  .clip(region);


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  ghmRaw,
  {
    min: 0,
    max: 1,
    palette: ['white', 'yellow', 'orange', 'red']
  },
  'gHM raw (~1 km)',
  false
);

Map.addLayer(
  ghmFinal,
  {
    min: 0,
    max: 1,
    palette: ['white', 'yellow', 'orange', 'red']
  },
  'gHM harmonized to 30 m',
  true
);


// --------------------------------------------------------------------------
// 7. Export to Earth Engine Asset
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: ghmFinal,
  description: 'gHM_2016_Aq',
  assetId: 'users/omarorellanahn/INRIA/gHM_2016_Aq',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e13
});
