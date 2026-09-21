// ============================================================================
// 06. HUMAN IMPACT COVARIATES — POPULATION DENSITY AND BUILT VOLUME
// AQUITAINE, FRANCE
// ============================================================================
// Two human-impact covariates were generated for the Aquitaine study area:
//
// 1) Population density from WorldPop 2020
//    - original variable: population count per pixel
//    - derived variable: population density (inhabitants/km²)
//    - transformation: log(x + 1)
//
// 2) Built volume from GHSL BUILT-V 2020
//    - original variable: built_volume_total
//    - unit interpretation: m³ per 100-m pixel, numerically equivalent to m³/ha
//    - transformation: log(x + 1)
//
// Water areas were identified using GHSL SMOD 2020 and assigned a constant
// fill value of 0.0001 after transformation. Both layers were harmonized to
// the 30-m working grid using bilinear resampling and exported in
// WGS 84 / UTM zone 30N (EPSG:32630).
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

Map.centerObject(region, 9);


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
// 3. Population density from WorldPop 2020
// --------------------------------------------------------------------------
var wp = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'FRA'))
  .filter(ee.Filter.eq('year', 2020));

var popCount = ee.Image(wp.first())
  .select('population')
  .clip(region)
  .rename('pop_count');

var pixelAreaKm2 = ee.Image.pixelArea().divide(1e6);

var popDens = popCount
  .divide(pixelAreaKm2)
  .rename('pop_dens_hab_km2');

var popDensLog = popDens
  .add(1)
  .log()
  .rename('pop_dens_log1p');

var popDensOut = popDensLog
  .resample('bilinear')
  .unmask(0.0001)
  .where(landMask.eq(0), 0.0001)
  .float()
  .clip(region);


// --------------------------------------------------------------------------
// 4. Built volume from GHSL BUILT-V 2020
// --------------------------------------------------------------------------
var builtVol = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_V/2020')
  .select('built_volume_total')
  .clip(region)
  .rename('built_vol_m3_ha');

var builtVolLog = builtVol
  .add(1)
  .log()
  .rename('built_vol_log1p');

var builtVolOut = builtVolLog
  .resample('bilinear')
  .unmask(0.0001)
  .where(landMask.eq(0), 0.0001)
  .float()
  .clip(region);


// --------------------------------------------------------------------------
// 5. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  popDensOut,
  {min: 0, max: 8},
  'Population density log1p',
  false
);

Map.addLayer(
  builtVolOut,
  {min: 0, max: 12},
  'Built volume log1p',
  true
);


// --------------------------------------------------------------------------
// 6. Export to Earth Engine Assets
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: popDensOut,
  description: 'WorldPop_pop_dens_log1p_2020_Aq',
  assetId: 'users/omarorellanahn/INRIA/WorldPop_pop_dens_log1p_2020_Aq',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e13
});

Export.image.toAsset({
  image: builtVolOut,
  description: 'GHSL_built_vol_log1p_2020_Aq',
  assetId: 'users/omarorellanahn/INRIA/GHSL_built_vol_log1p_2020_Aq',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e13
});
