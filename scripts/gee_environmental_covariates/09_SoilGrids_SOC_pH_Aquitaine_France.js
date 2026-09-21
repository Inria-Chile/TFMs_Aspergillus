// ============================================================================
// 09. SOILGRIDS SOC AND pH (0–5 cm) — AQUITAINE, FRANCE
// ============================================================================
// Soil organic carbon (SOC) and soil pH were obtained from SoilGrids 2.0
// mean predictions for the 0–5 cm depth interval.
//
// SoilGrids has a native nominal spatial resolution of 250 m. Processing was
// therefore performed first at 250-m spatial support. Remaining gaps over
// terrestrial areas were progressively filled using focal means. Only after
// obtaining complete 250-m surfaces were the continuous covariates bilinearly
// resampled to the common 30-m working grid.
//
// Unit conversions:
//   SOC: SoilGrids dg/kg -> % organic carbon          : divide by 100
//   pH : SoilGrids pH × 10 -> standard pH units      : divide by 10
//
// For locations outside the terrestrial SoilGrids coverage, study-defined
// aquatic values were assigned after terrestrial gap filling:
//   SOC = 0.00001 %
//   pH  = 7.8
//
// These aquatic values are operational values used to provide complete
// environmental covariates for terrestrial and aquatic sampling locations;
// they are not SoilGrids predictions.
//
// Reference:
// Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, G.B.M.,
// Kempen, B., Ribeiro, E., & Rossiter, D. (2021).
// SoilGrids 2.0: producing soil information for the globe with
// quantified spatial uncertainty.
// SOIL, 7, 217–240.
// https://doi.org/10.5194/soil-7-217-2021
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var CRS = 'EPSG:32630';
var NATIVE_SCALE = 250;
var EXPORT_SCALE = 30;

Map.centerObject(region, 9);


// --------------------------------------------------------------------------
// 2. Operational land-water mask
// --------------------------------------------------------------------------
// SRTM was used in the original workflow as an operational land mask.
// The mask is aligned to the 250-m SoilGrids processing support before use.

var landMask = ee.Image('CGIAR/SRTM90_V4')
  .gt(-100)
  .unmask(0)
  .reproject({
    crs: CRS,
    scale: NATIVE_SCALE
  })
  .clip(region);


// --------------------------------------------------------------------------
// 3. Function for progressive terrestrial gap filling
// --------------------------------------------------------------------------
function fillSoilGrid(image) {

  var filled = image;

  filled = filled.unmask(
    filled.focal_mean({
      radius: 1,
      kernelType: 'circle',
      iterations: 8
    })
  );

  filled = filled.unmask(
    filled.focal_mean({
      radius: 3,
      kernelType: 'circle',
      iterations: 8
    })
  );

  filled = filled.unmask(
    filled.focal_mean({
      radius: 6,
      kernelType: 'circle',
      iterations: 8
    })
  );

  return filled;
}


// ============================================================================
// 4. SOIL ORGANIC CARBON — 0–5 cm
// ============================================================================

// SoilGrids SOC is stored as dg/kg.
// Division by 100 converts directly to percent organic carbon.

var socRaw = ee.Image('projects/soilgrids-isric/soc_mean')
  .select('soc_0-5cm_mean')
  .reproject({
    crs: CRS,
    scale: NATIVE_SCALE
  })
  .divide(100)
  .rename('SOC_0_5cm_pct')
  .clip(region);


// Fill terrestrial gaps at 250-m spatial support.
var socNative = fillSoilGrid(socRaw)
  .unmask(0.00001)
  .where(landMask.eq(0), 0.00001)
  .rename('SOC_0_5cm_pct')
  .float()
  .clip(region);


// Continuous variable: bilinear resampling only after native-scale filling.
var socFinal = socNative
  .resample('bilinear')
  .clip(region);


// ============================================================================
// 5. SOIL pH — 0–5 cm
// ============================================================================

// SoilGrids pH is stored as pH × 10.
// Division by 10 restores standard pH units.

var phRaw = ee.Image('projects/soilgrids-isric/phh2o_mean')
  .select('phh2o_0-5cm_mean')
  .reproject({
    crs: CRS,
    scale: NATIVE_SCALE
  })
  .divide(10)
  .rename('pH_0_5cm')
  .clip(region);


// Fill terrestrial gaps at 250-m spatial support.
var phNative = fillSoilGrid(phRaw)
  .unmask(7.8)
  .where(landMask.eq(0), 7.8)
  .rename('pH_0_5cm')
  .float()
  .clip(region);


// Continuous variable: bilinear resampling only after native-scale filling.
var phFinal = phNative
  .resample('bilinear')
  .clip(region);


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------
var socVis = {
  min: 0,
  max: 8,
  palette: [
    'ffffcc',
    'c7e9b4',
    '7fcdbb',
    '41b6c4',
    '2c7fb8',
    '253494'
  ]
};

var phVis = {
  min: 4.5,
  max: 8.5,
  palette: [
    'd7191c',
    'fdae61',
    'ffffbf',
    'a6d96a',
    '1a9641'
  ]
};

Map.addLayer(
  socFinal,
  socVis,
  'SoilGrids SOC 0–5 cm (%)',
  false
);

Map.addLayer(
  phFinal,
  phVis,
  'SoilGrids pH 0–5 cm',
  true
);


// --------------------------------------------------------------------------
// 7. Export SOC
// Native processing scale: 250 m
// Final working grid: 30 m
// Resampling: bilinear
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: socFinal,
  description: 'SoilGrids_SOC_0_5cm_pct_Aq',
  assetId: 'users/omarorellanahn/INRIA/SoilGrids_SOC_0_5cm_pct_Aq',
  region: region.geometry(),
  scale: EXPORT_SCALE,
  crs: CRS,
  maxPixels: 1e13
});


// --------------------------------------------------------------------------
// 8. Export pH
// Native processing scale: 250 m
// Final working grid: 30 m
// Resampling: bilinear
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: phFinal,
  description: 'SoilGrids_pH_0_5cm_Aquit',
  assetId: 'users/omarorellanahn/INRIA/SoilGrids_pH_0_5cm_Aquit',
  region: region.geometry(),
  scale: EXPORT_SCALE,
  crs: CRS,
  maxPixels: 1e13
});
