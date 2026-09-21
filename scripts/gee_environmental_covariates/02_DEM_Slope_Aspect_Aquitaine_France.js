// ============================================================================
// 02. DEM, SLOPE AND ASPECT — AQUITAINE, FRANCE
// ============================================================================
// Elevation and terrain covariates were derived from the Copernicus DEM GLO-30.
// The 6-km processing extent generated in Script 01 was used to retain spatial
// information beyond the administrative study boundary and minimize edge
// effects during subsequent raster processing.
//
// The Copernicus DEM tiles were mosaicked using their native projection and
// clipped to the processing extent. Slope was derived from the DEM in degrees
// and converted to percent slope, while aspect was retained in degrees
// (0–360°). The three rasters were exported at 30-m spatial resolution in
// WGS 84 / UTM zone 30N (EPSG:32630) to the INRIA Earth Engine Asset folder.
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var roiFC = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);


// --------------------------------------------------------------------------
// 2. Copernicus DEM GLO-30
// --------------------------------------------------------------------------
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');
var proj = demCol.first().projection();

var dem = demCol
  .mosaic()
  .setDefaultProjection(proj)
  .clip(roiFC);


// --------------------------------------------------------------------------
// 3. Terrain covariates
// --------------------------------------------------------------------------

// Slope: degrees -> percent
var slopeDeg = ee.Terrain.slope(dem);
var slopePct = slopeDeg
  .multiply(Math.PI / 180)
  .tan()
  .multiply(100)
  .rename('slope_pct');

// Aspect: 0–360 degrees
var aspect = ee.Terrain.aspect(dem).rename('aspect_deg');


// --------------------------------------------------------------------------
// 4. Visualization
// --------------------------------------------------------------------------
Map.centerObject(roiFC, 7);

Map.addLayer(
  dem,
  {
    min: 34,
    max: 310,
    palette: [
      '081d58','253494','225ea8','1d91c0','41b6c4','7fcdbb',
      'c7e9b4','edf8b1','ffffb2','fecc5c','fd8d3c','f03b20',
      'bd6b3a','8c510a','543005'
    ]
  },
  'Elevation (m)'
);

Map.addLayer(
  slopePct,
  {min: 0, max: 7, palette: ['006837','e6ed36','d73027']},
  'Slope (%)',
  false
);

Map.addLayer(
  aspect,
  {
    min: 0,
    max: 360,
    palette: ['red','yellow','green','cyan','blue','magenta','red']
  },
  'Aspect (degrees)',
  false
);


// --------------------------------------------------------------------------
// 5. Export settings
// --------------------------------------------------------------------------
var SCALE_EXPORT = 30;
var CRS_UTM = 'EPSG:32630';
var exportRegion = roiFC.geometry().bounds(1);


// --------------------------------------------------------------------------
// 6. Export to Earth Engine Assets
// --------------------------------------------------------------------------

// DEM
Export.image.toAsset({
  image: dem.rename('elevation_m'),
  description: 'DEM_GLO30_30m_Aquitaine',
  assetId: 'users/omarorellanahn/INRIA/DEM_GLO30_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// Slope (%)
Export.image.toAsset({
  image: slopePct,
  description: 'SLOPE_pct_30m_Aquitaine',
  assetId: 'users/omarorellanahn/INRIA/SLOPE_pct_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});

// Aspect (degrees)
Export.image.toAsset({
  image: aspect,
  description: 'ASPECT_deg_30m_Aquitaine',
  assetId: 'users/omarorellanahn/INRIA/ASPECT_deg_30m_Aquitaine',
  region: exportRegion,
  scale: SCALE_EXPORT,
  crs: CRS_UTM,
  maxPixels: 1e13
});
