// ============================================================================
// 14. DISTANCE TO SEA — AQUITAINE, FRANCE
// ============================================================================
// Distance to the sea was calculated as cumulative geodetic distance from
// each pixel to the nearest marine-source pixel.
//
// Marine areas were delineated as the portion of the processing extent lying
// outside the land polygons of France and Spain. A manually delineated
// estuarine polygon (Estuario_poli) was incorporated to preserve the
// coastline and estuarine configuration used in the analytical workflow.
//
// A constant-cost raster (cost = 1) was used with ee.Image.cumulativeCost().
// With geodeticDistance = true, cumulative cost represents distance in meters.
//
// The distance surface was calculated directly on the 30-m working grid in
// WGS 84 / UTM zone 30N (EPSG:32630). No bilinear resampling was required,
// because the covariate was generated directly at the target resolution.
//
// Maximum search distance: 200 km.
// ============================================================================


var Estuario_poli =
    /* color: #d63000 */
    /* shown: false */
    ee.Geometry.Polygon(
        [[[-1.0611791283725314, 45.56915699670993],
          [-1.0628957421420626, 45.568015358921194],
          [-1.0646981866000704, 45.567474575027795],
          [-1.0707921654819064, 45.558580937578895],
          [-1.0771573961377845, 45.549758210421096],
          [-1.1018766344190345, 45.51993756072705],
          [-0.9961332262159095, 45.44387150367634],
          [-0.8780301988721595, 45.34164683636334],
          [-0.7992536956591789, 45.162243446243636],
          [-0.6550581390185539, 45.03816497607174],
          [-0.5767805511279289, 44.959507262267024],
          [-0.5603010589404289, 44.89630878234965],
          [-0.3157434316688512, 44.901172672545854],
          [-0.2937707754188512, 44.93423621439334],
          [-0.4434594961219762, 45.00613227541598],
          [-0.6027612539344762, 45.115745618733214],
          [-0.6813239332895371, 45.23103397672018],
          [-0.6881903883676621, 45.299176440353044],
          [-0.6627845045785996, 45.408227073781575],
          [-0.8058490339398117, 45.54833235711842],
          [-0.9666096507384858, 45.60879808720806],
          [-0.9937321482970796, 45.62200602894687],
          [-1.0275371769751551, 45.62685676651902],
          [-1.031957457431698, 45.623765453791655],
          [-1.032412246115031, 45.61985186616853],
          [-1.0317685159514567, 45.61841109561401],
          [-1.0308672937224528, 45.61763066277254],
          [-1.0300519021819254, 45.61727045933599],
          [-1.0297085794280192, 45.616880236335255],
          [-1.0298802408049723, 45.61627988795467],
          [-1.0410948206580506, 45.59984313145322],
          [-1.0545097427442252, 45.58013586006182],
          [-1.0577970816707771, 45.572668230158556],
          [-1.0601145102596443, 45.56978422814913]]]);

// --------------------------------------------------------------------------
// 1. Settings and study-area processing extent
// --------------------------------------------------------------------------
var SCALE = 30;
var CRS_UTM = 'EPSG:32630';
var MAX_DISTANCE = 200000;  // meters

var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. Land geometry
// --------------------------------------------------------------------------
// National polygons define the general coastline.
//
// Estuario_poli corresponds to the manually delineated estuarine geometry
// imported/drawn in the Earth Engine Code Editor.

var countryLand = ee.FeatureCollection(
  'USDOS/LSIB_SIMPLE/2017'
)
  .filter(
    ee.Filter.inList(
      'country_na',
      ['France', 'Spain']
    )
  );


// Combine national land polygons with the estuarine geometry.
var landGeometry = countryLand
  .merge(Estuario_poli)
  .union(100)
  .geometry();


// --------------------------------------------------------------------------
// 3. Marine-source geometry
// --------------------------------------------------------------------------
// Sea = area within the processing extent that is outside the land geometry.

var seaGeometry = roi.difference(
  landGeometry,
  100
);


// --------------------------------------------------------------------------
// 4. Working grid and source rasters
// --------------------------------------------------------------------------
// Constant cost raster generated directly on the final 30-m UTM grid.

var costImage = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  });


// Marine pixels used as cumulative-cost sources.

var seaSource = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .clip(seaGeometry)
  .selfMask()
  .rename('sea');


// Mask restricting the final covariate to the study-area processing extent.

var roiMask = ee.Image.constant(1)
  .reproject({
    crs: CRS_UTM,
    scale: SCALE
  })
  .clip(roi)
  .selfMask();


// --------------------------------------------------------------------------
// 5. Geodetic distance to sea
// --------------------------------------------------------------------------
var distSeaRaw = costImage
  .cumulativeCost({
    source: seaSource,
    maxDistance: MAX_DISTANCE,
    geodeticDistance: true
  })
  .rename('dist_sea_m');


// Final distance covariate restricted to the study area.

var distSea = distSeaRaw
  .updateMask(roiMask)
  .rename('dist_sea_m');


// --------------------------------------------------------------------------
// 6. Visualization
// --------------------------------------------------------------------------
var distanceVis = {
  min: 0,
  max: 150000,
  palette: [
    '0000FF',
    '00AAFF',
    '00FFAA',
    'FFFF00',
    'FF8800',
    'FF0000'
  ]
};


// Marine-source pixels.
Map.addLayer(
  seaSource,
  {
    palette: ['0000FF'],
    opacity: 0.5
  },
  'Marine source',
  false
);


// Full cumulative-distance surface.
// Useful for checking distance propagation beyond the final ROI mask.
Map.addLayer(
  distSeaRaw,
  distanceVis,
  'Distance to sea - full calculation',
  false
);


// Final covariate used in the study.
Map.addLayer(
  distSea,
  distanceVis,
  'Distance to sea',
  true
);


// Study-area boundary.
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
// Distance is generated directly at 30 m; no additional resampling is applied.

Export.image.toAsset({
  image: distSea,
  description: 'Sea_Distance_Aq',
  assetId: 'users/omarorellanahn/INRIA/Sea_Distance_Aq',
  region: roi,
  crs: CRS_UTM,
  scale: SCALE,
  maxPixels: 1e13,
  pyramidingPolicy: {
    '.default': 'mean'
  }
});
