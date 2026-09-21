// ============================================================================
// 10. MODIS MONTHLY LAND SURFACE TEMPERATURE — AQUITAINE, FRANCE
// ============================================================================
// Monthly daytime land-surface temperature (LST) covariates were derived
// from the MODIS MOD21C3 Collection 6.1 monthly product.
//
// Processing period: 2020–2025.
//
// For each calendar month, the mean LST across the six-year period was
// calculated. MOD21C3 scale factors and offsets are already applied to the
// assets ingested into Google Earth Engine; therefore, no additional scale
// factor was applied. Temperature was converted from Kelvin to degrees
// Celsius by subtracting 273.15.
//
// Coastal NoData areas were progressively filled on the original MOD21C3
// grid before spatial interpolation. The completed monthly surfaces were
// sampled and interpolated using kriging through the Open Earth Engine
// Library (OEEL), using the same covariance function and interpolation
// parameters employed in the original analytical workflow.
//
// The annual temperature layer (T_ANUAL) was calculated as the mean of the
// 12 kriged monthly temperature layers.
//
// As temperature is continuous, bilinear resampling was applied only after
// interpolation, when harmonizing the final surfaces to the common 30-m
// working grid. Export at 30 m does not imply 30-m native information.
//
// MODIS product:
// MOD21C3.061 Terra Land Surface Temperature and 3-Band Emissivity
// Monthly L3 Global CMG.
// DOI: https://doi.org/10.5067/MODIS/MOD21C3.061
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var roiFC = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

// Additional 10-km processing buffer used during filling and interpolation
// to reduce boundary effects around the final study extent.
var processingRegion = roiFC.geometry().buffer(10000);

Map.centerObject(roiFC, 8);


// --------------------------------------------------------------------------
// 2. MODIS monthly LST collection
// --------------------------------------------------------------------------
var modisLST = ee.ImageCollection('MODIS/061/MOD21C3');

var years  = ee.List.sequence(2020, 2025);
var months = ee.List.sequence(1, 12);

var monthNames = [
  'Enero',
  'Febrero',
  'Marzo',
  'Abril',
  'Mayo',
  'Junio',
  'Julio',
  'Agosto',
  'Septiembre',
  'Octubre',
  'Noviembre',
  'Diciembre'
];

var monthNamesEE = ee.List(monthNames);


// --------------------------------------------------------------------------
// 3. Retrieve monthly MOD21C3 images
// --------------------------------------------------------------------------
function getMonthlyTemperature(startDate, endDate) {

  var image = modisLST
    .filterDate(startDate, endDate)
    .first();

  return ee.Algorithms.If(
    image,
    ee.Image(image)
      .select('LST_Day')
      .clip(processingRegion),
    null
  );
}


// --------------------------------------------------------------------------
// 4. Build monthly image collection for 2020–2025
// --------------------------------------------------------------------------
var monthlyImages = years.map(function(year) {

  return months.map(function(month) {

    var monthIndex = ee.Number(month).subtract(1);
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate = startDate.advance(1, 'month');

    var temperature = ee.Image(
      getMonthlyTemperature(startDate, endDate)
    );

    return ee.Algorithms.If(
      temperature,
      temperature.set({
        'year': year,
        'month': month,
        'month_name': monthNamesEE.get(monthIndex),
        'system:time_start': startDate.millis()
      }),
      null
    );

  });

}).flatten();

var validMonthlyImages = ee.List(monthlyImages)
  .map(function(image) {
    return ee.Algorithms.If(image, image, null);
  })
  .removeAll([null]);

var monthlyCollection = ee.ImageCollection.fromImages(
  validMonthlyImages
);


// --------------------------------------------------------------------------
// 5. Mean temperature for each calendar month
// --------------------------------------------------------------------------
// MOD21C3 values in Earth Engine already include the MODIS scale factor.
// Values are therefore in Kelvin and only require conversion to °C.

var monthlyMeanK = ee.Image.cat(

  monthNames.map(function(monthName, index) {

    var month = ee.Number(index).add(1);

    return monthlyCollection
      .filter(ee.Filter.eq('month', month))
      .mean()
      .rename('T_' + monthName);

  })
);

var monthlyMeanC = monthlyMeanK
  .subtract(273.15);


// --------------------------------------------------------------------------
// 6. Fill coastal NoData on the source MOD21C3 grid
// --------------------------------------------------------------------------
// Gap filling is performed before kriging and before final resampling.
// Radius is expressed in source-image pixels.

var FILL_RADIUS = 5;
var FILL_ITERATIONS = 5;

var monthlyFilled = ee.Image.cat(

  monthNames.map(function(monthName) {

    var band = monthlyMeanC.select('T_' + monthName);
    var filled = band;

    for (var i = 0; i < FILL_ITERATIONS; i++) {

      var localMean = filled.focalMean({
        radius: FILL_RADIUS,
        units: 'pixels',
        kernelType: 'circle'
      });

      filled = filled.unmask(localMean);
    }

    return filled.rename('T_' + monthName);

  })
);


// --------------------------------------------------------------------------
// 7. Sample completed monthly surfaces for kriging
// --------------------------------------------------------------------------
var modisProjection = ee.Image(modisLST.first())
  .select('LST_Day')
  .projection();

var modisScale = modisProjection.nominalScale();

var samplingImage = monthlyFilled
  .addBands(ee.Image.pixelLonLat());

var points = samplingImage.sample({
  region: processingRegion,
  scale: modisScale,
  projection: modisProjection,
  geometries: true
});


// --------------------------------------------------------------------------
// 8. Monthly kriging
// --------------------------------------------------------------------------
// The 400-m grid and covariance parameters below reproduce the interpolation
// configuration used in the original workflow.

var oeel = require('users/OEEL/lib:loadAll');

var krigingProjection = ee.Projection('EPSG:32630');

function krigeMonth(monthName) {

  var bandName = 'T_' + monthName;

  var monthlyRaster = points
    .reduceToImage(
      [bandName],
      ee.Reducer.first()
    )
    .reproject(
      krigingProjection.atScale(400)
    )
    .rename(bandName);

  var covarianceFunction = function(distance) {
    return distance
      .multiply(-0.1)
      .exp()
      .multiply(140);
  };

  return oeel.Image.kriging({
    covFun: covarianceFunction,
    radius: 15,
    image: monthlyRaster
  })
    .select('estimate')
    .rename(bandName);
}


// --------------------------------------------------------------------------
// 9. Combine monthly kriging results
// --------------------------------------------------------------------------
var krigedBands = monthNames.map(krigeMonth);

var monthlyKriging = ee.Image
  .cat(krigedBands)
  .toFloat();


// --------------------------------------------------------------------------
// 10. Annual mean temperature
// --------------------------------------------------------------------------
var annualTemperature = monthlyKriging
  .reduce(ee.Reducer.mean())
  .rename('T_ANUAL');

var temperatureFinal = monthlyKriging
  .addBands(annualTemperature);


// --------------------------------------------------------------------------
// 11. Harmonization to the common 30-m working grid
// --------------------------------------------------------------------------
// Temperature is continuous:
// final reprojection/resampling -> bilinear.

var temperature30m = temperatureFinal
  .clip(roiFC)
  .resample('bilinear');


// --------------------------------------------------------------------------
// 12. Visualization
// --------------------------------------------------------------------------
var temperatureVis = {
  min: 5,
  max: 30,
  palette: [
    '040274',
    '315aff',
    '7de0ea',
    'b3f3b3',
    'fff49c',
    'ffc56e',
    'ff8e54',
    'f93306',
    '800000'
  ]
};

Map.addLayer(
  temperature30m.select('T_Abril'),
  temperatureVis,
  'April temperature',
  true
);

Map.addLayer(
  temperature30m.select('T_ANUAL'),
  temperatureVis,
  'Annual temperature',
  false
);


// --------------------------------------------------------------------------
// 13. Export final 13-band temperature product
// --------------------------------------------------------------------------
// Bands:
// T_Enero ... T_Diciembre + T_ANUAL
//
// Final Asset verified in the analytical workflow:
// EPSG:4326, nominal pixel size = 30 m.

Export.image.toAsset({
  image: temperature30m,
  description: 'Temperature_Monthly_30m_Aq_v21',
  assetId:
    'users/omarorellanahn/INRIA/Temperature_Monthly_30m_Aq_v21',
  region: roiFC.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
