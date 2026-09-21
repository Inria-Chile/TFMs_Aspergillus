// ============================================================================
// 12. CHIRPS MONTHLY PRECIPITATION — AQUITAINE, FRANCE
// ============================================================================
// Monthly precipitation covariates were derived from CHIRPS v2.0 Daily
// precipitation for the period 2000–2024.
//
// Daily precipitation (mm/day) was summed within each calendar month to obtain
// monthly precipitation totals. For each calendar month, the long-term mean
// across 2000–2024 was subsequently calculated.
//
// CHIRPS v2.0 has a nominal spatial resolution of 0.05 degrees
// (approximately 5.6 km). The 12 long-term monthly precipitation surfaces
// were spatially interpolated using kriging through the Open Earth Engine
// Library (OEEL), reproducing the interpolation configuration used in the
// original analytical workflow.
//
// Because precipitation is a continuous variable, bilinear resampling was
// applied only after interpolation when harmonizing the final surfaces to
// the common 30-m working grid.
//
// Export at 30 m does not imply an increase in the native spatial information
// content of CHIRPS.
//
// Reference:
// Funk, C. et al. (2015). The climate hazards infrared precipitation with
// stations—a new environmental record for monitoring extremes.
// Scientific Data, 2, 150066.
// https://doi.org/10.1038/sdata.2015.66
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var roi = region.geometry();

Map.centerObject(region, 8);


// --------------------------------------------------------------------------
// 2. CHIRPS v2.0 Daily precipitation
// Native spatial resolution: ~0.05 degrees (~5.6 km)
// Units: mm/day
// --------------------------------------------------------------------------
var chirps = ee.ImageCollection(
  'UCSB-CHG/CHIRPS/DAILY'
);

var years = ee.List.sequence(2000, 2024);
var months = ee.List.sequence(1, 12);

var monthNames = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December'
];

var monthNamesEE = ee.List(monthNames);


// --------------------------------------------------------------------------
// 3. Monthly precipitation totals for each year
// --------------------------------------------------------------------------
function monthlyPrecipitation(startDate, endDate) {

  return chirps
    .filterDate(startDate, endDate)
    .select('precipitation')
    .sum()
    .clip(roi);
}


var monthlyImages = years.map(function(year) {

  return months.map(function(month) {

    month = ee.Number(month);

    var monthIndex = month.subtract(1);

    var startDate = ee.Date.fromYMD(
      year,
      month,
      1
    );

    var endDate = startDate.advance(
      1,
      'month'
    );

    var precipitation = monthlyPrecipitation(
      startDate,
      endDate
    );

    return precipitation.set({
      'year': year,
      'month': month,
      'month_name': monthNamesEE.get(monthIndex),
      'system:time_start': startDate.millis()
    });

  });

}).flatten();


var monthlyCollection = ee.ImageCollection.fromImages(
  monthlyImages
);


// --------------------------------------------------------------------------
// 4. Long-term monthly means, 2000–2024
// --------------------------------------------------------------------------
var monthlyMeans = ee.Image.cat(

  monthNames.map(function(monthName, index) {

    var month = ee.Number(index).add(1);

    return monthlyCollection
      .filter(
        ee.Filter.eq('month', month)
      )
      .mean()
      .rename('P_' + monthName);

  })
);


// Long-term annual precipitation calculated as the sum of the 12 climatological
// monthly means. This layer is used for visualization and is not exported here.
var annualPrecipitation = monthlyMeans
  .reduce(ee.Reducer.sum())
  .rename('P_ANNUAL');


// --------------------------------------------------------------------------
// 5. Native CHIRPS projection and sampling scale
// --------------------------------------------------------------------------
var chirpsProjection = ee.Image(
  chirps.first()
)
  .select('precipitation')
  .projection();

var chirpsScale = chirpsProjection.nominalScale();

print(
  'CHIRPS nominal scale (m):',
  chirpsScale
);


// --------------------------------------------------------------------------
// 6. Sample the 12 monthly climatological surfaces
// --------------------------------------------------------------------------
var samplePoints = monthlyMeans.sample({
  region: roi,
  scale: chirpsScale,
  projection: chirpsProjection,
  geometries: true
});

print(
  'Number of points used for kriging:',
  samplePoints.size()
);


// --------------------------------------------------------------------------
// 7. Monthly kriging
// --------------------------------------------------------------------------
// The covariance function, search radius, and 5.5-km interpolation raster
// reproduce the configuration used in the original analytical workflow.

var oeel = require(
  'users/OEEL/lib:loadAll'
);

var interpolationProjection = ee.Projection(
  'EPSG:4326'
);


function krigeMonth(monthName) {

  var bandName = 'P_' + monthName;

  var monthlyRaster = samplePoints
    .reduceToImage(
      [bandName],
      ee.Reducer.first()
    )
    .reproject(
      interpolationProjection.atScale(5500)
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


// Generate the 12 interpolated monthly bands.
var krigedBands = monthNames.map(
  krigeMonth
);

var monthlyKriging = ee.Image
  .cat(krigedBands)
  .toFloat();

print(
  'Kriged monthly precipitation bands:',
  monthlyKriging.bandNames()
);


// --------------------------------------------------------------------------
// 8. Harmonization to the common 30-m working grid
// --------------------------------------------------------------------------
// Precipitation is continuous, therefore bilinear resampling is used during
// final reprojection to the 30-m working grid.

var precipitation30m = monthlyKriging
  .clip(roi)
  .resample('bilinear');


// --------------------------------------------------------------------------
// 9. QA/QC visualization of the interpolation
// --------------------------------------------------------------------------
// January is reconstructed separately for map inspection. This avoids asking
// Earth Engine to render the complete 12-band processing graph merely to
// inspect one interpolated month.

var januaryRaw = monthlyMeans
  .select('P_January');


// Sample January alone to keep the visualization graph compact.
var januaryPoints = januaryRaw.sample({
  region: roi,
  scale: chirpsScale,
  projection: chirpsProjection,
  geometries: true
});


var januaryRaster = januaryPoints
  .reduceToImage(
    ['P_January'],
    ee.Reducer.first()
  )
  .reproject(
    interpolationProjection.atScale(5500)
  )
  .rename('P_January');


var januaryCovariance = function(distance) {

  return distance
    .multiply(-0.1)
    .exp()
    .multiply(140);

};


var januaryKriged = oeel.Image.kriging({
  covFun: januaryCovariance,
  radius: 15,
  image: januaryRaster
})
  .select('estimate')
  .rename('P_January')
  .clip(roi);


// Final QA view after bilinear harmonization.
var january30m = januaryKriged
  .resample('bilinear')
  .clip(roi);


// --------------------------------------------------------------------------
// 10. Visualization
// --------------------------------------------------------------------------
var precipPalette = [
  '7f1c00',
  'c92c00',
  'ff3800',
  'ffbd59',
  'ffefb3',
  'd9ffd3',
  'c9fffd',
  'a6cdff',
  '5d61ff',
  '1e22d1',
  '161998'
];


var annualVis = {
  min: 500,
  max: 1600,
  palette: precipPalette
};


var monthlyVis = {
  min: 30,
  max: 150,
  palette: precipPalette
};


// Original January climatology.
Map.addLayer(
  januaryRaw,
  monthlyVis,
  'January - original CHIRPS climatology',
  false
);


// January after kriging.
Map.addLayer(
  januaryKriged,
  monthlyVis,
  'January - kriging',
  true
);


// January after kriging and final bilinear harmonization.
Map.addLayer(
  january30m,
  monthlyVis,
  'January - kriging + bilinear',
  false
);


// Annual climatological precipitation.
Map.addLayer(
  annualPrecipitation,
  annualVis,
  'Mean annual precipitation 2000-2024',
  false
);


// Sampling points used only for January QA/QC.
Map.addLayer(
  januaryPoints,
  {color: 'black'},
  'January kriging sample points',
  false
);


// --------------------------------------------------------------------------
// 11. Export 12-band monthly precipitation product
// --------------------------------------------------------------------------
// Source: CHIRPS ~5.6 km
// Interpolation: OEEL kriging
// Final resampling: bilinear
// Working grid: 30 m, EPSG:4326

Export.image.toAsset({
  image: precipitation30m,
  description: 'Precipitation_Monthly_Kriging_30m_Aquitaine',
  assetId:
    'users/omarorellanahn/INRIA/Precipitation_Monthly_30m_Aquitaine',
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
