// ============================================================================
// 13. PCA OF MONTHLY CHIRPS PRECIPITATION — AQUITAINE, FRANCE
// ============================================================================
// Principal Component Analysis (PCA) was applied to the 12 long-term monthly
// precipitation covariates generated in Script 12.
//
// The monthly precipitation surfaces represent mean monthly precipitation
// for 2000–2024 after kriging interpolation and harmonization to the common
// 30-m working grid.
//
// PCA was computed from the covariance matrix of mean-centered monthly
// precipitation variables across the study-area processing extent.
//
// The first three principal components were exported in the original PCA
// product. Subsequent analytical steps retained the components required for
// the final environmental-covariate set.
//
// Because PCA scores are continuous variables and the source precipitation
// product is stored in EPSG:4326 while the PCA product is exported in
// EPSG:32630, bilinear resampling is explicitly used during final reprojection.
//
// Reference:
// Jolliffe, I.T., & Cadima, J. (2016). Principal component analysis:
// a review and recent developments. Philosophical Transactions of the
// Royal Society A, 374, 20150202.
// https://doi.org/10.1098/rsta.2015.0202
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

Map.centerObject(region, 9);


// --------------------------------------------------------------------------
// 2. Load the 12 monthly precipitation covariates
// --------------------------------------------------------------------------
var monthlyBands = [
  'P_January',
  'P_February',
  'P_March',
  'P_April',
  'P_May',
  'P_June',
  'P_July',
  'P_August',
  'P_September',
  'P_October',
  'P_November',
  'P_December'
];

var precipitation = ee.Image(
  'users/omarorellanahn/INRIA/Precipitation_Monthly_30m_Aquitaine'
)
  .select(monthlyBands)
  .clip(region);

print(
  'Monthly precipitation bands used in PCA:',
  precipitation.bandNames()
);


// --------------------------------------------------------------------------
// 3. Mean-center the monthly precipitation variables
// --------------------------------------------------------------------------
var meanDict = precipitation.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region.geometry(),
  scale: 30,
  maxPixels: 1e12,
  bestEffort: true
});

var means = ee.Image.constant(
  monthlyBands.map(function(band) {
    return meanDict.getNumber(band);
  })
).rename(monthlyBands);

var centered = precipitation.subtract(means);


// --------------------------------------------------------------------------
// 4. Covariance matrix
// --------------------------------------------------------------------------
var covarianceResult = centered
  .toArray()
  .reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e12,
    bestEffort: true
  });

var covarianceArray = ee.Array(
  covarianceResult.get('array')
);


// --------------------------------------------------------------------------
// 5. Eigenvalues and eigenvectors
// --------------------------------------------------------------------------
var eigens = covarianceArray.eigen();

var eigenValues = eigens.slice(1, 0, 1);
var eigenVectors = eigens.slice(1, 1);


// --------------------------------------------------------------------------
// 6. Explained variance
// --------------------------------------------------------------------------
var eigenFlat = eigenValues.project([0]);

var totalVariance = eigenFlat
  .reduce(ee.Reducer.sum(), [0])
  .get([0]);

var explainedVariance = eigenFlat
  .divide(totalVariance)
  .multiply(100);

print('Eigenvalues:', eigenFlat);
print('Explained variance by PC (%):', explainedVariance);


// Print cumulative explained variance.
explainedVariance.toList().getInfo(function(values) {

  var cumulative = 0;

  print('=== EXPLAINED VARIANCE — PRECIPITATION ===');

  values.forEach(function(value, index) {

    cumulative += value;

    print(
      'P_PC' + (index + 1) +
      ': ' + value.toFixed(2) + '%' +
      ' | Cumulative: ' + cumulative.toFixed(2) + '%'
    );

  });

});


// --------------------------------------------------------------------------
// 7. PCA transformation
// --------------------------------------------------------------------------
var pcNames = [
  'P_PC1',
  'P_PC2',
  'P_PC3',
  'P_PC4',
  'P_PC5',
  'P_PC6',
  'P_PC7',
  'P_PC8',
  'P_PC9',
  'P_PC10',
  'P_PC11',
  'P_PC12'
];

var arrayImage = centered.toArray();

var pcList = ee.List.sequence(0, 11).map(function(i) {

  i = ee.Number(i);

  var eigenVector = eigenVectors
    .slice(0, i, i.add(1))
    .project([1]);

  var eigenVectorImage = ee.Image
    .constant(eigenVector.toList())
    .toArray();

  return arrayImage.arrayDotProduct(eigenVectorImage);
});

var pcArray = ee.ImageCollection(pcList)
  .toBands()
  .rename(pcNames);

print('Generated precipitation PCA bands:', pcArray.bandNames());


// --------------------------------------------------------------------------
// 8. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  pcArray.select('P_PC1'),
  {
    min: -240,
    max: 85,
    palette: ['ffffcc', '41b6c4', '0c2c84']
  },
  'Precipitation PC1',
  true
);

Map.addLayer(
  pcArray.select('P_PC2'),
  {
    min: -2,
    max: 2,
    palette: ['8c510a', 'f5f5f5', '01665e']
  },
  'Precipitation PC2',
  false
);

Map.addLayer(
  pcArray.select('P_PC3'),
  {
    min: -1.5,
    max: 1.5,
    palette: ['7b3294', 'f7f7f7', '1b7837']
  },
  'Precipitation PC3',
  false
);


// --------------------------------------------------------------------------
// 9. Export original three-component PCA product
// --------------------------------------------------------------------------
// Source working grid: 30 m, EPSG:4326
// Final PCA grid:     30 m, EPSG:32630
// Continuous PCs -> bilinear resampling during reprojection.

var pcaExport = pcArray
  .select([
    'P_PC1',
    'P_PC2',
    'P_PC3'
  ])
  .resample('bilinear');

Export.image.toAsset({
  image: pcaExport,
  description: 'PCA_Precipitacion_12meses_Aquitania',
  assetId:
    'users/omarorellanahn/INRIA/PCA_Precipitacion_12meses_Aquitania',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e12
});
