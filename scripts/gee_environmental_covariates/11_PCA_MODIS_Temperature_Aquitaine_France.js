// ============================================================================
// 11. PCA OF MONTHLY MODIS TEMPERATURE — AQUITAINE, FRANCE
// ============================================================================
// Principal Component Analysis (PCA) was applied to the 12 monthly temperature
// covariates generated in Script 10.
//
// Only the 12 monthly temperature bands were included. The annual temperature
// layer (T_ANUAL) was intentionally excluded because it is derived directly
// from the same monthly variables and would therefore introduce deterministic
// redundancy into the PCA.
//
// PCA was computed from the covariance matrix of mean-centered monthly
// temperature variables across the study-area processing extent.
//
// The first three principal components were exported in the original PCA
// product. Subsequent analytical steps retained the components required for
// the final covariate set.
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
// 2. Load the 12 monthly temperature covariates
// T_ANUAL is intentionally excluded from PCA.
// --------------------------------------------------------------------------
var monthlyBands = [
  'T_Enero',
  'T_Febrero',
  'T_Marzo',
  'T_Abril',
  'T_Mayo',
  'T_Junio',
  'T_Julio',
  'T_Agosto',
  'T_Septiembre',
  'T_Octubre',
  'T_Noviembre',
  'T_Diciembre'
];

var temperature = ee.Image(
  'users/omarorellanahn/INRIA/Temperature_Monthly_30m_Aq_v21'
)
  .select(monthlyBands)
  .clip(region);

print('Monthly temperature bands used in PCA:', temperature.bandNames());


// --------------------------------------------------------------------------
// 3. Mean-center the monthly temperature variables
// --------------------------------------------------------------------------
var meanDict = temperature.reduceRegion({
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

var centered = temperature.subtract(means);


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

  print('=== EXPLAINED VARIANCE ===');

  values.forEach(function(value, index) {

    cumulative += value;

    print(
      'PC' + (index + 1) +
      ': ' + value.toFixed(2) + '%' +
      ' | Cumulative: ' + cumulative.toFixed(2) + '%'
    );

  });

});


// --------------------------------------------------------------------------
// 7. PCA transformation
// --------------------------------------------------------------------------
var pcNames = [
  'T_PC1',
  'T_PC2',
  'T_PC3',
  'T_PC4',
  'T_PC5',
  'T_PC6',
  'T_PC7',
  'T_PC8',
  'T_PC9',
  'T_PC10',
  'T_PC11',
  'T_PC12'
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

print('Generated PCA bands:', pcArray.bandNames());


// --------------------------------------------------------------------------
// 8. Visualization
// --------------------------------------------------------------------------
Map.addLayer(
  pcArray.select('T_PC1'),
  {
    min: -14,
    max: 18,
    palette: ['2166ac', 'f7f7f7', 'd6604d']
  },
  'Temperature PC1',
  true
);

Map.addLayer(
  pcArray.select('T_PC2'),
  {
    min: -11,
    max: 4,
    palette: ['7b3294', 'f7f7f7', '1b7837']
  },
  'Temperature PC2',
  false
);

Map.addLayer(
  pcArray.select('T_PC3'),
  {
    min: -3,
    max: 8,
    palette: ['e66101', 'f7f7f7', '5e3c99']
  },
  'Temperature PC3',
  false
);


// --------------------------------------------------------------------------
// 9. Export original three-component PCA product
// --------------------------------------------------------------------------
Export.image.toAsset({
  image: pcArray.select([
    'T_PC1',
    'T_PC2',
    'T_PC3'
  ]),
  description: 'PCA_Temperatura_12meses_Aquitania',
  assetId:
    'users/omarorellanahn/INRIA/PCA_Temperatura_12meses_Aquitania',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e12
});
