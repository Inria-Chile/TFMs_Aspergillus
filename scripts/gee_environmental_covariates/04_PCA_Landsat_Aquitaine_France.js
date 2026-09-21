// ============================================================================
// 04. PCA OF LANDSAT 8/9 SURFACE REFLECTANCE — AQUITAINE, FRANCE
// ============================================================================
// Principal Component Analysis (PCA) was applied to the six optical bands of
// the 2024 Landsat 8/9 surface-reflectance composite generated in Script 03.
//
// PCA was based on the covariance matrix of mean-centered reflectance bands.
// Eigenvalues and eigenvectors were used to quantify explained variance and
// project the original six-band dataset into orthogonal principal components.
//
// Only LS_PC1 and LS_PC2 were retained as final environmental covariates,
// because together they captured most of the spectral variance while reducing
// dimensionality and redundancy among the original Landsat bands.
//
// Scientific basis:
// - Jolliffe & Cadima (2016): Principal Component Analysis and dimensionality
//   reduction using covariance/eigen decomposition.
// ============================================================================


// --------------------------------------------------------------------------
// 1. Study-area processing extent and Landsat composite
// --------------------------------------------------------------------------
var region = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);

var bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];

var lsImage = ee.Image(
  'users/omarorellanahn/INRIA/L89_2024_Mosaico'
).select(bands).clip(region);


// --------------------------------------------------------------------------
// 2. Mean-center the six Landsat bands
// --------------------------------------------------------------------------
var meanDict = lsImage.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: region.geometry(),
  scale: 30,
  maxPixels: 1e12,
  bestEffort: true
});

var means = ee.Image.constant(
  bands.map(function(b) {
    return meanDict.getNumber(b);
  })
).rename(bands);

var centered = lsImage.subtract(means);


// --------------------------------------------------------------------------
// 3. Covariance matrix
// --------------------------------------------------------------------------
var covarResult = centered.toArray().reduceRegion({
  reducer: ee.Reducer.centeredCovariance(),
  geometry: region.geometry(),
  scale: 30,
  maxPixels: 1e12,
  bestEffort: true
});

var covarArray = ee.Array(covarResult.get('array'));


// --------------------------------------------------------------------------
// 4. Eigenvalues and eigenvectors
// --------------------------------------------------------------------------
var eigens = covarArray.eigen();

var eigenValues  = eigens.slice(1, 0, 1);
var eigenVectors = eigens.slice(1, 1);


// --------------------------------------------------------------------------
// 5. Explained variance
// --------------------------------------------------------------------------
var eigenFlat = eigenValues.project([0]);
var totalVar = eigenFlat.reduce(ee.Reducer.sum(), [0]).get([0]);
var varPct = eigenFlat.divide(totalVar).multiply(100);

print('Explained variance by PC (%):', varPct);

varPct.toList().getInfo(function(values) {
  var cumulative = 0;

  print('=== LANDSAT PCA EXPLAINED VARIANCE ===');

  values.forEach(function(value, i) {
    cumulative += value;

    print(
      'LS_PC' + (i + 1) +
      ': ' + value.toFixed(2) + '%' +
      ' | Cumulative: ' + cumulative.toFixed(2) + '%'
    );
  });
});


// --------------------------------------------------------------------------
// 6. PCA projection: six bands -> six principal components
// --------------------------------------------------------------------------
var pcNames = [
  'LS_PC1', 'LS_PC2', 'LS_PC3',
  'LS_PC4', 'LS_PC5', 'LS_PC6'
];

var arrayImage = centered.toArray();

var pcList = ee.List.sequence(0, 5).map(function(i) {

  i = ee.Number(i);

  var vector = eigenVectors
    .slice(0, i, i.add(1))
    .project([1]);

  var vectorImage = ee.Image.constant(
    vector.toList()
  ).toArray();

  return arrayImage.arrayDotProduct(vectorImage);
});

var pcArray = ee.ImageCollection(pcList)
  .toBands()
  .rename(pcNames);


// --------------------------------------------------------------------------
// 7. Visualization of retained components
// --------------------------------------------------------------------------
Map.centerObject(region, 9);

Map.addLayer(
  pcArray.select('LS_PC1'),
  {
    min: -0.3,
    max: 0.3,
    palette: ['2166ac', 'f7f7f7', 'd6604d']
  },
  'Landsat PC1'
);

Map.addLayer(
  pcArray.select('LS_PC2'),
  {
    min: -0.1,
    max: 0.1,
    palette: ['7b3294', 'f7f7f7', '1b7837']
  },
  'Landsat PC2',
  false
);


// --------------------------------------------------------------------------
// 8. Export PCA product
// --------------------------------------------------------------------------
// Three components were exported in the original PCA raster.
// LS_PC1 and LS_PC2 were subsequently retained as final covariates.

Export.image.toAsset({
  image: pcArray.select(['LS_PC1', 'LS_PC2', 'LS_PC3']),
  description: 'PCA_Landsat_2024_Aquitania',
  assetId: 'users/omarorellanahn/INRIA/PCA_Landsat_2024_Aquitania',
  region: region.geometry(),
  scale: 30,
  crs: 'EPSG:32630',
  maxPixels: 1e12
});
