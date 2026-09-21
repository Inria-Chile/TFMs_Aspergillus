// ============================================================================
// 01. STUDY AREA DEFINITION — AQUITAINE, FRANCE
// ============================================================================
// The study area was defined from second-level administrative units (ADM2)
// of FAO GAUL 2015. In France, ADM2 units correspond to departments.
// The four departments containing the field-sampling locations were merged
// and surrounded by a 6-km buffer. The buffer was used only as an auxiliary
// processing extent to reduce edge effects during raster clipping, resampling,
// neighborhood operations, and processing of coarse-resolution covariates,
// ensuring complete environmental information for samples near boundaries.
// ============================================================================


// 1. GAUL ADM2 departments containing field samples
var admin2 = ee.FeatureCollection('FAO/GAUL/2015/level2');

var deps = [
  'Pyrenees-Atlantique',
  'Landes',
  'Gironde',
  'Charente-Maritime'
];

var studyADM2 = admin2
  .filter(ee.Filter.eq('ADM0_NAME', 'France'))
  .filter(ee.Filter.inList('ADM2_NAME', deps));


// 2. Merge departments and create 6-km processing buffer
var studyArea = studyADM2.geometry();
var studyAreaBuffer = studyArea.buffer(6000);

var studyAreaBufferFC = ee.FeatureCollection([
  ee.Feature(studyAreaBuffer)
]);


// 3. Existing study-area asset
var originalBuffer = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
);


// 4. Field samples
var samples50 = ee.FeatureCollection(
  'users/omarorellanahn/INRIA/Sampling_50_sites_EMERG_2024_INRIA'
);


// 5. Check whether reconstructed and existing buffers are identical
var difference = originalBuffer.geometry()
  .symmetricDifference(studyAreaBuffer, 1)
  .area(1);

print('Selected departments:', studyADM2.aggregate_array('ADM2_NAME'));
print('Number of samples:', samples50.size());
print('Spatial difference with existing buffer (m²):', difference);


// 6. Visualization
Map.centerObject(studyADM2, 7);
Map.addLayer(studyADM2, {color:'red'}, 'GAUL ADM2 departments');
Map.addLayer(studyAreaBufferFC, {color:'yellow'}, '6-km processing buffer');
Map.addLayer(samples50, {color:'blue'}, 'Field samples');


// 7. Export — EXACT original asset path
// IMPORTANT: Earth Engine will NOT overwrite the asset if it already exists.
// Run this export only if the original asset has first been removed.
Export.table.toAsset({
  collection: studyAreaBufferFC,
  description: 'Buffer_6km_Part_Aquitania_France',
  assetId: 'users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France'
});
