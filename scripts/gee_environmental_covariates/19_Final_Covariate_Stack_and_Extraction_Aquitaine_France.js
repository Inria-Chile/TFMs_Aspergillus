// ============================================================================
// 19. FINAL ENVIRONMENTAL COVARIATE STACK AND SAMPLE EXTRACTION
// AQUITAINE, FRANCE
// ============================================================================
// This script integrates the environmental covariates generated in Scripts
// 01–18 and prepares the final covariate dataset used in the analytical
// workflow.
//
// Raster products previously generated in Google Earth Engine or externally
// in SAGA GIS are loaded from the project Asset repository. Deterministic
// transformations required for the final analytical variables are performed
// here, including:
//
//   - Aspect (degrees)        -> sine and cosine
//   - Flow direction (0–360) -> sine and cosine
//   - Flow accumulation      -> log10(x + 1)
//
// Flow length and curvature derivatives generated in SAGA GIS are retained
// directly after band renaming.
//
// PCA-derived climate and Landsat covariates are reduced to the principal
// components retained for the final analytical dataset.
//
// The final covariate stack contains 36 continuous environmental,
// topographic, climatic, atmospheric, soil, land-cover, distance, and
// anthropogenic variables.
//
// Covariate values are extracted at the 50 field-sampling locations using
// ee.Image.sampleRegions() at a 30-m working scale and exported as a CSV file
// for subsequent statistical and machine-learning analyses.
//
// NOTE:
// Some candidate covariates were generated and evaluated during preprocessing
// but were not retained in the final 36-variable dataset. In particular,
// CO_mean, landforms, and geomorphons are excluded from the final stack.
// ============================================================================
// -------------------------
// 1) Load ROI + Samples
// -------------------------
var roiFC     = ee.FeatureCollection("users/omarorellanahn/INRIA/Buffer_6km_Part_Aquitania_France");
var roi       = roiFC.geometry();
var samples50 = ee.FeatureCollection("users/omarorellanahn/INRIA/Sampling_50_sites_EMERG_2024_INRIA");

// -------------------------
// 2) Load Raster Assets
// -------------------------
// Topography
var dem    = ee.Image("users/omarorellanahn/INRIA/DEM_GLO30_30m_Aquitaine");//1
var slope  = ee.Image("users/omarorellanahn/INRIA/SLOPE_pct_30m_Aquitaine");//2
var aspect = ee.Image("users/omarorellanahn/INRIA/ASPECT_deg_30m_Aquitaine");//sin and cos 3, 4

// Hydrology
var flowDirD8  = ee.Image("users/omarorellanahn/INRIA/Flow_Direction_Aq");//No
var flowDir360 = ee.Image("users/omarorellanahn/INRIA/Flow_Direction_Aq360");//sin and cos 5, 6
var flowAcc    = ee.Image("users/omarorellanahn/INRIA/Flow_Accumulation_Aq");//Log 7
var flowLen    = ee.Image("users/omarorellanahn/INRIA/Flow_Length_Aq");//8
var distSea    = ee.Image("users/omarorellanahn/INRIA/Sea_Distance_Aq");//9
var distRivers = ee.Image("users/omarorellanahn/INRIA/Distance_to_Rivers_30m_Aquitaine");//10
var urbanDist  = ee.Image("users/omarorellanahn/INRIA/Urban1km2_Distance_Aq");//11

// Landforms / derivados del terreno
var curvature   = ee.Image("users/omarorellanahn/INRIA/Curvature_Aquitaine");//12
var planCurv    = ee.Image("users/omarorellanahn/INRIA/Plan_Curvature_Aq");//13
var profileCurv = ee.Image("users/omarorellanahn/INRIA/Profile_Curvature_Aq");//14
var landforms   = ee.Image("users/omarorellanahn/INRIA/LandForms_Aq");//15
var geomorphons = ee.Image("users/omarorellanahn/INRIA/Geomorphons_Aq");//16

// Dynamic World (10m)
var dwTrees = ee.Image("users/omarorellanahn/INRIA/DW_2024_Prob_Bosque_Trees_10m");//17
var dwBuilt = ee.Image("users/omarorellanahn/INRIA/DW_2024_Prob_Built_10m");//18
var dwCrops = ee.Image("users/omarorellanahn/INRIA/DW_2024_Prob_Cultivos_Crops_10m");//19

// Landsat
var l89NDFI         = ee.Image("users/omarorellanahn/INRIA/L89_2024_NDFI_30m");//20
var PCA_Landsat_L89 = ee.Image("users/omarorellanahn/INRIA/PCA_Landsat_2024_Aquitania");//CP1, CP2, 21 y 22

// Climate
var precipMonthly  = ee.Image("users/omarorellanahn/INRIA/Precipitation_Monthly_30m_Aquitaine");//CP1, CP2, 23 y 24
var tempMonthly    = ee.Image("users/omarorellanahn/INRIA/Temperature_Monthly_30m_Aq_v21");//CP1, CP2, 25 y 26
var WindSpeed      = ee.Image("users/omarorellanahn/INRIA/GWA_WindSpeed_10m_Aq");//27
var PCA_Precip     = ee.Image("users/omarorellanahn/INRIA/PCA_Precipitation_12meses_Aq");//CP1, CP2, 23 y 24
var PCA_Temp       = ee.Image("users/omarorellanahn/INRIA/PCA_Temperature_12meses_Aq");//CP1, CP2, 25 y 26

// Sentinel-5P atmospheric covariates
var no2  = ee.Image("users/omarorellanahn/INRIA/NO2_mean_30m_Aquitaine");//28
var so2  = ee.Image("users/omarorellanahn/INRIA/SO2_mean_30m_Aquitaine");//29
var co   = ee.Image("users/omarorellanahn/INRIA/CO_mean_30m_Aquitaine");//30
var hcho = ee.Image("users/omarorellanahn/INRIA/HCHO_mean_30m_Aquitaine");//31
var ai   = ee.Image("users/omarorellanahn/INRIA/AER_AI_mean_30m_Aquitaine");//32

// Soil
var ph  = ee.Image("users/omarorellanahn/INRIA/SoilGrids_pH_0_5cm_Aquit");//33
var soc = ee.Image("users/omarorellanahn/INRIA/SoilGrids_SOC_0_5cm_pct_Aq");//34

// Human impact
var Human_Footprint_Index = ee.Image("users/omarorellanahn/INRIA/HFI_2020_log1p_Aq").rename("HFI");//35
var Human_Modification_Index = ee.Image("users/omarorellanahn/INRIA/gHM_2016_Aq").rename("HMI");//36
var Population_density = ee.Image("users/omarorellanahn/INRIA/WorldPop_pop_dens_log1p_2020_Aq").rename("HPD");//37
var Global_Human_Built_volume = ee.Image("users/omarorellanahn/INRIA/GHSL_built_vol_log1p_2020_Aq").rename("HBV");//38
var nighttime_lights = ee.Image("users/omarorellanahn/INRIA/VIIRS_ntl_mean_log1p_2024_Aq").rename("NTL");//39




// -------------------------
// Clip helper
// -------------------------
function clipROI(img){ return img.clip(roi); }

// ============================================================================
//  3) MAP VISUALIZATION
// ============================================================================
Map.addLayer(roiFC,     {color: 'yellow'}, 'ROI (Buffer 6km)', false);
Map.addLayer(samples50, {color: 'red'},    'Samples 50', true);

// Topography
Map.addLayer(dem,    {min:0,   max:400, palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'DEM (30m)', false);
Map.addLayer(slope,  {min:0,   max:15,  palette:['006837','ffffbf','a50026']}, 'Slope % (30m)', false);
Map.addLayer(aspect, {min:0,   max:360, palette:['red','yellow','green','cyan','blue','magenta','red']}, 'Aspect deg (30m)', false);

// Hydrology
var accLog = clipROI(flowAcc).add(1).log10();
Map.addLayer(accLog,    {min:0, max:6, palette:['f7fbff','c6dbef','6baed6','2171b5','08306b']}, 'Flow Acc log10', false);
Map.addLayer(flowLen,   {min:4370, max:210000, palette:['1f78b4','33a02c','e31a1c','ff7f00','6a3d9a','b15928','a6cee3','fb9a99']}, 'Flow Length', false);
Map.addLayer(flowDir360,{min:0, max:360, palette:['red','yellow','green','cyan','blue','magenta','red']}, 'Flow Direction 0-360', false);
Map.addLayer(distSea,   {min:0, max:100000, palette:['0000FF','00AAFF','00FFAA','FFFF00','FF8800','FF0000']}, 'Distance to Sea (m)', false);
Map.addLayer(distRivers,{min:0, max:5000,   palette:['07057e','e1e924','e91e06']}, 'Distance to Rivers (m)', false);
Map.addLayer(urbanDist, {min:0, max:15000,  palette:['00FF00','FFFF00','FF0000']}, 'Urban distance', false);

// Terrain
Map.addLayer(curvature,   {min:-0.05, max:0.05, palette:['2c7bb6','ffffbf','d7191c']}, 'Curvature', false);
Map.addLayer(planCurv,    {min:-0.05, max:0.05, palette:['2c7bb6','ffffbf','d7191c']}, 'Plan Curvature', false);
Map.addLayer(profileCurv, {min:-0.05, max:0.05, palette:['2c7bb6','ffffbf','d7191c']}, 'Profile Curvature', false);
Map.addLayer(geomorphons, {min:232, max:3220, palette:['440154','482878','3e4989','31688e','26828e','1f9e89','35b779','6ece58','b5de2b','fde725']}, 'Geomorphons', false);
Map.addLayer(landforms,   {min:1,   max:8,    palette:['2b83ba','abdda4','ffffbf','fdae61','d7191c','1a9641','a6d96a','f46d43','74add1','fef08b']}, 'LandForms', false);

// Land cover
Map.addLayer(dwTrees, {min:0, max:1, palette:['f0fbd8','167e39']}, 'DW Trees prob', false);
Map.addLayer(dwBuilt, {min:0, max:1, palette:['f0fbd8','bd2408']}, 'DW Built prob', false);
Map.addLayer(dwCrops, {min:0, max:1, palette:['f0fbd8','867ffb']}, 'DW Crops prob', false);
Map.addLayer(l89NDFI, {min:-0.1, max:1, palette:['d63000','d6b95b','e5e445','e5e445','2c9437']}, 'L89 NDFI', false);

// Landsat PCA
Map.addLayer(PCA_Landsat_L89.select('LS_PC1'), {min:-0.16, max:0.31, palette:['2166ac','f7f7f7','d6604d']}, 'PCA Landsat PC1', false);

// Climate
Map.addLayer(precipMonthly.select('P_June'),  {min:40, max:130, palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'Precipitation June', false);
Map.addLayer(tempMonthly.select('T_Junio'),   {min:22, max:34,  palette:['313695','4575b4','74add1','abd9e9','e0f3f8','ffffbf','fee090','fdae61','f46d43','d73027','a50026']}, 'Temperature June', false);
Map.addLayer(WindSpeed,                        {min:2,  max:7,   palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','feb24c','f03b20','bd0026']}, 'Wind Speed 10m', false);
Map.addLayer(PCA_Temp.select('T_PC1'),         {min:-14, max:28, palette:['2166ac','f7f7f7','d6604d']}, 'Temperature PCA PC1', false);
Map.addLayer(PCA_Precip.select('P_PC1'),       {min:-300, max:60, palette:['ffffcc','41b6c4','0c2c84']}, 'Precipitation PCA PC1', false);

// Atmospheric covariates
Map.addLayer(no2,  {min:0.0000125, max:0.000037,  palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'S5P NO2', false);
Map.addLayer(so2,  {min:-0.000050, max:0.000227,  palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'S5P SO2', false);
Map.addLayer(co,   {min:0.024,     max:0.033,      palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'S5P CO', false);
Map.addLayer(hcho, {min:0.00005,   max:0.00013,    palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'S5P HCHO', false);
Map.addLayer(ai,   {min:-0.50,     max:-0.16,      palette:['081d58','225ea8','41b6c4','a1dab4','ffffcc','fdae61','f46d43','d73027']}, 'S5P Aerosol Index', false);

// Soil
Map.addLayer(ph,  {min:4.5, max:8.0, palette:['d7191c','fdae61','ffffbf','a6d96a','1a9641']}, 'pH 0-5cm', false);
Map.addLayer(soc, {min:0,   max:10,  palette:['ffffcc','c7e9b4','7fcdbb','41b6c4','2c7fb8','253494']}, 'SOC 0-5cm (%)', false);


// Human impact
Map.addLayer(Human_Footprint_Index, {min:0,   max:9,  palette:['e8f5e9', 'a5d6a7', 'ffffbf', 'fdae61', 'f46d43', 'd73027', '7f0000']}, 'HFI_Human_Footprint_Index', false);
Map.addLayer(Human_Modification_Index , {min:0,   max:1,  palette:['e8f5e9', 'a5d6a7', 'ffffbf', 'fdae61', 'f46d43', 'd73027', '7f0000']}, 'HMI_Human_Modification_Index ', false);
Map.addLayer(Population_density, {min:0,   max:8,  palette:['e8f5e9', 'a5d6a7', 'ffffbf', 'fdae61', 'f46d43', 'd73027', '7f0000']}, 'PD_Population_density', false);
Map.addLayer(Global_Human_Built_volume, {min:0,   max:10,  palette:['e8f5e9', 'a5d6a7', 'ffffbf', 'fdae61', 'f46d43', 'd73027', '7f0000']}, 'GHBV_Global_Human_Built_volume', false);
Map.addLayer(nighttime_lights, {min:0,   max:3,  palette:['000000', '0d0887','6a00a8', 'b12a90','e16462', 'fca636', 'f0f921', 'ffffff']}, 'NTL_nighttime_lights', false);



// ============================================================================
//  4) COVARIATE PREPARATION FOR EXTRACTION
// ============================================================================

// --- Cyclic sine/cosine transformations ---
// Aspect
var asp_rad = aspect.multiply(Math.PI / 180);
var asp_sin = asp_rad.sin().rename('asp_sin');
var asp_cos = asp_rad.cos().rename('asp_cos');

// Flow direction 360 degrees
var fd_rad  = flowDir360.multiply(Math.PI / 180);
var fd_sin  = fd_rad.sin().rename('flow_dir_sin');
var fd_cos  = fd_rad.cos().rename('flow_dir_cos');

// --- Flow Accumulation log10 ---
var flowAcc_log = flowAcc.add(1).log10().rename('flow_acc_log');

// --- Rename generic bands (b1) ---
var flowLen_r     = flowLen    .rename('flow_len');
var distSea_r     = distSea    .rename('dist_sea');
var urbanDist_r   = urbanDist  .rename('dist_urban');
var curvature_r   = curvature  .rename('curv');
var planCurv_r    = planCurv   .rename('curv_plan');
var profileCurv_r = profileCurv.rename('curv_prof');
var landforms_r   = landforms  .rename('landforms');
var geomorphons_r = geomorphons.rename('geomorphons');

// --- Rename PCA bands for clarity ---
var PCA_Precip_r  = PCA_Precip.rename(['P_PC1','P_PC2','P_PC3']);
var PCA_Temp_r    = PCA_Temp  .rename(['T_PC1','T_PC2','T_PC3']);

// ============================================================================
// 5. FINAL COVARIATE STACK — 36 VARIABLES
// ============================================================================
// The initial candidate stack contained 39 covariates.
//
// Three candidate variables were subsequently excluded during preprocessing:
//
//   - CO_mean:
//       Removed after collinearity screening.
//
//   - landforms:
//       Removed because it represents a categorical terrain classification,
//       whereas the final analytical covariate set was restricted to
//       continuous variables.
//
//   - geomorphons:
//       Removed because it represents a categorical terrain classification,
//       whereas the final analytical covariate set was restricted to
//       continuous variables.
//
// The corresponding source rasters are retained above for documentation of
// the original candidate-variable pool, but they are not included in the
// final 36-covariate analytical stack.
// ============================================================================
var covStack = dem                  // elevation_m         [1]
  .addBands(slope)                  // slope_pct           [2]
  .addBands(asp_sin)                // asp_sin             [3]
  .addBands(asp_cos)                // asp_cos             [4]
  .addBands(flowAcc_log)            // flow_acc_log        [5]
  .addBands(flowLen_r)              // flow_len            [6]
  .addBands(fd_sin)                 // flow_dir_sin        [7]
  .addBands(fd_cos)                 // flow_dir_cos        [8]
  .addBands(distSea_r)              // dist_sea            [9]
  .addBands(distRivers)             // dist_river_m        [10]
  .addBands(urbanDist_r)            // dist_urban          [11]
  .addBands(curvature_r)            // curv                [12]
  .addBands(planCurv_r)             // curv_plan           [13]
  .addBands(profileCurv_r)          // curv_prof           [14]
  .addBands(dwTrees)                // trees               [15]
  .addBands(dwBuilt)                // built               [16]
  .addBands(dwCrops)                // crops               [17]
  .addBands(l89NDFI)                // NDFI                [18]
  .addBands(PCA_Landsat_L89.select(['LS_PC1','LS_PC2']))// LS_PC1,PC2 [19-20]
  .addBands(PCA_Precip_r.select(['P_PC1','P_PC2']))     // P_PC1,PC2  [21-22]
  .addBands(PCA_Temp_r.select(['T_PC1','T_PC2']))       // T_PC1,PC2  [23-24]
  .addBands(WindSpeed)              // wind_speed_10m      [25]
  .addBands(no2)                    // NO2_mean            [26]
  .addBands(so2)                    // SO2_mean            [27]
  .addBands(hcho)                   // HCHO_mean           [28]
  .addBands(ai)                     // AER_AI_mean         [29]
  .addBands(ph)                     // pH_0_5cm            [30]
  .addBands(soc)                    // SOC_0_5cm_pct       [31]
  .addBands(Human_Footprint_Index)  // HFI                 [32]
  .addBands(Human_Modification_Index)// HMI                [33]
  .addBands(Population_density)     // HPD                 [34]
  .addBands(Global_Human_Built_volume)// HBV               [35]
  .addBands(nighttime_lights);      // NTL                 [36]


// Verify: this final analytical stack should contain 36 bands.
print('Stack bands:', covStack.bandNames());
print('Number of covariates:', covStack.bandNames().length());

// ============================================================================
//  6) SAMPLE INTERSECTION: sampleRegions
// ============================================================================
var samples_cov = covStack.sampleRegions({
  collection: samples50,
  scale:      30,
  geometries: true,
  tileScale:  4
});

print('Number of extracted features:', samples_cov.size());
print('First feature preview:', samples_cov.first());

// ============================================================================
//  7) EXPORT TO GOOGLE DRIVE
// ============================================================================
Export.table.toDrive({
  collection:     samples_cov,
  description:    'Samples_50_Covariables_INRIA_2024',
  folder:         'GEE_INRIA',
  fileNamePrefix: 'Samples_50_Covariables_INRIA_2024',
  fileFormat:     'CSV'
});
