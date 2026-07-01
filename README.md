# zambia-maize-yield-modeling
Working python file for RF ExtraTrees maize yield prediction in Zambia (using historical maize yield data, historical climate/soils data and future GCMs)
# import packages + organization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib
import warnings
import geopandas as gpd
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, ttest_1samp
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import partial_dependence

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── GLOBAL PLOT SETTINGS ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman'],
    'axes.titlesize'   : 15,
    'axes.labelsize'   : 14,
    'xtick.labelsize'  : 12,
    'ytick.labelsize'  : 12,
    'legend.fontsize'  : 12,
    'figure.titlesize' : 16,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/Users/colleenhenegan/Desktop/May_26_Modeling')
HIST_FILE  = BASE_DIR / '5.12.26_Historical_Data_All_Years.csv'
GCM_DIR    = BASE_DIR / '5.12.26_GCMs_for_Model'
ERA5_DIR   = BASE_DIR / 'Area_Weighted_ERA_districts'
SHP_FILE   = BASE_DIR / 'geospatial_data' / 'admin_boundary_shapefiles' / 'zmb_admbnda_adm2_dmmu_20201124.shp'
OUT_DIR    = BASE_DIR / 'model_outputs_v3'
MODEL_DIR  = BASE_DIR / 'saved_models'
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print('Paths:')
for label, p in [('Historical', HIST_FILE), ('GCM folder', GCM_DIR),
                  ('ERA5 folder', ERA5_DIR), ('Shapefile', SHP_FILE)]:
    print(f'  {label:<15}: {p.exists()}')
    # ── VARIABLE NAME LOOKUP ──────────────────────────────────────────────────────
VAR_LABELS = {
    'latitude'           : 'Latitude (°)',
    'longitude'          : 'Longitude (°)',
    'awc_maize_mm'       : 'Available Water Capacity (mm)',
    'wet_days'           : 'Wet Days (days)',
    'heat_days'          : 'Heat Days >32°C (days)',
    'heat_days_35'       : 'Heat Days >35°C (days)',
    'cold_days'          : 'Cold Days (days)',
    'avg_dry_spell'      : 'Average Dry Spell Length (days)',
    'max_dry_spell'      : 'Maximum Dry Spell Length (days)',
    'max_5day_rain'      : 'Maximum 5-Day Rainfall (mm)',
    'pet_season'         : 'Seasonal Potential Evapotranspiration (mm)',
    'aridity_index'      : 'Aridity Index',
    'gsl'                : 'Growing Season Length (days)',
    'precip_10'          : 'Precipitation — October (mm)',
    'precip_11'          : 'Precipitation — November (mm)',
    'precip_12'          : 'Precipitation — December (mm)',
    'precip_1'           : 'Precipitation — January (mm)',
    'precip_2'           : 'Precipitation — February (mm)',
    'precip_3'           : 'Precipitation — March (mm)',
    'precip_4'           : 'Precipitation — April (mm)',
    'precip_5'           : 'Precipitation — May (mm)',
    'tmax_10'            : 'Maximum Temperature — October (°C)',
    'tmax_11'            : 'Maximum Temperature — November (°C)',
    'tmax_12'            : 'Maximum Temperature — December (°C)',
    'tmax_1'             : 'Maximum Temperature — January (°C)',
    'tmax_2'             : 'Maximum Temperature — February (°C)',
    'tmax_3'             : 'Maximum Temperature — March (°C)',
    'tmax_4'             : 'Maximum Temperature — April (°C)',
    'tmax_5'             : 'Maximum Temperature — May (°C)',
    'tmin_10'            : 'Minimum Temperature — October (°C)',
    'tmin_11'            : 'Minimum Temperature — November (°C)',
    'tmin_12'            : 'Minimum Temperature — December (°C)',
    'tmin_1'             : 'Minimum Temperature — January (°C)',
    'tmin_2'             : 'Minimum Temperature — February (°C)',
    'tmin_3'             : 'Minimum Temperature — March (°C)',
    'tmin_4'             : 'Minimum Temperature — April (°C)',
    'tmin_5'             : 'Minimum Temperature — May (°C)',
    'soil_ph_0_30cm'     : 'Soil pH (0–30 cm)',
    'soil_soc_0_30cm_gkg': 'Soil Organic Carbon (g/kg, 0–30 cm)',
    'yield'              : 'Maize Yield (MT/ha)',
    'predicted_yield'    : 'Predicted Maize Yield (MT/ha)',
    'yield_change_pct'   : 'Yield Change from Baseline (%)',
    'yield_change_abs'   : 'Yield Change from Baseline (MT/ha)',
}

def vlab(col):
    return VAR_LABELS.get(col, col)

# ── COLOR PALETTES ────────────────────────────────────────────────────────────
SSP_COLORS   = {'SSP2-4.5': '#2c7bb6', 'SSP5-8.5': '#d7191c'}
AER_COLORS = {
    'I'  : '#E69F00',   # orange
    'IIA': '#009E73',   # green  
    'IIB': '#CC79A7',   # pink/purple
    'III': '#D55E00',   # vermillion
}
GCM_COLORS   = {
    'CNRM_CM6'  : '#e41a1c',
    'GFDL_ESM4' : '#377eb8',
    'MIROC_ES2L': '#4daf4a',
    'MPI_ESM'   : '#984ea3',
    'NorESM'    : '#ff7f00',
    'UKESM1'    : '#a65628',
}
GCM_SHORT = {
    'CNRM_CM6'  : 'CNRM',
    'GFDL_ESM4' : 'GFDL',
    'MIROC_ES2L': 'MIROC',
    'MPI_ESM'   : 'MPI',
    'NorESM'    : 'NorESM',
    'UKESM1'    : 'UKESM1',
}

# ── TIME PERIODS ──────────────────────────────────────────────────────────────
BASELINE_START = 1998
BASELINE_END   = 2023
PERIODS = {
    'Baseline (1998–2023)'   : (1998, 2023),
    'Near-term (2024–2049)'  : (2024, 2049),
    'Mid-century (2050–2074)': (2050, 2074),
    'End-century (2075–2100)': (2075, 2100),
}
FUTURE_PERIODS = {k: v for k, v in PERIODS.items() if v[0] >= 2024}
PERIOD_ORDER   = list(PERIODS.keys())
FUTURE_ORDER   = list(FUTURE_PERIODS.keys())
PERIOD_SHORT   = {
    FUTURE_ORDER[0]: '2024–2049',
    FUTURE_ORDER[1]: '2050–2074',
    FUTURE_ORDER[2]: '2075–2100',
}

# ── OTHER CONSTANTS ───────────────────────────────────────────────────────────
SSP_LABELS = {'245': 'SSP2-4.5', '585': 'SSP5-8.5'}
TARGET     = 'yield'
AERS       = ['I', 'IIA', 'IIB', 'III']

# Top 9 climatic variables (pre-selected, no soil/lat/lon)
TOP9_CLIM = [
    'tmax_4', 'tmax_3', 'tmax_2', 'precip_3',
    'tmin_1', 'precip_2', 'aridity_index',
    'wet_days', 'avg_dry_spell'
]

# Colormap for each top9 variable
VAR_CMAPS = {
    'tmax_4'      : 'YlOrRd',
    'tmax_3'      : 'YlOrRd',
    'tmax_2'      : 'YlOrRd',
    'tmax_1'      : 'YlOrRd',
    'tmin_1'      : 'PuBuGn',
    'precip_2'    : 'YlGnBu',
    'heat_days'   : 'YlOrRd',
    'wet_days'    : 'YlGnBu',
    'max_dry_spell': 'YlOrBr',
}

# Map axes settings
MAP_XTICKS = [22, 24, 26, 28, 30, 32, 34]
MAP_YTICKS = [-18, -16, -14, -12, -10, -8]

print('Configuration complete.')

# Historical data
hist = pd.read_csv(HIST_FILE)
print(f'Historical: {hist.shape} | Districts: {hist["district"].nunique()} | '
      f'Years: {hist["growing_season"].min()}–{hist["growing_season"].max()}')

# Shapefile
gdf = gpd.read_file(SHP_FILE)
shp_d  = set(gdf['ADM2_EN'].unique())
hist_d = set(hist['district'].unique())
print(f'Shapefile: {len(gdf)} features | District match: {len(shp_d & hist_d)}/115')

# GCM files
gcm_files = sorted(GCM_DIR.glob('*.csv'))
print(f'\nGCM files found: {len(gcm_files)}')
gcm_raw = []
for p in gcm_files:
    parts     = p.stem.split('_')
    ssp_code  = next((x for x in parts if x in SSP_LABELS), None)
    ssp_label = SSP_LABELS.get(ssp_code, ssp_code)
    gcm_name  = '_'.join(parts[:parts.index(ssp_code)])
    df        = pd.read_csv(p)
    df['gcm'] = gcm_name
    df['ssp'] = ssp_label
    gcm_raw.append(df)
    print(f'  {p.name:<45} GCM={gcm_name}  SSP={ssp_label}')

gcm_all = pd.concat(gcm_raw, ignore_index=True)
print(f'\nTotal GCM rows: {len(gcm_all):,} | Years: {gcm_all["growing_season"].min()}–{gcm_all["growing_season"].max()}')
X      = hist[FEATURES].values
y      = hist[TARGET].values
groups = hist['district'].values
gkf    = GroupKFold(n_splits=5)

param_grid = {
    'n_estimators'    : [500],
    'max_features'    : list(range(18, 29)),
    'min_samples_leaf': [3],
    'max_depth'       : [None, 20, 30],
}

search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, n_iter=30, scoring='r2', cv=gkf,
    random_state=42, verbose=1, n_jobs=-1
)
search_et = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=42, n_jobs=-1),
    param_grid, n_iter=30, scoring='r2', cv=gkf,
    random_state=42, verbose=1, n_jobs=-1
)
search_rf.fit(X, y, groups=groups)
search_et.fit(X, y, groups=groups)

print(f'\nRF  CV R²: {search_rf.best_score_:.3f} | params: {search_rf.best_params_}')
print(f'ET  CV R²: {search_et.best_score_:.3f} | params: {search_et.best_params_}')

if search_rf.best_score_ >= search_et.best_score_:
    search_nat     = search_rf
    model_type_nat = 'RandomForest'
else:
    search_nat     = search_et
    model_type_nat = 'ExtraTrees'

print(f'\nNational model: {model_type_nat} | CV R²: {search_nat.best_score_:.3f}')

# Train final national model on all historical data
rf_nat = search_nat.best_estimator_.__class__(
    **search_nat.best_params_, random_state=42, n_jobs=-1
)
rf_nat.fit(X, y)

y_ins_nat = rf_nat.predict(X)
nat_cv_r2   = search_nat.best_score_
nat_ins_r2  = r2_score(y, y_ins_nat)
nat_ins_rmse = np.sqrt(mean_squared_error(y, y_ins_nat))
nat_ins_mae  = mean_absolute_error(y, y_ins_nat)

print(f'National model trained on all {len(X)} observations.')
print(f'  CV R²         : {nat_cv_r2:.3f}  ← report this')
print(f'  In-sample R²  : {nat_ins_r2:.3f}  ← do not report')
print(f'  In-sample RMSE: {nat_ins_rmse:.3f} MT/ha')
print(f'  In-sample MAE : {nat_ins_mae:.3f} MT/ha')

rf_aer      = {}
cv_aer      = {}
search_aer  = {}
model_types = {}

for aer in AERS:
    sub = hist[hist['agro_ecol'] == aer].copy()
    Xa  = sub[FEATURES].values
    ya  = sub[TARGET].values
    ga  = sub['district'].values

    # 5-fold for AER models
    n_splits = min(5, len(np.unique(ga)))
    gkf_aer  = GroupKFold(n_splits=n_splits)

    s_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, n_iter=20, scoring='r2', cv=gkf_aer,
        random_state=42, verbose=0, n_jobs=-1
    )
    s_et = RandomizedSearchCV(
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        param_grid, n_iter=20, scoring='r2', cv=gkf_aer,
        random_state=42, verbose=0, n_jobs=-1
    )
    s_rf.fit(Xa, ya, groups=ga)
    s_et.fit(Xa, ya, groups=ga)

    best_s = s_rf if s_rf.best_score_ >= s_et.best_score_ else s_et
    mtype  = 'RF' if s_rf.best_score_ >= s_et.best_score_ else 'ET'

    final = best_s.best_estimator_.__class__(
        **best_s.best_params_, random_state=42, n_jobs=-1
    )
    final.fit(Xa, ya)

    rf_aer[aer]      = final
    cv_aer[aer]      = best_s.best_score_
    search_aer[aer]  = best_s
    model_types[aer] = mtype

    y_aer_ins  = final.predict(Xa)
    rmse_aer   = np.sqrt(mean_squared_error(ya, y_aer_ins))
    mae_aer    = mean_absolute_error(ya, y_aer_ins)

    print(f'AER {aer:4s} | n={len(sub):4d} | districts={len(np.unique(ga)):3d} | '
          f'{mtype} | CV R²={best_s.best_score_:.3f} | '
          f'RMSE={rmse_aer:.3f} | MAE={mae_aer:.3f}')
