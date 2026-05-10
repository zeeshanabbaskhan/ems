# ML Models — Complete Reference

## Overview

The EMS system uses a **three-model ML pipeline** trained on the **MobiAct** dataset for elderly fall monitoring:

| Model | Task | Input | Output |
|---|---|---|---|
| Fall Binary | Fall vs. non-fall | 144-D feature vector | Fall probability (0–1) |
| ADL Multiclass | Activity recognition | 144-D feature vector | Activity label (5 classes) |
| Fall-Type | Fall classification | 263-D → 150-D selected | Fall type code (4 classes) |

All models are trained with **subject-wise splits** (leakage-safe), serialised with **joblib**, and loaded at server startup via `inference_manifest.json`.

---

## Dataset: MobiAct

- IMU sensor data from a smartphone worn on the belt/pocket
- Sensors: 3-axis accelerometer (m/s²), 3-axis gyroscope (rad/s), orientation (azimuth, pitch, roll in degrees)
- Sample rate: **50 Hz**
- **67 subjects** split 45 train / 22 test (subject-wise, leakage-safe)
- Activity windows extracted with sliding window approach

### ADL Classes (Activities of Daily Living)
| Code | Full Name |
|---|---|
| STD | Standing |
| WAL | Walking |
| JOG | Jogging |
| SIT | Sitting |
| LYI | Lying |
| JUM | Jumping |
| STU | Stairs Up |
| STN | Stairs Down |
| SCH | Sit to Stand |
| CHU | Stand to Sit |
| CSI | Car Step In |
| CSO | Car Step Out |

### Fall Type Classes
| Code | Full Name |
|---|---|
| FOL | Forward lying fall |
| FKL | Front knees lying fall |
| BSC | Backward sitting-chair fall |
| SDL | Sideward lying fall |

---

## Feature Engineering

### v2 — 144-D Orientation-Invariant Features (Production)

**Source:** [`flask_backend/app/motion_enhanced_features.py`](flask_backend/app/motion_enhanced_features.py)  
**Training parity:** [`scripts/baseline_fall/enhanced_features.py`](scripts/baseline_fall/enhanced_features.py)

Window size: **128 samples @ 50 Hz = 2.56 s**

#### Gravity Separation

A Hamming-windowed FIR low-pass filter (cutoff = **0.3 Hz**) separates gravity from dynamic acceleration without scipy dependency:

```python
def _lowpass(signal, cutoff_hz=0.3, fs=50):
    n = min(int(fs / cutoff_hz), sig_len // 2)
    n = n if n % 2 == 1 else n + 1
    kernel = np.hamming(n)
    kernel /= kernel.sum()
    # Per-axis convolution with trim
```

`linear_acc = acc - gravity` (gravity-removed accelerometer signal)

#### Derived Signals (6 total)
1. `acc_mag` — accelerometer magnitude = √(ax²+ay²+az²)
2. `lin_mag` — linear acceleration magnitude (gravity removed)
3. `gyro_mag` — gyroscope magnitude
4. `acc_jerk` — derivative of acc_mag (prepend first sample)
5. `lin_jerk` — derivative of lin_mag
6. `gyro_jerk` — derivative of gyro_mag

#### Feature Layout (total = 144)

| Block | Count | Description |
|---|---|---|
| Magnitude stats | 84 | 6 signals × 14 statistics each |
| Magnitude frequency | 18 | 3 signals × 6 frequency features each |
| Axis stats (linear_acc) | 18 | 3 axes × 6 statistics |
| Axis stats (gyro) | 18 | 3 axes × 6 statistics |
| Cross-axis correlations | 3 | Pearson r for linear_acc pairs (01, 02, 12) |
| Orientation means | 3 | mean(azimuth), mean(pitch), mean(roll) |
| **Total** | **144** | |

#### 14 Magnitude Statistics (`_mag_stats`)
1. mean
2. std
3. median
4. min
5. max
6. peak-to-peak (ptp)
7. 5th percentile
8. 95th percentile
9. RMS = √(mean(x²))
10. mean abs diff = mean(|diff(x)|)
11. arc-length = sum(|diff(x)|)
12. variance
13. power = sum(x²) / len
14. peak = max(|x|)

#### 6 Magnitude Frequency Features (`_mag_freq`)
1. dominant frequency (Hz) — argmax of FFT magnitudes excluding DC
2. spectral energy = sum(|FFT|²) / n
3. spectral entropy = −Σ p·log(p) over normalized PSD
4. slow band power (0–1 Hz)
5. mid band power (1–3 Hz)
6. high band power (≥3 Hz)

#### 6 Axis Statistics (`_axis_stats`, per axis × 3 axes = 18)
1. mean
2. std
3. max
4. min
5. RMS
6. 95th percentile

#### Cross-axis Correlations
- Pearson r between linear_acc pairs: (x,y), (x,z), (y,z)
- NaN (zero-std axes) → 0.0

#### Orientation Means
- mean(azimuth), mean(pitch), mean(roll) — critical for LYI vs. STD separation

---

### 263-D Fall-Type Features (Optional Branch)

**Source:** [`scripts/baseline_falltype/feature_extractors.py`](scripts/baseline_falltype/feature_extractors.py)  
**Class:** `CompleteFallFeatureExtractor`

Window size: **300 samples @ 50 Hz = 6 s** (impact-centered)  
Preprocessing: Butterworth low-pass filter (cutoff=10 Hz, order=4) per axis

#### Feature Breakdown (263 total, before MI selection)

| Block | Count | Per Source | Description |
|---|---|---|---|
| ACC time features | 90 | 30 per axis × 3 axes | Time-domain stats on filtered acc |
| ACC freq features | 36 | 12 per axis × 3 axes | PSD band power, centroid, spread, entropy |
| GYRO time features | 60 | 20 per axis × 3 axes | Time-domain stats (truncated) |
| GYRO freq features | 24 | 8 per axis × 3 axes | Frequency features (truncated) |
| Orientation features | 15 | 5 per axis × 3 axes | Stats + diff-stats + final orientation |
| Fall impact features | 50 | — | Impact-specific temporal + gyro peaks |
| Cross-sensor features | 15 | — | Acc/gyro correlations, peak offsets, orientation change |
| SMA | 1 | — | Signal Magnitude Area of filtered acc |
| **Total** | **263** | | Before MI selection |

##### 30 Time-Domain Statistics (per signal)
- mean, std, median, min, max, ptp, RMS, mean-abs
- Percentiles: 10, 25, 75, 90, 95, 99
- skewness, kurtosis (scipy)
- mean-abs-diff, max-abs-diff, std-diff, arc-length, diff-power, 95th-diff-percentile
- mean-crossing rate, zero-crossing rate
- total energy, power, peak absolute value
- histogram entropy (20 bins)

##### 12 Frequency Features (per signal, Welch PSD)
- Band power: 0–1, 1–3, 3–6, 6–10, 10–15, 15–25 Hz (6 features)
- Total PSD power
- Spectral centroid
- Spectral spread
- Spectral entropy (log2)
- Dominant frequency (Hz)
- Peak dominance ratio = peak_psd / second_peak_psd

##### Fall Impact Features (50 per window)
- Impact position (normalized index)
- Pre-impact: mean, std, max, min, 95th pct (1 s before)
- Impact region: max, std, mean, relative peak pos (±0.1 s)
- Post-impact: mean, std, max, min, duration ratio (0.2–2 s after)
- Peak value, rise, ratio vs. pre-impact mean
- Peak count, mean peak height, std peak heights, first peak position
- Gyro magnitude: max, mean, std, peak position, per-axis max/mean, integrated angular change

##### Cross-Sensor Features (15 per window)
- Acc axis correlations: (x,y), (x,z), (y,z) — 3 features
- Acc-gyro magnitude correlation — 1 feature
- Gyro-acc peak timing offset — 1 feature
- Orientation change around impact (Δazimuth, Δpitch, Δroll) — 3 features

##### Orientation Features (15 per window)
- Per axis (azimuth, pitch, roll): mean, std, median, min, max, ptp, mean-abs-diff, max-abs-diff, arc-length — 9 features each → 27 total (truncated to 15)
- Final orientation values: last sample of azimuth, pitch, roll — 3 features

#### Mutual Information Feature Selection
**Source:** [`scripts/baseline_falltype/feature_selection_mi.py`](scripts/baseline_falltype/feature_selection_mi.py)

- Computes MI between each of 263 raw features and fall-type labels
- Selects top **150 features** by MI score
- Feature indices stored in `selected_features.pkl`
- Applied at inference: `xs[:, fall_type_indices]` → (1, 150) before model prediction

---

## Model Training Pipelines

### 1. Fall Binary Detection

**Entry:** [`scripts/baseline_fall/train_fall_detection_mobiact.py`](scripts/baseline_fall/train_fall_detection_mobiact.py)  
**Core:** [`scripts/baseline_fall/fall_detection_core.py`](scripts/baseline_fall/fall_detection_core.py)  
**Config:** [`scripts/baseline_fall/config.py`](scripts/baseline_fall/config.py)

#### Configuration
```python
RANDOM_STATE = 42
N_CV_FOLDS = 5
SUBJECT_TRAIN_FRACTION = 0.80  # 45 of 67 subjects
FEATURE_DIM = 116  # legacy; v2 uses 144-D
```

#### Pipeline Steps
1. Load MobiAct windows (acc, gyro, ori) via `mobiact_dataset.py`
2. Extract 116/144-D features
3. Subject-wise train/test split (80% subjects → train)
4. Apply `RobustScaler` (fitted on train only)
5. Balance with **SMOTETomek** (ratio=0.5) or **SMOTE** fallback
6. Train 5 classifiers; pick best by **F1 score** (weighted)
7. Save: `best_fall_model.pkl`, `scaler_fall.pkl`

#### Class Imbalance Handling (`sampling.py`)
- `balance_fall_train()`: SMOTETomek with sampling_strategy=0.5; falls back to SMOTE if SMOTETomek fails
- `balance_adl_train()`: ADASYN or SMOTE for multiclass (min 100 samples/class)

#### Subject-Level Split (`subject_split.py`)
- Groups windows by subject ID; shuffles subjects; picks first 80% for train
- Falls back to stratified index split if subject metadata is missing
- Ensures no subject appears in both train and test (leakage prevention)

#### Estimators (`fall_detection_models.py`)

**Legacy XGBoost Baseline** (original Colab style):
```python
XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss"
)
```

**Multi-Model Comparison** (production training):
| Model | Key Params |
|---|---|
| LightGBM | 300 trees, max_depth=8, lr=0.05, num_leaves=31, L1+L2=0.1 |
| XGBoost | 300 trees, max_depth=8, lr=0.05, L1+L2=0.1 |
| Random Forest | 300 trees, max_depth=15, min_samples_split=5 |
| Gradient Boosting | 200 trees, max_depth=6, lr=0.05, subsample=0.8 |
| Voting Ensemble | Soft vote: LGBM + XGB + RF |

**Selection criterion:** Best weighted F1 on held-out subjects

#### Results (v2 Training)
- **Accuracy: 99.3%**, F1: 0.993, ROC-AUC: 0.997
- 45 train subjects, 22 test subjects
- 5-fold CV accuracy: 0.994

---

### 2. ADL Multiclass Recognition

**Entry:** [`scripts/baseline_adl/train_mobiact_adl.py`](scripts/baseline_adl/train_mobiact_adl.py)  
**Config:** [`scripts/baseline_adl/config.py`](scripts/baseline_adl/config.py)

#### Configuration
```python
RANDOM_STATE = 42
N_CV_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 100  # rare classes dropped
FEATURE_DIM = 116  # legacy; v2 uses 144-D
USE_SMOTE = False  # ADASYN preferred
```

#### Pipeline Steps
1. Filter fall windows out; keep ADL-only rows
2. Drop classes with < 100 samples
3. Re-encode labels 0…C-1 with `LabelEncoder`
4. Stratified train/test split (preserves class proportions)
5. Apply `RobustScaler`
6. Train 5 classifiers; pick best by **weighted F1**
7. Save: `adl_xgb_model.pkl`, `adl_scaler.pkl`, `adl_label_encoder.pkl`

#### Estimators (`baseline_adl/models.py`)
| Model | Key Params |
|---|---|
| XGBoost | 100 trees, max_depth=6, lr=0.1, colsample=0.8 |
| LightGBM | 100 trees, max_depth=6, lr=0.1, num_leaves=31 |
| Random Forest | 50 trees, max_depth=10 |
| Logistic Regression | C=1.0, max_iter=1000, multiclass=auto |
| Decision Tree | max_depth=10, min_samples_split=5 |

#### Results (v2 Training)
- **Accuracy: 95.9%**, F1: 0.959, CV: 0.994
- 5 ADL classes: JOG, LYI, SIT, STD, WAL

---

### 3. Fall-Type Classification

**Entry:** [`scripts/baseline_falltype/train_fall_type_mobiact.py`](scripts/baseline_falltype/train_fall_type_mobiact.py)  
**Config:** [`scripts/baseline_falltype/config.py`](scripts/baseline_falltype/config.py)

#### Configuration
```python
SAMPLING_RATE_HZ = 50
WINDOW_SAMPLES = 300        # 6-second windows
FALL_TYPE_RAW_DIM = 263     # raw feature vector
N_SELECTED_FEATURES = 150   # after MI selection
FALL_CODES = ["BSC", "FOL", "FKL", "SDL"]
```

#### Pipeline Steps
1. Load fall-only windows from MobiAct (FOL/FKL/BSC/SDL prefixed files)
2. Extract **263-D raw features** per window (`CompleteFallFeatureExtractor`)
3. Apply `StandardScaler` (fitted on all fall-type train data)
4. MI feature selection → top 150 feature indices saved
5. Train 5 classifiers (300-tree ensembles); pick best by **accuracy**
6. Save: `best_fall_classifier.pkl`, `scaler.pkl`, `selected_features.pkl`, `label_encoder.pkl`

#### Estimators (`fall_type_models.py`)
| Model | Key Params |
|---|---|
| LightGBM | 300 trees, max_depth=8, lr=0.05, subsample=0.8 |
| XGBoost | 300 trees, max_depth=8, lr=0.05, subsample=0.8 |
| Random Forest | 300 trees, max_depth=15, min_samples_split=5 |
| Gradient Boosting | 200 trees, max_depth=6, lr=0.05 |
| Voting Ensemble | Soft vote: LGBM + XGB + RF |

#### Impact-Centered Window Extraction (`fall_window_dataset.py`)
- Searches for CSV files with FOL/FKL/BSC/SDL prefixes
- Aligns accelerometer, gyroscope, and orientation data
- Extracts 300-sample window centered on peak acceleration (impact)

---

## Inference Pipeline

**Source:** [`scripts/inference/motion_pipeline.py`](scripts/inference/motion_pipeline.py)  
**Re-exported by:** [`flask_backend/app/services/motion_xgb_service.py`](flask_backend/app/services/motion_xgb_service.py)

### InferenceArtifacts Dataclass

```python
@dataclass(frozen=True)
class InferenceArtifacts:
    manifest: dict           # parsed inference_manifest.json
    fall_model: Any | None   # binary fall classifier (joblib)
    fall_scaler: Any | None  # RobustScaler for fall model
    adl_model: Any           # ADL multiclass classifier
    adl_scaler: Any          # RobustScaler for ADL model
    adl_encoder: Any         # LabelEncoder (index → code)
    fall_type_model: Any | None      # optional fall-type classifier
    fall_type_scaler: Any | None     # StandardScaler for fall-type
    fall_type_indices: np.ndarray    # 150 MI-selected feature indices
    fall_type_encoder: Any | None    # LabelEncoder (index → FOL/FKL/BSC/SDL)
    enhanced_dim: int        # 144
    fall_type_dim: int       # 263
    fall_threshold: float    # 0.55
    fall_binary_enabled: bool
    fall_binary_issue: str | None
    fall_type_enabled: bool
```

### load_artifacts()

1. Reads `inference_manifest.json`
2. Loads ADL model + scaler + encoder (required; startup fails if missing)
3. Loads fall-type model + scaler + MI indices + encoder (optional; null in manifest disables)
4. Loads fall binary model + scaler (optional; dim mismatch disables with logged reason)
5. Validates scaler `n_features_in_` against manifest dimensions
6. Returns frozen `InferenceArtifacts`

### run_inference() — Two-Stage Decision Tree

```
Input: 144-D enhanced_features vector

Stage 1: Fall Detection
  If fall_binary_enabled:
    x → fall_scaler.transform → fall_model.predict_proba
    p_fall = proba[:, 1]   (positive class probability)
    is_fall = (p_fall >= fall_threshold)   [threshold = 0.55]
  Else:
    Skip to ADL branch (fall_binary_issue logged)

Stage 2a: Not a fall → ADL Branch
  x → adl_scaler.transform → adl_model.predict
  label = adl_encoder.inverse_transform(prediction)
  Returns: branch="adl", activity_label, activity_class_index

Stage 2b: Fall detected → Fall-Type Branch
  If fall_type_enabled AND (fall_type_features OR acc_window provided):
    Build 263-D vector from acc_window/gyro_window/ori_window
    → fall_type_scaler.transform
    → select columns [fall_type_indices]  (150 of 263)
    → fall_type_model.predict
    label = fall_type_encoder.inverse_transform
    Returns: branch="fall", fall_type_code, fall_type_label
  Else:
    Returns: branch="fall", fall_type_code=None, skipped_reason
```

### Output Schema
```python
{
    "is_fall": bool,
    "fall_probability": float,         # 0.0–1.0
    "fall_threshold": float,           # 0.55
    "schema_version": str,             # "2.0"
    "branch": "adl" | "fall",
    # ADL branch only:
    "activity_class_index": int | None,
    "activity_label": str | None,
    # Fall branch only:
    "fall_type_code": str | None,      # "FOL","FKL","BSC","SDL"
    "fall_type_label": str | None,
    "fall_type_class_index": int | None,
    "fall_type_skipped_reason": str | None,
}
```

### Feature Safety
- All features sanitized: `np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`
- Dimension check: raises `ValueError` if `len(enhanced_features) != 144`
- sklearn/XGBoost/LightGBM feature-name warnings suppressed (runtime sends plain arrays)

---

## Inference Manifest

**File:** [`flask_backend/models/inference_manifest.json`](flask_backend/models/inference_manifest.json)

```json
{
  "schema_version": "2.0",
  "enhanced_feature_dim": 144,
  "fall_type_raw_dim": 263,
  "fall_type_selected_features": 150,
  "fall_probability_threshold": 0.55,
  "random_state_training": 42,
  "artifacts": {
    "fall_binary": {
      "model_path": "baseline_adl&fall/fall_xgb_model.pkl",
      "scaler_path": "baseline_adl&fall/fall_scaler.pkl"
    },
    "adl": {
      "model_path": "baseline_adl&fall/adl_xgb_model.pkl",
      "scaler_path": "baseline_adl&fall/adl_scaler.pkl",
      "label_encoder_path": "baseline_adl&fall/adl_label_encoder.pkl"
    },
    "fall_type": null
  }
}
```

- `fall_type: null` disables fall-type branch at runtime
- All paths are relative to `flask_backend/models/`
- `sync_inference_manifest.py` updates dims from loaded scaler shapes after retraining

---

## Model Artifacts Directory

```
flask_backend/models/
├── inference_manifest.json
├── baseline_adl&fall/
│   ├── fall_xgb_model.pkl      (XGBoost binary classifier, 144-D input)
│   ├── fall_scaler.pkl          (RobustScaler, 144 features)
│   ├── adl_xgb_model.pkl       (XGBoost multiclass, 144-D input)
│   ├── adl_scaler.pkl           (RobustScaler, 144 features)
│   ├── adl_label_encoder.pkl   (LabelEncoder: int → ADL code)
│   └── results_v2.json          (training metrics snapshot)
└── baseline_falltype/           (optional, populated when fall-type trained)
    ├── best_fall_classifier.pkl
    ├── scaler.pkl
    ├── selected_features.pkl    (150 MI feature indices as list[int])
    └── label_encoder.pkl
```

---

## Training Results (v2 Snapshot)

**File:** `flask_backend/models/baseline_adl&fall/results_v2.json`

| Metric | Fall Binary | ADL Multiclass |
|---|---|---|
| Accuracy | 99.3% | 95.9% |
| Weighted F1 | 0.993 | 0.959 |
| ROC-AUC | 0.997 | — |
| CV Accuracy | 0.994 | 0.994 |
| Train subjects | 45 | 45 |
| Test subjects | 22 | 22 |

---

## VoteBuffer — ADL Smoothing

**Source:** [`flask_backend/app/ml_bridge.py`](flask_backend/app/ml_bridge.py)

Per-patient sliding majority-vote buffer to stabilize rapid ADL prediction flips:

```python
VOTE_BUFFER_SIZE = 7  # 7 windows × ~0.5 s overlap ≈ 3.5 s smoothing

class VoteBuffer:
    def push(self, label: str, is_fall: bool) -> str:
        if is_fall:
            self._buf.clear()   # Falls bypass voting immediately
            return label
        self._buf.append(label)
        if len(self._buf) < self._size:
            return label        # Return raw until buffer fills
        return Counter(self._buf).most_common(1)[0][0]  # Majority vote

    @property
    def confidence(self) -> float:
        # Fraction of buffer = majority label
```

- Falls bypass the buffer and reset it (immediate alert, no smoothing delay)
- ADL labels stable after 7 consecutive windows (≈3.5 s at 50% overlap)
- One `VoteBuffer` instance per `patient_id` (in-memory dict, reset on server restart)

---

## Raw Sample → Feature Vector Flow

**Source:** [`flask_backend/app/ml_bridge.py`](flask_backend/app/ml_bridge.py) — `samples_to_feature_vector()`

```
Input: list[dict] with keys acc_x/y/z, gyro_x/y/z, azimuth/pitch/roll
         (azimuth/pitch/roll optional → 0.0 default)

1. Parse into acc (n,3), gyro (n,3), ori (n,3) arrays

2. For 144-D enhanced branch (fall detection + ADL):
   Linear interpolate → 128 rows each (training window size)
   → extract_enhanced_features() → (1, 144) feature matrix → feat_144

3. For 263-D fall-type branch:
   Linear interpolate → 300 rows each (6-second window)
   → acc_300, gyro_300, ori_300  (returned for optional fall-type use)

Output: (feat_144, acc_300, gyro_300, ori_300)
```

Linear resampling: `np.interp()` per axis, from `n` original rows to target length

---

## Training Entry Points

### Unified Runner
```bash
python scripts/run_training.py fall-detection   # train fall binary
python scripts/run_training.py adl              # train ADL model
python scripts/run_training.py fall-type        # train fall-type model
python scripts/run_training.py all              # train all three
python scripts/run_training.py sync-manifest    # update manifest dims
```

### Individual Scripts
```bash
# Fall detection (116-D or 144-D features, multi-model comparison)
python scripts/baseline_fall/train_fall_detection_mobiact.py

# ADL recognition (144-D features)
python scripts/baseline_adl/train_mobiact_adl.py

# Fall-type classification (263-D → 150-D MI selected)
python scripts/baseline_falltype/train_fall_type_mobiact.py

# Legacy combined baseline (fall + ADL, XGBoost only)
python scripts/baseline_fall/train_mobiact_baselines.py
```

### Demo / Verification
```bash
# Simulate IMU windows through inference pipeline
python scripts/simulate_inference_demo.py

# Verify all three models load + produce correct output
python scripts/verify_three_models.py

# Check MobiAct baseline inference end-to-end
python scripts/verify_mobiact_baseline_inference.py

# Compare baseline models
python scripts/baseline_models_comparison.py
```

---

## Detector State & Severity Mapping

**Source:** [`flask_backend/app/detector_state.py`](flask_backend/app/detector_state.py)

After `run_inference()` returns a fall probability, a second layer of heuristics calibrates severity:

### Stationary Motion Guard

Prevents false alarms when a phone is stationary in a pocket/table:

```python
STATIONARY_GYRO_PEAK_DPS = 95.0      # below → possibly stationary
STATIONARY_PEAK_ACC_G = 2.05         # below → possibly stationary
STATIONARY_STILLNESS_MIN = 0.62      # above → probably stationary

def _effective_fall_probability(p, sig):
    if p <= 0.35:
        return p, False  # Low prob: no guard needed
    looks_stationary = (
        sig["peak_gyro_dps"] < 95.0 and
        sig["peak_acc_g"] < 2.05 and
        sig["stillness"] >= 0.62
    )
    if looks_stationary:
        dampened = min(p, 0.14 + p * 0.22)  # Soft cap
        return dampened, True
    return p, False
```

### Severity Thresholds (default MEDIUM profile)
| Threshold | Severity |
|---|---|
| p_eff ≥ 0.80 (DETECTOR_CFG fall_score) | `fall_detected` |
| p_eff ≥ 0.58 | `high_risk` |
| p_eff ≥ 0.35 | `medium` |
| below 0.35 | `low` |

### Impact Evidence Guard
Even if severity = `fall_detected`, the alert is **downgraded to `high_risk`** unless:
- `peak_acc_g >= 1.70 g` (FALL_MIN_PEAK_ACC_G), OR
- `peak_gyro_dps >= 180.0` (FALL_MIN_PEAK_GYRO_DPS)

### Score Computation
```python
score = max(p_eff, sig["peak_acc_g"] / 5.0 * 0.3 + p_eff * 0.7)
score = min(1.0, score)
```

### Signal Metrics Computed from Raw Batch
- `peak_acc_g` — max accelerometer magnitude in G (÷9.80665)
- `peak_gyro_dps` — max gyroscope magnitude in degrees/s (×180/π)
- `peak_jerk` — max abs difference between consecutive acc magnitudes
- `stillness` — 1 − (std/mean of acc magnitudes); higher = more stationary

---

## Heuristic Fallback (No ML)

When ML artifacts are unavailable or inference throws:

### Fall Probability Heuristic
```python
# Max acc magnitude / 25.0, clamped to [0, 1]
p_fall = min(1.0, max(0.0, np.max(mags) / 25.0))
```

### Activity Label Heuristic
```python
# Tuned for phone IMU ~gravity magnitude (m/s²)
if peak_mag > 22.0 or std_mag > 3.5 or peak_gyro > 4.0:  → "running"
if std_mag < 0.30 and 8.5 <= mean_mag <= 10.8 and peak_gyro < 0.8:  → "standing"
if std_mag < 0.55 and peak_gyro < 1.2:  → "sitting"
if std_mag < 2.0:  → "walking"
else:  → "moving"
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.26, <3 | Array ops, FFT |
| pandas | ≥2.0, <4 | Data loading/prep |
| scipy | ≥1.11, <2 | Welch PSD, Butterworth filter, stats |
| scikit-learn | ≥1.6.1, <2 | Scaler, LabelEncoder, RF, GB, metrics |
| xgboost | ≥2.0, <4 | Primary classifier |
| lightgbm | ≥4.0, <5 | Comparison classifier |
| imbalanced-learn | ≥0.12, <1 | SMOTE, ADASYN, SMOTETomek |
| joblib | ≥1.3, <2 | Model serialization |
| torch | ≥2.2, <3 | Reserved (future deep models) |
| tqdm | ≥4.65, <5 | Batch feature extraction progress |
| openpyxl | ≥3.1, <4 | Excel report output |

Training requirements: `scripts/requirements-training.txt`  
Runtime requirements: `requirements.txt` + `flask_backend/requirements.txt`
