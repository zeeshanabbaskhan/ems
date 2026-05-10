# StepSafe AI — Elder Fall Monitoring System
## Complete Project Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset & Data Loading](#3-dataset--data-loading)
4. [Data Preprocessing & Windowing](#4-data-preprocessing--windowing)
5. [Train/Test Split Strategy](#5-traintest-split-strategy)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training & Evaluation](#7-model-training--evaluation)
8. [Backend Architecture](#8-backend-architecture)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Frontend–Backend Integration](#10-frontendbackend-integration)
11. [Real-Time Inference Pipeline](#11-real-time-inference-pipeline)
12. [Database Schema](#12-database-schema)
13. [Authentication & Security](#13-authentication--security)
14. [Deployment & Configuration](#14-deployment--configuration)
15. [Test Coverage](#15-test-coverage)

---

## 1. Project Overview

**StepSafe AI** is a real-time elder fall monitoring and activity recognition system designed for elderly care. It combines a Flutter mobile application (worn or carried by the patient/elder), a FastAPI backend server with XGBoost machine learning models, and a caregiver dashboard that receives real-time fall alerts.

### Goals

| Goal | Description |
|------|-------------|
| Fall Detection | Detect the 4 MobiAct fall types (FOL, FKL, BSC, SDL) with high recall in real time |
| Activity Recognition | Classify 5 daily activities (WAL, JOG, SIT, STD, LYI) to provide context to caregivers |
| Real-Time Alerting | Push WebSocket notifications to caregivers within seconds of a detected fall |
| Multi-Role Access | Admin, Caregiver, and Elder roles with independent auth flows |
| Orientation Invariance | Feature design robust to how the phone is held/placed |

### Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Mobile App | Flutter / Dart | SDK ^3.9.0 |
| Backend API | FastAPI + Uvicorn | ≥0.110 / ≥0.27 |
| ML Models | XGBoost | ≥2.0 |
| Feature Scaling | scikit-learn RobustScaler | ≥1.6.1 |
| Auth | JWT (HS256) + bcrypt | PyJWT ≥2.8 |
| Database | SQLite (via Python stdlib) | — |
| Serialization | Pydantic v2 | ≥2.5 |
| Sensor Access | sensors_plus, motion_core | ^7.0.0 / ^0.0.5 |
| Real-time Push | WebSocket (web_socket_channel) | ^3.0.1 |
| Maps | flutter_map + geolocator | ^8.3.0 / ^14.0.2 |

---

## 2. System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEPSAFE AI SYSTEM                             │
└─────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────────┐          ┌──────────────────────┐
     │   ELDER'S PHONE      │          │  CAREGIVER'S PHONE   │
     │   (Flutter App)      │          │   (Flutter App)      │
     │                      │          │                      │
     │  ┌────────────────┐  │          │  ┌────────────────┐  │
     │  │ SensorStreaming │  │          │  │   Dashboard    │  │
     │  │    Service     │  │          │  │  Alert Panel   │  │
     │  │ Acc+Gyro+Ori   │  │          │  │  Live Map      │  │
     │  │  @ 50 Hz       │  │          │  └────────┬───────┘  │
     │  └───────┬────────┘  │          │           │          │
     │          │ 128-sample │          │  ┌────────▼───────┐  │
     │          │ window     │          │  │  WebSocket     │  │
     │  ┌───────▼────────┐  │          │  │  Listener      │  │
     │  │ MonitoringCtrl │  │          │  │  + HTTP Poll   │  │
     │  │ (ChangeNotifier│  │          │  └────────┬───────┘  │
     │  └───────┬────────┘  │          │           │          │
     │          │ HTTP POST  │          └───────────┼──────────┘
     └──────────┼────────────┘                      │
                │                                   │
                │  POST /api/v1/ingest/live          │ WS /api/v1/ws/caregiver
                │                                   │
     ┌──────────▼───────────────────────────────────▼──────────────────────┐
     │                     FASTAPI BACKEND (Python)                        │
     │                                                                      │
     │  ┌────────────┐  ┌───────────────┐  ┌──────────┐  ┌─────────────┐  │
     │  │  Auth JWT  │  │  monitoring_  │  │ realtime │  │  database   │  │
     │  │  bcrypt    │  │   routes.py   │  │  hub.py  │  │   .py       │  │
     │  └────────────┘  └───────┬───────┘  └────┬─────┘  └─────────────┘  │
     │                          │                │               │          │
     │              ┌───────────▼──────┐         │         SQLite DB       │
     │              │  ML Inference    │         │         (7 tables)      │
     │              │  Pipeline        │         │                          │
     │              │                  │         │                          │
     │              │ ┌──────────────┐ │    WebSocket                      │
     │              │ │  ml_bridge   │ │    Broadcast                      │
     │              │ │  .py         │ │    on fall_detected                │
     │              │ │ 144-D feat   │ │                                   │
     │              │ └──────┬───────┘ │                                   │
     │              │        │         │                                   │
     │              │ ┌──────▼───────┐ │                                   │
     │              │ │motion_enhanced│ │                                   │
     │              │ │_features.py  │ │                                   │
     │              │ └──────┬───────┘ │                                   │
     │              │        │         │                                   │
     │              │ ┌──────▼───────┐ │                                   │
     │              │ │   XGBoost    │ │                                   │
     │              │ │  Fall Model  │ │                                   │
     │              │ │  ADL Model   │ │                                   │
     │              │ └──────────────┘ │                                   │
     │              └──────────────────┘                                   │
     │                                                                      │
     │              ┌──────────────────┐                                   │
     │              │  detector_state  │ → severity, score, alert decision  │
     │              └──────────────────┘                                   │
     └──────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  TRAINED MODELS    │
                    │  (flask_backend/   │
                    │   models/)         │
                    │                    │
                    │  fall_xgb_model    │
                    │  fall_scaler       │
                    │  adl_xgb_model     │
                    │  adl_scaler        │
                    │  adl_label_encoder │
                    └────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `SensorStreamingService` | `sensor_streaming_service.dart` | Reads accelerometer/gyroscope/orientation at 50 Hz, buffers 128-sample windows with 50% overlap |
| `MonitoringController` | `monitoring_controller.dart` | Central state (ChangeNotifier), orchestrates sessions, alerts, alarm, GPS, caregiver refresh |
| `BackendApiClient` | `api_client.dart` | All HTTP calls with 25 s timeout, 2-attempt retry |
| `MotionFeatureExtractor` | `motion_feature_extractor.dart` | Client-side 128-D feature extraction (legacy path; server recomputes with 144-D v2) |
| `monitoring_routes.py` | `monitoring_routes.py` | All REST endpoints + WebSocket handler (1 693 lines) |
| `motion_enhanced_features.py` | `motion_enhanced_features.py` | Server-side 144-D orientation-invariant feature extraction |
| `ml_bridge.py` | `ml_bridge.py` | Converts raw sensor samples → 144-D vector + 300-row windows; `VoteBuffer` for ADL smoothing |
| `motion_pipeline.py` | `scripts/inference/motion_pipeline.py` | Loads frozen models, runs fall → ADL two-stage inference |
| `detector_state.py` | `detector_state.py` | Converts ML fall probability into severity label and score |
| `realtime_hub.py` | `realtime_hub.py` | Async WebSocket broadcast to registered caregiver connections |
| `database.py` | `database.py` | SQLite schema init, migrations, thread-safe connection context manager |
| `auth_jwt.py` | `auth_jwt.py` | JWT creation/decode (HS256, 7-day TTL), bcrypt password hashing |

---

## 3. Dataset & Data Loading

### MobiAct Dataset v2.0

| Property | Value |
|----------|-------|
| Name | MobiAct Dataset v2.0 |
| Subjects | 67 healthy adults (various ages) |
| Sensor | Smartphone IMU (accelerometer, gyroscope, orientation) |
| Sampling rate | 50 Hz |
| Format | Annotated CSV files (`*_annotated.csv`) |
| Data columns | `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`, `azimuth`, `pitch`, `roll`, `label` |
| Units | Accelerometer: m/s², Gyroscope: rad/s, Orientation: degrees |

### Activity Labels Used

| Label | Type | Description |
|-------|------|-------------|
| WAL | ADL | Walking |
| JOG | ADL | Jogging |
| SIT | ADL | Sitting |
| STD | ADL | Standing |
| LYI | ADL | Lying |
| FOL | Fall | Forward lying fall |
| FKL | Fall | Front knees lying fall |
| BSC | Fall | Backward sitting-chair fall |
| SDL | Fall | Sideward lying fall |

### Data Loading Flow

```
MobiAct Dataset Root
     │
     ▼
glob(**/*_annotated.csv)          ← recursive scan for all CSV files
     │
     ▼
parse_subject_id(filepath)        ← extract integer subject ID from filename
     │                               e.g. "sub_05_WAL_annotated.csv" → subject 5
     ▼
pd.read_csv(csv_path)             ← load each CSV
     │
     ├─ validate required columns (acc_x/y/z, gyro_x/y/z, azimuth, pitch, roll, label)
     │
     ├─ normalise labels: strip whitespace, uppercase
     │
     └─ filter rows: keep only rows where label ∈ ALL_LABELS (5 ADL + 4 Fall)
```

### Per-Segment Sliding Window

```
For each CSV file:
  Group consecutive same-label rows into contiguous segments
       │
       ▼ (itertools.groupby on label column)
  For each segment of length ≥ WINDOW_SIZE (128):
       │
       ▼
  Slide window: start = 0, step = STEP_SIZE (64), until start + 128 ≤ seg_len
       │
       ▼
  Record: { subject, activity, is_fall, acc[128×3], gyro[128×3], ori[128×3] }

Total windows collected across all subjects: tens of thousands
Rare classes (< 100 windows) are dropped
```

---

## 4. Data Preprocessing & Windowing

### Windowing Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `WINDOW_SIZE` | 128 samples | ≈ 2.56 s at 50 Hz |
| `STEP_SIZE` | 64 samples | 50% overlap — doubles effective training set |
| `SAMPLE_RATE` | 50 Hz | |
| `MIN_WINDOWS_PER_CLASS` | 100 | Rare-class guard |

### Preprocessing Steps

```
Raw CSV rows (per contiguous segment)
         │
         ▼
1. Segment isolation
   groupby(label) → ensures windows never straddle two activities
         │
         ▼
2. Sliding window (128 samples, step 64)
   each window → (128, 3) arrays for acc / gyro / ori
         │
         ▼
3. Rare class removal
   class count < 100 windows → dropped
         │
         ▼
4. Feature extraction (see §6)
   (128, 3) arrays → 144-D float32 vector
         │
         ▼
5. NaN / Inf sanitisation
   np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
         │
         ▼
6. RobustScaler (fit on train split only)
   separate scalers for fall pipeline and ADL pipeline
         │
         ▼
7. Label encoding
   LabelEncoder for ADL (5 classes: JOG→0, LYI→1, SIT→2, STD→3, WAL→4)
   Binary label for fall (0=ADL, 1=Fall)
```

### Gravity Separation (applied inside feature extraction)

```
acc_window (128, 3)
         │
         ▼
Hamming-windowed FIR low-pass filter
  cutoff = 0.3 Hz, kernel size = min(fs/cutoff, sig_len//2), must be odd
  applied per axis via np.convolve(mode='full'), centre-trimmed
         │
         ▼
gravity (128, 3)  ←  slow, quasi-static component
         │
         ▼
linear_acc = acc - gravity  ←  dynamic, orientation-invariant
```

---

## 5. Train/Test Split Strategy

### Subject-Aware Split (Prevents Data Leakage)

```
All 67 subjects
       │
       ▼
np.random.default_rng(RANDOM_SEED=42).permutation(subjects)
       │
       ├──── Test subjects  (33% = 22 subjects): IDs 2,6,8,18,19,22,25,27-30,33,34,41,43,45,51,52,58,59,65,67
       │
       └──── Train subjects (67% = 45 subjects): IDs 1,3,4,5,7,9-17,20,21,23,24,26,31,32,35-40,42,44,46-50,53-57,60-64,66
```

**Key design decision:** Windows from the same subject are entirely in either train or test — never split across both. This ensures the model is evaluated on unseen individuals, simulating real-world deployment where the model must generalise to new users.

### Class Distribution

| Task | Train | Test |
|------|-------|------|
| Fall Detection (total windows) | ~67% of all windows | ~33% of all windows |
| ADL (train ADL/test ADL) | ADL windows from 45 subjects | ADL windows from 22 subjects |

### Scaling

```
RobustScaler — fit on X_train only, then applied to both X_train and X_test.
Two separate scalers:
  - fall_scaler.pkl  → used for fall detection pipeline (all 9 classes)
  - adl_scaler.pkl   → used for ADL classification pipeline (5 ADL classes only)
```

---

## 6. Feature Engineering

### Feature Design Philosophy (v2 — Orientation-Invariant)

The v2 feature design treats phone orientation as a nuisance variable. Instead of relying on raw axis values (which change when the phone is rotated), the primary signal comes from **magnitude signals** (rotation-invariant by definition) and **gravity-subtracted linear acceleration**.

### Feature Vector Layout — 144 Total

```
┌─────────────────────────────────────────────────────────────────────┐
│ BLOCK 1: Magnitude statistics  (6 signals × 14 stats = 84 features) │
├─────────────────────────────────────────────────────────────────────┤
│  Signal 1: acc_mag  = √(ax²+ay²+az²)                               │
│  Signal 2: lin_mag  = √(linear_acc_x²+y²+z²)  ← gravity-removed   │
│  Signal 3: gyro_mag = √(gx²+gy²+gz²)                              │
│  Signal 4: acc_jerk  = diff(acc_mag)                               │
│  Signal 5: lin_jerk  = diff(lin_mag)                               │
│  Signal 6: gyro_jerk = diff(gyro_mag)                              │
│                                                                     │
│  14 stats per signal:                                               │
│   mean, std, median, min, max, ptp, p5, p95, RMS,                 │
│   mean_abs_diff, arc_length, variance, power, peak                 │
├─────────────────────────────────────────────────────────────────────┤
│ BLOCK 2: Magnitude frequency features  (3 × 6 = 18 features)       │
├─────────────────────────────────────────────────────────────────────┤
│  Signals: acc_mag, lin_mag, gyro_mag                               │
│                                                                     │
│  6 features per signal:                                             │
│   dominant_freq, spectral_energy, spectral_entropy,                │
│   band_slow (0–1 Hz), band_mid (1–3 Hz), band_high (3+ Hz)        │
├─────────────────────────────────────────────────────────────────────┤
│ BLOCK 3: Axis-specific statistics  (2 sensors × 18 = 36 features)  │
├─────────────────────────────────────────────────────────────────────┤
│  Sensors: linear_acc (gravity-removed), gyro                        │
│                                                                     │
│  6 stats × 3 axes = 18 per sensor:                                 │
│   mean, std, max, min, RMS, p95                                    │
├─────────────────────────────────────────────────────────────────────┤
│ BLOCK 4: Cross-axis correlations of linear_acc  (3 features)        │
│   Pearson r: (x,y), (x,z), (y,z)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ BLOCK 5: Orientation means  (3 features)                            │
│   mean(azimuth), mean(pitch), mean(roll)                           │
│   (discriminates LYI vs STD)                                       │
└─────────────────────────────────────────────────────────────────────┘
                        Total = 84 + 18 + 36 + 3 + 3 = 144
```

### Feature Extraction Flow

```
acc (128,3)  gyro (128,3)  ori (128,3)
      │
      ├──► _lowpass(acc) → gravity (128,3)
      │          Hamming FIR, cutoff=0.3 Hz
      │
      ├──► linear_acc = acc - gravity
      │
      ├──► acc_mag  = ‖acc‖   per sample   → 128 scalars
      ├──► lin_mag  = ‖linear_acc‖          → 128 scalars
      ├──► gyro_mag = ‖gyro‖               → 128 scalars
      │
      ├──► acc_jerk  = diff(acc_mag)        → 128 scalars
      ├──► lin_jerk  = diff(lin_mag)        → 128 scalars
      ├──► gyro_jerk = diff(gyro_mag)       → 128 scalars
      │
      ├──► _mag_stats(each 6 signals)       → 6 × 14 = 84 features
      ├──► _mag_freq(acc_mag, lin_mag, gyro_mag) → 3 × 6 = 18 features
      ├──► _axis_stats(linear_acc, gyro)    → 2 × 18 = 36 features
      ├──► cross_corr(linear_acc)           → 3 features
      └──► ori_means(azimuth, pitch, roll)  → 3 features
                                                 ─────────
                                                144 features
```

### Client-Side vs. Server-Side Features

| | Flutter (`motion_feature_extractor.dart`) | Server (`motion_enhanced_features.py`) |
|--|--|--|
| Dimension | 128-D (legacy v1 design) | 144-D (v2 orientation-invariant) |
| Gravity separation | None | Hamming FIR low-pass at 0.3 Hz |
| Jerk features | No | Yes (6 jerk signals) |
| FFT | Naive DFT O(N²) | NumPy FFT O(N log N) |
| Band energy features | No | Yes (3 frequency bands) |
| Used for inference? | No — server rebuilds from raw windows | Yes — primary path |

**Server always recomputes** the 144-D vector from the raw 128×3 sensor matrices sent alongside. The Flutter client sends both the (now-legacy) 128-D vector and the raw windows; the server ignores the client features and builds its own at parity with training.

---

## 7. Model Training & Evaluation

### Model Architecture: Two-Stage XGBoost Pipeline

```
Raw sensor window (128 samples)
         │
         ▼
144-D Feature Vector (RobustScaled)
         │
    ┌────┴─────────────────────────────────┐
    │                                       │
    ▼                                       ▼
STAGE 1: Fall Detection              (same feature vector)
  XGBClassifier (binary)             ─── if is_fall=False ──►
  objective: binary:logistic                                  │
  n_estimators: 400                  STAGE 2: ADL Classification
  max_depth: 6                         XGBClassifier (multiclass)
  learning_rate: 0.05                  objective: multi:softprob
  subsample: 0.8                       n_classes: 5
  colsample_bytree: 0.8                same hyperparameters
  reg_alpha: 0.1                       outputs: JOG/LYI/SIT/STD/WAL
  reg_lambda: 1.0
  random_state: 42
         │
    p_fall ≥ 0.55?
    ┌────┘
    │ Yes: branch = "fall"
    │      → fall-type subclassifier (optional, disabled in v2)
    │
    └ No: branch = "adl"
         → ADL model predicts activity label
```

### Cross-Validation

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Scoring: F1 weighted
Performed on training split only
```

### Evaluation Results (Test Set — 22 unseen subjects)

#### Fall Detection (Binary: ADL vs Fall)

| Metric | Value |
|--------|-------|
| Accuracy | **99.31%** |
| F1 Weighted | **99.32%** |
| F1 Macro | **92.06%** |
| ROC-AUC | **99.72%** |
| 5-fold CV F1 mean | 99.76% |
| 5-fold CV F1 std | ±0.046% |

#### ADL Classification (5-class: JOG, LYI, SIT, STD, WAL)

| Metric | Value |
|--------|-------|
| Accuracy | **95.91%** |
| F1 Weighted | **95.94%** |
| F1 Macro | **91.08%** |
| 5-fold CV F1 mean | 99.40% |
| 5-fold CV F1 std | ±0.065% |

### Saved Artifacts

| File | Size | Purpose |
|------|------|---------|
| `baseline_adl&fall/fall_xgb_model.pkl` | 1.1 MB | Trained fall detection XGBoost |
| `baseline_adl&fall/fall_scaler.pkl` | 2.3 KB | RobustScaler for fall pipeline (144 features) |
| `baseline_adl&fall/adl_xgb_model.pkl` | 6.1 MB | Trained ADL classification XGBoost |
| `baseline_adl&fall/adl_scaler.pkl` | 2.3 KB | RobustScaler for ADL pipeline (144 features) |
| `baseline_adl&fall/adl_label_encoder.pkl` | 387 B | LabelEncoder: index ↔ class code |
| `inference_manifest.json` | — | Schema v2.0: paths, dims, thresholds |

### Inference Manifest (v2.0)

```json
{
  "schema_version": "2.0",
  "enhanced_feature_dim": 144,
  "fall_probability_threshold": 0.55,
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

### Threshold Design (Dual-Threshold System)

```
ML Fall Threshold (0.55):
  p_fall ≥ 0.55  → branch = "fall"  (routes inference to fall-type submodel)
  p_fall < 0.55  → branch = "adl"   (routes to ADL classifier)

Detector Alert Threshold (configurable, default 0.80):
  p_fall ≥ 0.80  → severity = "fall_detected" → alert created in DB + WebSocket push
  p_fall ≥ 0.58  → severity = "high_risk"
  p_fall ≥ 0.35  → severity = "medium"
  p_fall < 0.35  → severity = "low"

Caregiver sensitivity presets:
  Low:    fall_score=0.88, high_risk=0.68, medium=0.45
  Medium: fall_score=0.80, high_risk=0.58, medium=0.35  (default)
  High:   fall_score=0.72, high_risk=0.50, medium=0.28
```

---

## 8. Backend Architecture

### Directory Structure

```
flask_backend/
├── __init__.py
├── requirements.txt
├── captain-definition          ← Docker/CapRover build config
├── app/
│   ├── main.py                 ← FastAPI app init, lifespan, model loading
│   ├── monitoring_routes.py    ← All REST routes + WebSocket (1693 lines)
│   ├── ml_bridge.py            ← samples → 144-D features, VoteBuffer
│   ├── motion_enhanced_features.py  ← 144-D feature extractor (v2)
│   ├── detector_state.py       ← severity / score calculation
│   ├── database.py             ← SQLite schema + thread-safe connection
│   ├── auth_jwt.py             ← JWT + bcrypt
│   ├── settings.py             ← env-configurable paths
│   ├── realtime_hub.py         ← WebSocket caregiver registry + broadcast
│   ├── schemas_motion.py       ← Pydantic request/response models (inference)
│   ├── schemas_fall_feedback.py
│   ├── elder_credential_allocator.py
│   └── services/
│       └── motion_xgb_service.py  ← re-export from scripts/inference/
├── models/
│   ├── inference_manifest.json
│   └── baseline_adl&fall/
│       ├── fall_xgb_model.pkl
│       ├── fall_scaler.pkl
│       ├── adl_xgb_model.pkl
│       ├── adl_scaler.pkl
│       ├── adl_label_encoder.pkl
│       └── results_v2.json
└── tests/
    └── test_pipeline.py        ← 43 pytest tests
```

### Application Startup (Lifespan)

```
FastAPI lifespan startup:
  1. os.environ: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 (prevent thread storms)
  2. init_schema()      ← CREATE TABLE IF NOT EXISTS + idempotent migrations
  3. seed_default_admin()  ← create admin@local if no admin exists
  4. load_artifacts(manifest_path, model_root)
       ├── parse inference_manifest.json
       ├── joblib.load(adl_xgb_model.pkl + adl_scaler.pkl + adl_label_encoder.pkl)
       ├── joblib.load(fall_xgb_model.pkl + fall_scaler.pkl)
       ├── validate scaler.n_features_in_ == manifest enhanced_feature_dim (144)
       └── return InferenceArtifacts dataclass (frozen)
  5. set_inference_runtime(state)  ← expose artifacts to route handlers
```

### Complete REST API Reference

#### Authentication

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/auth/caregiver/signup` | None | Caregiver registration |
| POST | `/api/v1/auth/caregiver/login` | None | Caregiver login → JWT |
| POST | `/api/v1/auth/caregiver/patient-credentials` | Caregiver JWT | Enroll patient, generate elder username+password |
| POST | `/api/v1/auth/elder/login` | None | Elder login with generated credentials → JWT |
| POST | `/api/v1/auth/admin/login` | None | Admin login → JWT |

#### System

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/health` | None | Health + model load status |
| GET | `/api/v1/inference/status` | None | Model dimensions, thresholds |
| GET | `/api/v1/summary` | Any | Patient/alert counts |

#### Patients & Devices

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/patients` | Optional | Create patient record |
| GET | `/api/v1/patients/{id}` | Any | Get patient |
| POST | `/api/v1/devices` | Any | Register device for patient |
| GET | `/api/v1/devices/{id}` | Any | Get device |
| POST | `/api/v1/sessions` | Any | Start monitoring session |
| POST | `/api/v1/sessions/{id}/stop` | Any | Stop session |

#### Caregiver

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/caregiver/my-patients` | Caregiver JWT | List enrolled patients |
| DELETE | `/api/v1/caregiver/my-patients/{id}` | Caregiver JWT | Remove enrollment |

#### Live Monitoring

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/ingest/live` | Any | **Primary endpoint**: sensor batch → inference → detection → alert |
| GET | `/api/v1/monitor/patients/live` | Any | All live patient statuses |
| POST | `/api/v1/patients/me/location` | Elder JWT | Elder GPS upload |
| PUT | `/api/v1/detector/config` | Any | Update alert sensitivity thresholds |

#### Inference (standalone)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/inference/motion` | Any | Run model on provided features/windows (no session needed) |

#### Alerts

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/alerts` | Any | List alerts (filterable by status/patient) |
| POST | `/api/v1/alerts/manual` | Elder/Caregiver JWT | Trigger manual emergency alert |
| POST | `/api/v1/alerts/{id}/acknowledge` | Any | Acknowledge alert |
| POST | `/api/v1/alerts/{id}/resolve` | Any | Resolve alert |

#### Admin

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/admin/dashboard` | Admin JWT | System-wide dashboard stats |
| GET/POST/DELETE | `/api/v1/admin/caregivers` | Admin JWT | Manage caregiver accounts |
| GET/POST/DELETE | `/api/v1/admin/patients` | Admin JWT | Manage patient records |

#### WebSocket

| Protocol | Path | Auth | Description |
|----------|------|------|-------------|
| WS | `/api/v1/ws/caregiver?token=<JWT>` | Caregiver JWT | Real-time fall alert push |

#### Fall Feedback

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/events/fall-feedback` | Any | Submit elder fall confirmation/denial |

---

## 9. Frontend Architecture

### Flutter App Structure

```
app_frontend/
├── pubspec.yaml
├── lib/
│   ├── main.dart                         ← app entry, Provider setup
│   └── src/
│       ├── app.dart                      ← MaterialApp root, theme
│       ├── role_launcher.dart            ← Role selection screen
│       ├── api_client.dart               ← All HTTP calls
│       ├── api_config.dart               ← Backend URL constant
│       ├── models.dart                   ← Dart data models (Pydantic mirrors)
│       ├── monitoring_controller.dart    ← Central ChangeNotifier state (1880 lines)
│       ├── sensor_streaming_service.dart ← IMU reading + windowing
│       ├── motion_feature_extractor.dart ← Client-side feature extraction (legacy)
│       ├── motion_inference_helper.dart  ← Calls /api/v1/inference/motion
│       ├── patient_shell_pages.dart      ← Patient (elder) tab navigation
│       ├── admin_dashboard_screen.dart   ← Admin screen
│       ├── roles/app_roles.dart          ← Role constants
│       ├── http_client_factory.dart      ← Platform-agnostic HTTP
│       ├── http_client_factory_io.dart   ← Mobile/desktop HTTP
│       └── http_client_factory_web.dart  ← Web HTTP (CORS-safe)
└── test/
    └── widget_test.dart
```

### State Management: ChangeNotifier

`MonitoringController` is a single `ChangeNotifier` injected at the root via `Provider`. It is the single source of truth for:

| State | Type | Description |
|-------|------|-------------|
| `isStreaming` | bool | Whether sensor batch uploads are active |
| `lastDetection` | `DetectionResultModel?` | Most recent fall probability + severity |
| `liveStatus` | `LiveStatusModel?` | Polled live row (patient status summary) |
| `caregiverAlerts` | `List<AlertRecordModel>` | All open/recent alerts |
| `livePatients` | `List<LiveStatusModel>` | All live patient rows (caregiver view) |
| `activeAlert` | `AlertRecordModel?` | Currently active unresolved alert |
| `isAlarmPlaying` | bool | Whether the alarm ringtone is looping |
| `currentPosition` | `Position?` | Latest GPS fix |
| `isCaregiverAuthenticated` | bool | Caregiver JWT present |
| `hasElderSession` | bool | Elder JWT present |

### Role-Based UI Flows

```
App Launch → RoleLauncher
    │
    ├─── Elder / Patient
    │         │
    │         ├─ Elder Login (username + password from caregiver enrollment)
    │         │       → applyElderSession() → stores elder JWT
    │         │
    │         └─ patient_shell_pages.dart (tab navigation)
    │              ├── Live Tab      (risk meter, activity label, sensor preview)
    │              ├── Alert Tab     (manual emergency button)
    │              ├── Map Tab       (GPS location, home marker, directions)
    │              └── Settings Tab  (backend URL, device label, profile)
    │
    ├─── Caregiver
    │         │
    │         ├─ Caregiver Login / Signup
    │         │       → caregiverLogin() → stores caregiver JWT
    │         │       → connects WebSocket (/api/v1/ws/caregiver?token=...)
    │         │       → refreshCaregiverData() (summary + live patients + alerts)
    │         │
    │         └─ Caregiver Dashboard
    │              ├── Patient List  (live severity cards per patient)
    │              ├── Alerts Panel  (acknowledge / resolve)
    │              ├── Enroll Patient (generate username+password)
    │              └── Settings (sensitivity, alarm, email)
    │
    └─── Admin
              │
              ├─ Admin Login
              └─ Admin Dashboard
                   ├── Manage Caregivers (create / delete)
                   └── Manage Patients   (create / delete)
```

### Sensor Streaming Service

```
SensorStreamingService (windowSize=128, stepSize=64, targetHz=50)
    │
    ├── GyroscopeEventStream  → stores latest (gx,gy,gz) at highest available rate
    │
    ├── MotionCore.motionStream (fused quaternion attitude)
    │       → converts yaw/pitch/roll (rad) to MobiAct (azimuth/pitch/roll, degrees)
    │       → mobiActOrientationFromMotion():
    │           azimuth = yaw × (180/π), normalised to [0, 360)
    │           pitch   = roll_x × (180/π)
    │           roll    = pitch_y × (180/π)
    │
    └── AccelerometerEventStream  (primary timer: enforces 50 Hz minimum gap)
            → on each accelerometer tick:
                 if gap < 20ms: skip (enforce ≤50 Hz)
                 buffer.add(SensorReadingPayload{accX,Y,Z, gyroX,Y,Z, azimuth,pitch,roll})
                 if buffer.length ≥ 128:
                     take first 128 → batch
                     removeRange(0, 64)  ← 50% overlap
                     onBatch(batch)      → _handleSensorBatch in MonitoringController
```

### Key Flutter Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sensors_plus` | ^7.0.0 | Raw accelerometer + gyroscope streams |
| `motion_core` | ^0.0.5 | Fused quaternion orientation (AHRS) |
| `web_socket_channel` | ^3.0.1 | WebSocket for caregiver alert push |
| `geolocator` | ^14.0.2 | GPS position stream |
| `flutter_map` | ^8.3.0 | Interactive map (elder home / current position) |
| `shared_preferences` | ^2.3.2 | Persist backend URL, patient IDs, JWTs |
| `flutter_ringtone_player` | ^4.0.0+4 | Alarm ringtone for caregiver fall alerts |
| `wakelock_plus` | ^1.2.8 | Prevent screen sleep during monitoring |
| `http` | ^1.2.2 | HTTP client with timeout+retry |
| `url_launcher` | ^6.3.1 | Open Google Maps directions |

---

## 10. Frontend–Backend Integration

### Live Ingest Flow (Primary Data Path)

```
Flutter (Elder Phone)
        │
        │ SensorReadingPayload × 128
        │   { timestamp_ms, acc_x/y/z, gyro_x/y/z, azimuth, pitch, roll }
        │
        ▼
POST /api/v1/ingest/live
  { patient_id, device_id, session_id,
    source: "flutter_mobile",
    sampling_rate_hz: 50.0,
    acceleration_unit: "m_s2",
    gyroscope_unit: "rad_s",
    samples: [ ...128 SensorReadingPayload ] }
        │
        ▼
Backend: ingest_live()
  1. samples_to_feature_vector(samples)
       ├── parse acc/gyro/ori from dicts (zeros if missing)
       ├── _resample_rows(acc, 128)  ← linear interpolation to exactly 128
       └── extract_enhanced_features([window]) → feat (144,)

  2. run_inference(art, feat.tolist(), ...)
       ├── fall_scaler.transform(feat) → scaled
       ├── fall_model.predict_proba(scaled)[0,1] → p_fall
       ├── p_fall ≥ 0.55 → branch="fall"
       │     fall_type submodel (disabled) → fall_type_code=null
       └── p_fall < 0.55 → branch="adl"
             adl_scaler.transform(feat) → scaled_adl
             adl_model.predict(scaled_adl) → class_idx
             adl_encoder.inverse_transform([idx]) → "WAL" / "JOG" / etc.

  3. VoteBuffer.push(activity_label, is_fall=False)
       ← 7-window majority vote for ADL stability

  4. build_detection_payload(samples, p_fall, ...)
       ├── simple_signal_metrics(samples)
       │     → peak_acc_g, peak_gyro_dps, peak_jerk, stillness
       ├── _effective_fall_probability(p_fall, sig)
       │     ← dampen if looks_stationary (gyro < 95 dps, acc < 2.05g, stillness ≥ 0.62)
       └── _severity_from_fall_prob(p_eff, thr=0.80)
             → "fall_detected" / "high_risk" / "medium" / "low"

  5. If severity == "fall_detected" AND no open incident:
       INSERT INTO alerts (...)
       INSERT INTO fall_incidents (stage="awaiting_response", deadlines)
       background_tasks.add_task(_broadcast_alert_ws, caregiver_id, alert_payload)
                                   └── hub.broadcast_to_caregiver(cg_id, payload)

  6. UPSERT patient_live (latest status row per patient)

  7. Return IngestResponse:
       { detection: {...}, live_status: {...}, active_alert: {...}, telemetry: {...} }
        │
        ▼
Flutter receives response:
  _lastDetection = response.detection
  _liveStatus    = response.liveStatus
  _activeAlert   = response.activeAlert (if any)
  _statusMessage updated
  notifyListeners() → UI rebuilds

  Optional: call POST /api/v1/inference/motion with raw windows
  ← provides MotionInferenceResponseModel (fall_type_code, activity details)
```

### WebSocket Alert Flow (Caregiver Phone)

```
Caregiver logs in
    │
    ▼
MonitoringController._connectCaregiverAlertSocketIfNeeded()
    │
    └── WebSocketChannel.connect(
          ws[s]://<host>/api/v1/ws/caregiver?token=<JWT>
        )
          │
          ├── server: decode JWT, register cg_id in hub
          │
          └── ch.stream.listen( onData: (raw) {
                  m = jsonDecode(raw)
                  if m['type'] == 'alert':
                      refreshCaregiverData(silent: true)
                      _syncAlarmWithAlerts()
              })

When fall is detected (ingest endpoint):
    background_tasks.add_task(
        _broadcast_alert_ws, caregiver_id, alert_payload
    )
        │
        ▼
    hub.broadcast_to_caregiver(cg_id, {"type": "alert", ...payload})
        │
        ▼
    Caregiver phone WebSocket receives message
        → refreshCaregiverData() fetches updated alerts + live status
        → _syncAlarmWithAlerts() → starts alarm ringtone if alarm_eligible
```

### Caregiver Refresh Loop

```
Timer.periodic(Duration(seconds: 2)) (background)
    │
    └── refreshCaregiverData(silent: true)
              │
              ├── getSummary()           GET /api/v1/summary
              ├── getLivePatients()      GET /api/v1/monitor/patients/live
              ├── getAlerts()            GET /api/v1/alerts
              └── getCaregiverMyPatients() GET /api/v1/caregiver/my-patients
```

### Elder GPS Upload Flow

```
Geolocator.getPositionStream (distanceFilter=5m)
    │
    └── _throttledUploadPatientLocation(position)
              ├── require elder JWT and ≥25s since last upload
              └── POST /api/v1/patients/me/location
                    { latitude, longitude, accuracy_m, heading_degrees }
                         │
                         ▼
                  backend upserts patient_live.latitude/longitude
                         │
                         ▼
                  Caregiver map shows elder location on flutter_map
```

---

## 11. Real-Time Inference Pipeline

### Complete End-to-End Flow Diagram

```
Phone IMU Hardware (50 Hz)
         │
         │  acc (x,y,z) m/s²  +  gyro (x,y,z) rad/s  +  ori (az,pitch,roll) °
         ▼
SensorStreamingService
  Buffer 128 samples, slide by 64 (50% overlap)
         │ batch of 128 SensorReadingPayload
         ▼
MonitoringController._handleSensorBatch()
         │
         │ POST /api/v1/ingest/live (JSON, ~25KB per batch)
         ▼
FastAPI: ingest_live()
         │
         ├─ samples_to_feature_vector()
         │      │
         │      ├─ parse dict → acc(n,3), gyro(n,3), ori(n,3)
         │      ├─ _resample_rows() → acc(128,3), gyro(128,3), ori(128,3)
         │      └─ extract_enhanced_features()
         │              │
         │              ├─ _lowpass(acc) → gravity
         │              ├─ linear_acc = acc - gravity
         │              ├─ compute 6 magnitude/jerk signals
         │              ├─ _mag_stats × 6 → 84 features
         │              ├─ _mag_freq  × 3 → 18 features
         │              ├─ _axis_stats × 2 → 36 features
         │              ├─ cross_corr → 3 features
         │              └─ ori_means → 3 features
         │                          = 144-D float64 vector
         │
         ├─ run_inference(art, feat_144, ...)
         │      │
         │      ├─ fall_scaler.transform(feat_144)
         │      ├─ fall_model.predict_proba() → p_fall ∈ [0,1]
         │      │
         │      └─ p_fall ≥ 0.55?
         │            │YES              │NO
         │            ▼                ▼
         │       branch="fall"    adl_scaler.transform(feat_144)
         │       (fall type       adl_model.predict() → class_idx
         │        disabled)       adl_encoder → "WAL"/"JOG"/etc.
         │
         ├─ VoteBuffer.push(label, is_fall)
         │      ← 7-window majority vote (falls reset buffer)
         │      → smoothed activity label
         │
         ├─ build_detection_payload()
         │      │
         │      ├─ simple_signal_metrics() → peak_acc_g, peak_gyro_dps
         │      ├─ _effective_fall_probability() ← stationary guard
         │      └─ _severity_from_fall_prob(p_eff, thr)
         │            → "fall_detected" / "high_risk" / "medium" / "low"
         │
         ├─ [if fall_detected]
         │      INSERT alert + fall_incident into SQLite
         │      background: broadcast WebSocket to caregiver
         │
         ├─ UPSERT patient_live row
         │
         └─ Return JSON response
                  │
         ┌────────┴───────────┐
         ▼                    ▼
Elder phone UI          Caregiver WebSocket
  risk meter              alert notification
  activity label          alarm ringtone
  severity colour         dashboard refresh
```

### VoteBuffer — ADL Smoothing

```
Per-patient VoteBuffer (deque, maxlen=7)
         │
         ├── Falls bypass voting immediately (label = "Fall")
         │     → buffer.reset()
         │
         └── ADL: buf.push(label)
               if len(buf) < 7: return raw label
               else: return Counter(buf).most_common(1)[0][0]
                      ← majority vote over last 7 windows ≈ 3.5 s
```

### Stationary Motion Guard

```
Input: ML fall probability p_fall, raw signal metrics

if p_fall ≤ 0.35: return p_fall unchanged

looks_stationary = (
    peak_gyro_dps < 95.0
    AND peak_acc_g < 2.05
    AND stillness_ratio ≥ 0.62
)

if looks_stationary:
    p_eff = min(p_fall, 0.14 + p_fall × 0.22)  ← soft dampen
    ← prevents pocket/desk noise from triggering alerts
else:
    p_eff = p_fall

Additional fall_detected guard:
    if severity="fall_detected" but peak_acc_g < 1.70g AND peak_gyro_dps < 180 dps:
        downgrade to "high_risk"  ← no physical impact evidence
```

---

## 12. Database Schema

SQLite database at `data/elder_monitor.db` (auto-created on first run).

### Entity Relationship Overview

```
users ──────────── patients ──────── devices
  │                   │                │
  │ (caregiver_id)    │ (patient_id)   │ (device_id)
  │                   │                │
  ├── caregiver_patient     sessions ──┘
  │     (many-to-many)         │
  │                            │ (session_id)
  └── [elder_user_id]          │
                          alerts
                          fall_incidents
                          app_events
                          patient_live
```

### Table Definitions

#### `users`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `email` | TEXT UNIQUE | null for elders |
| `username` | TEXT UNIQUE | null for caregivers/admins |
| `password_hash` | TEXT | bcrypt |
| `role` | TEXT | `admin` / `caregiver` / `elder` |
| `full_name` | TEXT | |
| `created_at` | TEXT | ISO 8601 UTC |

#### `patients`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `full_name` | TEXT | |
| `age` | INTEGER | |
| `caregiver_id` | TEXT FK→users | primary caregiver |
| `elder_user_id` | TEXT FK→users | linked elder login |
| `home_address` | TEXT | |
| `emergency_contact` | TEXT | |
| `notes` | TEXT | medical notes |

#### `caregiver_patient`
| Column | Type | Notes |
|--------|------|-------|
| `caregiver_id` | TEXT FK→users | composite PK |
| `patient_id` | TEXT FK→patients | composite PK |

#### `devices`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `patient_id` | TEXT FK | |
| `label` | TEXT | e.g. "Patient phone" |
| `platform` | TEXT | e.g. "flutter_mobile" |

#### `sessions`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `patient_id` | TEXT FK | |
| `device_id` | TEXT FK | |
| `status` | TEXT | `active` / `stopped` |
| `sample_rate_hz` | REAL | typically 50.0 |
| `started_at` | TEXT | ISO 8601 |
| `stopped_at` | TEXT | null if active |

#### `alerts`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `patient_id` | TEXT | |
| `device_id` | TEXT | |
| `session_id` | TEXT | |
| `severity` | TEXT | `fall_detected` / `high_risk` / `medium` / `low` |
| `status` | TEXT | `open` / `acknowledged` / `resolved` |
| `message` | TEXT | human-readable reason |
| `score` | REAL | composite risk score 0–1 |
| `created_at` | TEXT | |
| `acknowledged_at` | TEXT | |
| `resolved_at` | TEXT | |
| `manually_triggered` | INTEGER | 0=auto, 1=manual |

#### `fall_incidents`
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID hex |
| `patient_id` | TEXT | |
| `session_id` | TEXT | |
| `stage` | TEXT | `awaiting_response` / `alarm_local` |
| `created_at` | TEXT | |
| `response_deadline_at` | TEXT | `created_at` + `FALL_RESPONSE_DEADLINE_SEC` (default 30s) |
| `alarm_deadline_at` | TEXT | `created_at` + `FALL_EMERGENCY_DEADLINE_SEC` (default 90s) |
| `fall_probability` | REAL | raw ML p_fall |
| `fall_type_code` | TEXT | FOL/FKL/BSC/SDL or null |
| `response` | TEXT | elder acknowledgement |
| `metadata_json` | TEXT | `{"branch": "fall", "ml_ok": true}` |

#### `patient_live`
| Column | Type | Notes |
|--------|------|-------|
| `patient_id` | TEXT PK | one row per patient |
| `patient_name` | TEXT | |
| `session_id` | TEXT | |
| `device_id` | TEXT | |
| `severity` | TEXT | latest severity |
| `score` | REAL | latest risk score |
| `fall_probability` | REAL | latest ML p_fall |
| `predicted_activity_class` | TEXT | latest ADL label |
| `last_message` | TEXT | |
| `sample_rate_hz` | REAL | |
| `active_alert_ids` | TEXT | JSON array of alert IDs |
| `updated_at` | TEXT | |
| `latitude` | REAL | latest GPS (migrated column) |
| `longitude` | REAL | |
| `location_accuracy_m` | REAL | |
| `location_updated_at` | TEXT | |
| `heading_degrees` | REAL | |

---

## 13. Authentication & Security

### Auth Flow

```
CAREGIVER SIGNUP / LOGIN
  POST /auth/caregiver/signup { full_name, email, password }
       │
       ├── hash_password(password) ← bcrypt.hashpw with gensalt()
       ├── INSERT INTO users (role="caregiver")
       └── create_token(user_id, role="caregiver", email) → JWT
               │ HS256, SECRET from env JWT_SECRET, TTL=7 days
               ▼
           Flutter stores token, sets Authorization: Bearer <token>

PATIENT ENROLLMENT (Caregiver only)
  POST /auth/caregiver/patient-credentials { caregiver_token, full_name, age, home_address }
       │
       ├── decode caregiver JWT → caregiver_id
       ├── generate elder username (slug of name + 4-digit suffix)
       ├── generate temporary password (random alphanumeric)
       ├── INSERT patient + INSERT elder user + INSERT caregiver_patient
       └── return { patient_id, username, temporary_password }
               ▼ (caregiver gives credentials to elder)

ELDER LOGIN
  POST /auth/elder/login { username, password }
       │
       ├── SELECT user WHERE username=?
       ├── bcrypt.checkpw(password, password_hash)
       └── create_token(user_id, role="elder", email=f"{username}@patients.local")
               ▼ Flutter stores elder JWT, uses for GPS uploads
```

### JWT Payload Structure

```json
{
  "sub": "<user_id_hex>",
  "role": "caregiver | elder | admin",
  "email": "<email_or_derived>",
  "iat": 1715000000,
  "exp": 1715604800
}
```

### Security Notes

| Concern | Implementation |
|---------|---------------|
| Password storage | bcrypt with random salt (bcrypt.gensalt()) |
| Token signing | HS256, secret from `JWT_SECRET` env var |
| Token expiry | 7 days |
| Role enforcement | `_claims_opt(authorization)` guard on each route |
| Admin default | `admin@local` / `admin123` — must be changed via `ADMIN_EMAIL`/`ADMIN_PASSWORD` env |
| CORS | `allow_origins=["*"]` — restrict in production |
| Thread safety | SQLite wrapped with `threading.Lock()` |

---

## 14. Deployment & Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | `sisfall-dev-change-me-in-production` | JWT signing secret |
| `ADMIN_PASSWORD` | `admin123` | Default admin password |
| `ADMIN_EMAIL` | `admin@local` | Default admin email |
| `MODEL_ROOT` | `flask_backend/models` | Directory containing model PKL files |
| `INFERENCE_MANIFEST` | `flask_backend/models/inference_manifest.json` | Manifest path |
| `REPO_ROOT` | Auto-detected from `__file__` | Repository root (for scripts/ path) |
| `FALL_RESPONSE_DEADLINE_SEC` | `30` | Seconds before escalating to local alarm |
| `FALL_EMERGENCY_DEADLINE_SEC` | `90` | Seconds before caregiver emergency mode |
| `EMS_DEBUG_SENSOR_LOGS` | `1` | Log raw sensor values per ingest batch |
| `OMP_NUM_THREADS` | `1` | Prevent XGBoost thread storms |
| `OPENBLAS_NUM_THREADS` | `1` | Prevent NumPy thread storms |

### CapRover / Docker Config (`captain-definition`)

```json
{
  "schemaVersion": 2,
  "dockerfileLines": [
    "FROM python:3.11-slim",
    "WORKDIR /app",
    "COPY flask_backend/requirements.txt .",
    "RUN pip install -r requirements.txt",
    "COPY . .",
    "CMD [\"uvicorn\", \"flask_backend.app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"80\"]"
  ]
}
```

### Running Locally

```bash
# Backend
cd ems/
pip install -r flask_backend/requirements.txt
uvicorn flask_backend.app.main:app --reload --port 8000

# Tests
python -m pytest flask_backend/tests/test_pipeline.py -v

# Flutter frontend
cd app_frontend/
flutter pub get
flutter run
```

### Flutter Backend URL Configuration

```dart
// app_frontend/lib/src/api_config.dart
class AppApiConfig {
  static const String backendBaseUrl = 'http://10.0.2.2:8000'; // Android emulator
  // Production: 'https://your-server.example.com'
}
```

---

## 15. Test Coverage

### Test Suite: `flask_backend/tests/test_pipeline.py`

**43 tests across 6 test classes — all passing.**

| Class | Tests | What is Verified |
|-------|-------|-----------------|
| `TestFeatureDimension` | 8 | Feature vector = exactly 144; no NaN/Inf; batch shapes correct; zero-gyro edge case |
| `TestSubFunctions` | 5 | `_lowpass` shape; `_mag_stats` = 14; `_mag_freq` = 6; `_axis_stats` = 18; constant-signal stats |
| `TestBridgeHelpers` | 6 | `build_enhanced_features_numpy` shape (144,); `samples_to_feature_vector` shapes (144,)+(300,3)×3; empty raises |
| `TestVoteBuffer` | 7 | Falls bypass+reset; majority vote; raw label below buffer size; reset clears; confidence 100%/75% |
| `TestModelLoading` | 10 | Manifest exists; `enhanced_feature_dim=144`; threshold=0.55; all PKL files present; `load_artifacts` succeeds; fall binary enabled; scaler dims = 144 |
| `TestEndToEndInference` | 7 | ADL response schema; walking window low fall prob; activity label in {JOG,LYI,SIT,STD,WAL}; fall branch schema; wrong dim raises; full sample ingest pipeline |

```
============================= test session starts =============================
43 passed in 4.52s
==============================
```

---

## Appendix — Scripts Directory

```
scripts/
├── baseline_fall/
│   ├── mobiact_dataset.py     ← CSV loading + segment-aware windowing
│   ├── enhanced_features.py   ← v1 128-D feature extractor (training reference)
│   ├── subject_split.py       ← subject-aware train/test partitioning
│   ├── sampling.py            ← class balancing utilities
│   ├── fall_detection_core.py ← core training orchestration
│   ├── fall_detection_models.py ← XGBoost model builder
│   ├── train_mobiact_baselines.py ← entry point for v2 training
│   └── train_fall_detection_mobiact.py ← fall detection training
│
├── baseline_adl/
│   ├── data_prep.py           ← ADL-specific data loading
│   ├── models.py              ← ADL model builder
│   └── train_mobiact_adl.py   ← ADL training entry point
│
├── baseline_falltype/
│   ├── feature_extractors.py  ← 263-D fall-type features
│   ├── feature_selection_mi.py ← mutual information feature selection (150-D)
│   ├── fall_type_models.py    ← fall-type multi-class model
│   └── train_fall_type_mobiact.py ← fall-type training
│
├── inference/
│   └── motion_pipeline.py     ← canonical inference: load_artifacts + run_inference
│
├── run_training.py            ← unified training runner
├── sync_inference_manifest.py ← update manifest after training
├── verify_mobiact_baseline_inference.py ← smoke test trained models
├── simulate_inference_demo.py ← demo with synthetic data
└── baseline_models_comparison.py ← compare v1 vs v2 features
```

---

*Report generated: 2026-05-10 | Model version: v2 (144-D orientation-invariant) | Schema version: 2.0*
