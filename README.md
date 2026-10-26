# APK Guardian — Explainable AI-Assisted Android Malware Detection

APK Guardian is an end-to-end Android malware detection and risk assessment system combining static analysis with Androguard, a Random Forest machine learning pipeline, rule-based explainability, and an APK sample library.

---

## 1. Deliverables & Built Android Installers

The compiled Android application packages (`.apk`) are ready for installation on any Android device running Android 7.0 (API 24) or newer.

| Artifact | Location | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Release APK (Optimized)** | `dist/APK_Guardian_release.apk`<br>`frontend/build/app/outputs/flutter-apk/app-release.apk` | ~484.9 MB | `575F6C76A839352D64BF3822F93BE6AA1E1209078A4AACE94E21C9639B683DE1` |
| **Debug APK** | `dist/APK_Guardian_debug.apk`<br>`frontend/build/app/outputs/flutter-apk/app-debug.apk` | ~1.31 GB | `2FB3D270C38BCFB896E303A9722C5593A69DA72CCCE57EAC82A7A3D65AF0EE21` |

> **Installation Note**: You can install either APK onto an Android device or emulator with:
> ```bash
> adb install dist/APK_Guardian_release.apk
> ```

---

## 2. Key Architecture & Features

### 2.1 Frontend (Flutter / Material 3)
- **Professional Cybersecurity UI**: Modern dark theme with navy/slate palettes (`#0F172A`, `#1E293B`), subtle cyan/teal accents, rounded cards, clean typography, and status indicators.
- **Security Dashboard**: Live backend connection indicator, scan statistics (Total Scanned, Benign, Suspicious), Model Status Banner, Quick Action buttons, Default APK Library overview, and recent scans feed.
- **Default APK Library Screen**:
  - Browse pre-indexed samples in `data/apk_library/{benign,malicious,unclassified}`.
  - Search by filename or SHA-256 hash.
  - Filter by ground-truth labels and scan statuses (`COMPLETED`, `NOT_SCANNED`, `MODEL_UNAVAILABLE`, etc.).
  - Sort by date added, filename, or file size.
  - View full SHA-256 hash with 1-click clipboard copy.
  - **1-Click Scan**: Direct analysis trigger without having to manually pick files.
  - **Import APK Action**: Upload new samples from device with user-defined ground-truth tags.
  - **Safe Deletion**: Unlink library samples without deleting historical scan records.
- **Scan New APK Workflow**: Interactive file picker for `.apk` files, pre-scan validation, live upload progress, and report presentation.
- **Explainable Reports Screen**: Transparent breakdown separating observed APK manifest/DEX facts (`OBSERVED`), Random Forest classification (`ML`), and heuristic risk scoring (`HEURISTIC`).
- **Configurable Server Address**: Dynamically change backend host/port in settings (e.g. `http://10.0.2.2:8000` for Android emulator or `http://192.168.x.x:8000` for physical Android phones).

### 2.2 Backend (FastAPI / Python)
- **Static Analysis Engine**: Uses Androguard 4.1.4 to inspect binary manifests (`AndroidManifest.xml`) and Dalvik bytecode (`classes.dex`).
- **Feature Schema**: 8 documented static features:
  1. `permissions_count`
  2. `activities_count`
  3. `services_count`
  4. `receivers_count`
  5. `api_calls_count` (distinct external framework method references)
  6. `strings_count`
  7. `dangerous_permissions` (pinned watch-list)
  8. `suspicious_api_calls` (pinned watch-list)
- **Security & Safety**:
  - Uploaded APKs are **strictly analyzed statically** and **never executed**.
  - Path traversal protection and UUID-based temporary file handling.
  - Safe removal of temporary files after extraction.
  - Concurrency control via semaphores (`max_concurrent_scans`).
- **Database (SQLite + SQLAlchemy 2.0)**:
  - Users, Scans, Reports, and Library Samples tables.
  - Deduplication by SHA-256 hash.
  - Reusable scan cache for identical files.
- **Default APK Library Management**:
  - Configurable directory (`data/apk_library/`).
  - Startup synchronization indexing real `.apk` files.
  - Separation of ground-truth labels from model predictions.

---

## 3. Getting Started & Running

### 3.1 Backend Setup & Startup
Ensure Python 3.10+ is installed.

```powershell
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Initialize .env with a secure random key
python -m app.cli generate-secret --write-env

# Initialize the SQLite database schema
python -m app.cli init-db

# Check environment, database, androguard, and model status
python -m app.cli check

# Start the FastAPI server
python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

The interactive OpenAPI documentation will be available at:
`http://localhost:8000/docs`

### 3.2 Connecting the Mobile App
1. **On Android Emulator**:
   - The Android emulator accesses the host machine through `http://10.0.2.2:8000`.
   - In APK Guardian, navigate to **Settings** and set the server URL to `http://10.0.2.2:8000`.
2. **On Physical Android Device**:
   - Ensure your computer and Android phone are on the same local Wi-Fi network.
   - Run `ipconfig` on Windows to find your IPv4 address (e.g., `192.168.1.50`).
   - In APK Guardian **Settings**, set the server URL to `http://192.168.1.50:8000`.
   - Tap **Save and test** to verify the connection.

---

## 4. Test Suite & Verification Results

All tests pass 100% with zero static analysis warnings or errors.

| Component | Test Suite | Tests Run | Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Backend Unit Tests** | `pytest tests/` | 318 | **318 PASSED** | Feature extraction, hashing, auth, library API, rate limiting, and contract verification. |
| **Frontend Unit & Widget Tests** | `flutter test` | 95 | **95 PASSED** | Screen layouts, mock backends, auth flows, scan flows, library state, and edge cases. |
| **Frontend Code Quality** | `flutter analyze` | - | **0 ISSUES** | Clean compilation with zero errors, zero warnings, and zero linter notices. |

---

## 5. Machine Learning & Dataset Notes

- **Model Serialization & Baseline**:
  - The Random Forest classifier bundle (`rf_baseline.joblib`, `feature_schema.json`, and metadata) is located in [`models/`](file:///C:/Users/dhara/.gemini/antigravity/scratch/APK_Guardian/models).
  - Feature schemas are strictly pinned: any missing or unexpected feature column raises a validation error.
  - In development and demonstration mode, `ALLOW_SYNTHETIC_MODEL=true` in `.env` activates the scikit-learn model inference pipeline while maintaining visible, honest labeling in the Flutter UI and JSON responses.
  - To train on a genuine dataset (e.g. CICMalDroid 2020 or Drebin), follow the instructions in [`docs/dataset_documentation.md`](file:///C:/Users/dhara/.gemini/antigravity/scratch/APK_Guardian/docs/dataset_documentation.md).
- **Honest Labeling**:
  - If no model is trained, APK Guardian marks the model status as `MODEL_UNAVAILABLE`, running the rule-based heuristic engine and reporting `basis: HEURISTIC_ONLY`.
  - When a model trained on synthetic data is used for pipeline testing, `trained_on_synthetic_test_data` is flagged `true`, and the UI visibly warns users that it is a test artifact.
  - To train a genuine model, place real samples according to `docs/dataset_documentation.md` and run:
    ```bash
    python ml_training/prepare_dataset.py --manifest dataset/metadata/manifest.csv
    python ml_training/train_model.py
    ```
