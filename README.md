# Forensic Fingerprint Identifier

A Flask-based fingerprint identification system that uses a Siamese neural network (`siamese_best.pt`) to match a query fingerprint against enrolled and indexed fingerprint databases.

## Features

- Enroll fingerprint images for a person (local DB)
- Index external dataset folders (for example SOCOFing)
- Build and reuse embedding index for faster search
- Identify a query fingerprint as:
  - **KNOWN PERSON FOUND**
  - **AMBIGUOUS MATCH**
  - **UNKNOWN / NOT IN DATABASE**
- Optional FAISS backend for fast inner-product search (falls back to NumPy)
- Maintenance actions from UI:
  - Rebuild index
  - Clear embeddings
  - Clear sources
  - Clear local DB
  - Clean reset all

## Tech Stack

- Python
- Flask
- PyTorch + TorchVision
- NumPy
- Pillow
- Optional: FAISS (`faiss-cpu`)

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── siamese_best.pt
├── db_sources.json
├── fingerprint_db/
├── archive/
│   └── SOCOFing/
└── templates/
    └── index.html
```

## Setup

### 1) Clone repository

```bash
git clone <your-repo-url>
cd "JV PROJ EXECUTION"
```

### 2) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

Optional (recommended for faster search):

```bash
pip install faiss-cpu
```

## Run the App

```bash
python app.py
```

Open in browser:

- http://127.0.0.1:5000

## How to Use

1. **Enroll Person Fingerprints**
   - Enter a `Person ID`
   - Upload 1 to 10 fingerprint images
2. **Index Existing Dataset Folder**
   - Provide absolute path to folder containing fingerprints
   - Example: `.../archive/SOCOFing`
3. **Identify Query Fingerprint**
   - Upload a query fingerprint image
   - Set threshold and margin values
   - Run identification

## Notes

- Model file `siamese_best.pt` must be present at project root.
- Local enrollment data is stored in `fingerprint_db/`.
- Indexed source folders are stored in `db_sources.json`.
- Embedding artifacts (`embedding_index.npz`, `embedding_index_meta.json`) are auto-generated.

## Troubleshooting

- **Model not found**: ensure `siamese_best.pt` exists in project root.
- **No enrolled fingerprints found**: enroll data or index a dataset folder first.
- **Import error for FAISS**: install with `pip install faiss-cpu` or continue using NumPy backend.

## License

Add your preferred license (MIT/Apache-2.0/etc.) before public release.
