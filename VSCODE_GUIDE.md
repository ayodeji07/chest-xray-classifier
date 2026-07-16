# Chest X-Ray Classifier — VSCode Setup & Run Guide

From a fresh clone to a running Streamlit demo.

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| Git | any |
| VSCode | any |
| 200MB free disk space | for subset mode |

---

## Phase 1 — Clone and environment

```bash
git clone https://github.com/ayodeji07/chest-xray-classifier
cd chest-xray-classifier

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
```

---

## Phase 2 — Choose your training mode

Open `.env` and set `TRAINING_MODE`:

```bash
TRAINING_MODE=subset   # default — 500 images, ~200MB, runs on any laptop
TRAINING_MODE=full     # full NIH dataset, ~45GB (see Phase 6)
TRAINING_MODE=kaggle   # use when running on Kaggle (T4 GPU)
```

---

## Phase 3 — Run the data pipeline

```bash
# Subset mode: downloads 500 images automatically
python -c "from src.data.extract import prepare_data; prepare_data()"
```

Expected output:
```
Downloading NIH labels CSV...
Downloading 500-image NIH subset (~200MB)...
Subset ready: 500 images
Labels loaded: 500 images
```

---

## Phase 4 — Run the notebooks

Open VSCode → Install Jupyter extension → Select kernel `.venv`

```
notebooks/00_data_exploration.ipynb     ← understand the dataset
notebooks/01_training_walkthrough.ipynb ← train the model
notebooks/02_evaluation.ipynb           ← evaluate performance
notebooks/03_gradcam_visualisation.ipynb ← visualise predictions
```

---

## Phase 5 — Train the model

```bash
# Start training (uses TRAINING_MODE from .env)
python -m src.training.train

# Resume from last checkpoint
python -m src.training.train --resume

# Dry run — 1 batch only, verifies the pipeline
python -m src.training.train --dry-run
```

Checkpoints are saved to `checkpoints/` after every epoch.
Best model is saved to `checkpoints/best_model.pt`.

---

## Phase 6 — Full dataset on Kaggle (free T4 GPU)

1. Go to kaggle.com → New notebook
2. Add dataset: **NIH Chest X-rays** (pre-mounted at `/kaggle/input/nih-chest-xrays/`)
3. Upload `notebooks/01_training_walkthrough.ipynb`
4. Set `TRAINING_MODE=kaggle` and `TRAIN_EPOCHS=10`
5. Enable GPU: Settings → Accelerator → GPU T4 x2
6. Run all cells (~3–4 hours for 10 epochs)
7. Download `best_model.pt` from the Output tab
8. Place it in `checkpoints/best_model.pt` on your laptop

---

## Phase 7 — Full dataset locally (when you have ~45GB space)

```bash
# Download first 3 batches (~12GB) for a strong model
python scripts/download_nih.py --batches 1 2 3

# Download all 12 batches (~45GB)
python scripts/download_nih.py

# Check download status
python scripts/download_nih.py --status

# Then train in full mode
TRAINING_MODE=full python -m src.training.train
```

---

## Phase 8 — Evaluate the model

```bash
python -m src.training.evaluate
```

Produces:
- Per-class AUC-ROC and AUPRC table
- ROC curve PNG → `data/processed/roc_curves_test.png`
- PR curve PNG  → `data/processed/pr_curves_test.png`
- Results JSON  → `data/processed/eval_results.json`

---

## Phase 9 — Run the Streamlit app

```bash
streamlit run app/app.py
```

Open http://localhost:8501

Features:
- Upload any chest X-ray
- See per-pathology probability bars
- Toggle Grad-CAM heatmap overlay
- Select which pathology class to visualise

---

## Phase 10 — Run the FastAPI backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@data/sample_xrays/example.png"
```

API docs: http://localhost:8000/docs

---

## Phase 11 — Run tests

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

Expected: 63+ tests pass.

---

## Phase 12 — Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Go to share.streamlit.io → New app
3. Set main file: `app/app.py`
4. Add secret:
   ```toml
   TRAINING_MODE = "subset"
   ```
5. Upload `checkpoints/best_model.pt` (if not in repo — it's gitignored)
   - Option: host on HuggingFace Hub and download at startup
6. Click Deploy

---

## Phase 13 — Deploy to HuggingFace Spaces (Gradio)

1. Create a new Space: huggingface.co/spaces
2. Set SDK: Gradio
3. Upload `app/app_gradio.py` as `app.py`
4. Upload `requirements.txt`
5. Upload `checkpoints/best_model.pt`
6. Space builds automatically and goes live

---

## Troubleshooting

**`FileNotFoundError: best_model.pt`**
Train the model first or download a pre-trained checkpoint:
```bash
python -m src.training.train --dry-run   # verify pipeline
python -m src.training.train             # full training
```

**`ModuleNotFoundError: No module named 'src'`**
Run from the repo root:
```bash
cd chest-xray-classifier
python -m src.training.train
```

**`CUDA out of memory`**
Reduce batch size in `.env`:
```bash
TRAIN_BATCH_SIZE_GPU=16   # default is 32
```

**Subset download fails**
Download manually from Kaggle:
```
https://www.kaggle.com/datasets/nih-chest-xrays/data
```
Place 500 PNG files in `data/raw/subset/`.

**Slow training on CPU**
Expected — CPU training on 500 images takes ~20 minutes per epoch.
Use Kaggle free T4 GPU for the full model (see Phase 6 above).

---

## Quick reference

```
python -m src.training.train              # train
python -m src.training.train --resume    # resume from checkpoint
python -m src.training.evaluate          # evaluate on test set
streamlit run app/app.py                 # Streamlit demo
uvicorn src.api.main:app --reload        # FastAPI backend
python scripts/download_nih.py --status  # check data download
pytest                                   # run tests
```
