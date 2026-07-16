# Chest X-Ray Pathology Classifier

**Computer Vision · Deep Learning · Medical Imaging · Grad-CAM Explainability**

Multi-label chest X-ray pathology classification using DenseNet121 (CheXNet architecture),
trained on NIH ChestX-ray14 with Grad-CAM explainability and a live Streamlit demo.

[![CI](https://github.com/ayodeji07/chest-xray-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ayodeji07/chest-xray-classifier/actions)

---

## What it does

Upload a posterior-anterior chest X-ray → receive:
- Per-pathology probabilities for 10 clinical conditions
- Grad-CAM heatmap showing which image regions influenced each prediction
- Severity indicators per finding

---

## Live demo

🔗 [chest-xray-classifier-app.streamlit.app](https://chest-xray-classifier-app.streamlit.app/)

The model checkpoint is hosted on [HuggingFace Hub](https://huggingface.co/ayodeji21/chest-xray-classifier)
and downloaded automatically on first run — see `app/app.py`.

---

## Quick start

```bash
git clone https://github.com/ayodeji07/chest-xray-classifier
cd chest-xray-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# Download 500-image dev subset (~200MB, automatic)
python -c "from src.data.extract import prepare_data; prepare_data()"

# Train (subset mode — ~20 min on CPU, or use Kaggle T4)
python -m src.training.train

# Run the app
streamlit run app/app.py
```

Full step-by-step: **[VSCODE_GUIDE.md](VSCODE_GUIDE.md)**

---

## 10 Pathology Classes

| Class | CheXNet AUC | Clinical significance |
|---|---|---|
| Cardiomegaly | 0.925 | Heart failure indicator |
| Pneumothorax | 0.889 | Critical — collapsed lung |
| Edema | 0.888 | Heart failure / fluid |
| Effusion | 0.864 | Fluid around lungs |
| Mass | 0.868 | Potential tumour |
| Pleural Thickening | 0.806 | Chronic disease |
| Atelectasis | 0.809 | Lung collapse |
| Consolidation | 0.790 | Pneumonia / infection |
| Nodule | 0.780 | Potential early cancer |
| Pneumonia | 0.768 | Acute infection |

---

## Tech stack

| Component | Library |
|---|---|
| Model backbone | DenseNet121 / EfficientNetV2-S (swappable) |
| Framework | PyTorch |
| Explainability | Grad-CAM (custom implementation) |
| Evaluation | sklearn (AUC-ROC, AUPRC, F1) |
| API | FastAPI |
| Demo | Streamlit + Gradio (HuggingFace Spaces) |
| Tests | pytest (63+ tests) |
| CI | GitHub Actions |

---

## Training modes

| Mode | Dataset | Storage | Device | AUC |
|---|---|---|---|---|
| `subset` | 500 images (auto-download) | ~200MB | CPU | ~0.70 |
| `full` (3 batches) | ~27k images | ~12GB | GPU | ~0.79 |
| `full` (all) | 112k images | ~45GB | GPU | ~0.83+ |
| `kaggle` | Pre-mounted NIH | 0MB local | T4 GPU | ~0.83+ |

Set `TRAINING_MODE` in `.env`. Full resume/checkpoint support — training can be interrupted and resumed.

---

## Model card

See **[MODEL_CARD.md](MODEL_CARD.md)** for:
- Intended use and out-of-scope uses
- Training data description and known biases
- Evaluation results vs CheXNet benchmark
- Ethical considerations and safeguards

> ⚠️ **Research use only.** Not a medical device.
> Do not use for clinical diagnosis or patient management.

---

## Project structure

```
src/
  utils/         config, logger, image utilities
  data/          dataset, transforms, dataloader, extract
  models/        model, losses, metrics
  training/      train, evaluate
  explainability/ gradcam, visualise
  api/           FastAPI app + routes
app/
  app.py         Streamlit demo
  app_gradio.py  HuggingFace Spaces version
notebooks/       00-03 walkthrough notebooks
tests/           pytest suite (63+ tests)
scripts/         download_nih.py
checkpoints/     model weights saved here
MODEL_CARD.md    Full model documentation
VSCODE_GUIDE.md  Setup and deployment guide
```

---

## Citation

```
Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection on
Chest X-Rays with Deep Learning. arXiv:1711.05225 (2017)

Wang et al., ChestX-ray8: Hospital-scale Chest X-ray Database and
Benchmarks. CVPR 2017.
```
