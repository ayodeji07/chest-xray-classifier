# Model Card — Chest X-Ray Pathology Classifier

**Model**: DenseNet121 fine-tuned on NIH ChestX-ray14
**Version**: 1.0
**Date**: 2026-07
**Architecture**: CheXNet (Rajpurkar et al., 2017)

---

## Model Details

| Property | Value |
|---|---|
| Architecture | DenseNet121 (pre-trained on ImageNet) |
| Task | Multi-label classification (10 pathology classes) |
| Input | Chest X-ray PNG/JPEG, resized to 224×224 |
| Output | Per-class sigmoid probabilities (0.0–1.0) |
| Framework | PyTorch 2.x |
| Parameters | ~7 million |

### Pathology Classes

| Class | Clinical Description |
|---|---|
| Atelectasis | Partial or complete lung collapse |
| Cardiomegaly | Enlarged heart shadow |
| Consolidation | Lung tissue filled with fluid/infection |
| Edema | Excess fluid in lung tissue |
| Effusion | Fluid accumulation around lungs |
| Mass | Soft tissue density >3cm |
| Nodule | Rounded opacity <3cm |
| Pleural Thickening | Thickening of the pleural membrane |
| Pneumonia | Lung parenchymal infection |
| Pneumothorax | Air in the pleural space |

---

## Intended Use

### Primary Intended Use
- Research and educational demonstration of medical imaging AI
- Portfolio demonstration of transfer learning and Grad-CAM explainability
- Reference implementation for CheXNet-style chest X-ray classification

### Intended Users
- ML/AI researchers and students
- Healthcare AI developers building prototype systems
- Educators teaching medical imaging and deep learning

### Out-of-Scope Uses
- **Clinical diagnosis** — this model must not be used to diagnose, treat, or manage patients
- **Screening programs** — not validated for population-level screening
- **Emergency triage** — not validated for time-critical clinical decisions
- **Replacing radiologist review** — outputs are not a substitute for expert interpretation

---

## Training Data

**Dataset**: NIH ChestX-ray14 (full dataset: 112,120 frontal-view X-ray
images from 30,805 unique patients, single US institution)

**This deployed model** was trained on a ~25,000-image subset (5 of the
12 available batches), not the full 112,120-image set — see [Training
modes](README.md#training-modes) for the accuracy/data-volume tradeoff.
Re-running training on more batches will improve on the numbers below.

- **Source**: National Institutes of Health Clinical Center
- **Labels**: NLP-extracted from radiology reports (not verified by radiologists)
- **Split**: Random 70/10/20 train/val/test split, stratified by class
  representation (the official NIH split files were not used for this
  run — see `src/data/dataloader.py::split_dataframe`)

### Known Dataset Limitations
- Labels were extracted from radiology reports using NLP — estimated label accuracy ~90%
- No confirmation that the labelled pathology is the primary finding
- Labels are binary (present/absent) — no severity grading

### Demographic Distribution (NIH ChestX-ray14)
| Attribute | Distribution |
|---|---|
| Sex | 56.5% male, 43.5% female |
| Age range | 1–95 years (mean 46.9) |
| Ethnicity | Not reported in dataset |
| Geography | Single institution (USA) |

---

## Evaluation Results

*Measured on the held-out random test split (n=5,000 images), using the
checkpoint currently deployed in the live demo. Reproduce with
`python -m src.training.evaluate`.*

| Pathology | AUC-ROC | AUPRC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Cardiomegaly | 0.919 | 0.402 | 0.269 | 0.740 | 0.394 |
| Effusion | 0.870 | 0.443 | 0.330 | 0.719 | 0.453 |
| Edema | 0.864 | 0.117 | 0.127 | 0.532 | 0.206 |
| Mass | 0.810 | 0.240 | 0.177 | 0.518 | 0.264 |
| Pneumothorax | 0.794 | 0.210 | 0.138 | 0.567 | 0.222 |
| Atelectasis | 0.790 | 0.302 | 0.227 | 0.662 | 0.339 |
| Consolidation | 0.780 | 0.143 | 0.116 | 0.583 | 0.194 |
| Pleural Thickening | 0.774 | 0.118 | 0.112 | 0.439 | 0.178 |
| Nodule | 0.719 | 0.188 | 0.147 | 0.454 | 0.222 |
| Pneumonia | 0.666 | 0.024 | 0.043 | 0.217 | 0.071 |
| **Mean** | **0.799** | **0.219** | — | — | — |

Precision/recall/F1 are computed at the default 0.5 decision threshold.
Note the low precision across most classes — this reflects the
dataset's heavy class imbalance (most images are "No Finding") more
than a defect specific to this model; CheXNet-style models trained
this way typically show the same pattern.

### Comparison with CheXNet Benchmark
Our implementation targets the CheXNet (Rajpurkar et al., 2017)
architecture. On ~22% of the full training data (~25k of 112k images),
this run reaches a mean AUC of **0.799** vs. the original CheXNet
paper's **0.841** on the full dataset — a reasonable result given the
smaller training set, with a clear path to closing the gap by training
on more batches (see `scripts/download_nih.py`).

---

## Limitations

### Technical Limitations
1. **Single institution data**: Trained on data from one US hospital — may not generalise to X-rays from different equipment, acquisition protocols, or patient populations
2. **Frontal views only**: NIH ChestX-ray14 contains primarily PA (posterior-anterior) views; performance on AP or lateral views is untested
3. **Label noise**: NLP-extracted labels have ~10% error rate — the model inherits this noise
4. **No uncertainty quantification**: The model outputs point estimates, not calibrated confidence intervals
5. **Image quality sensitivity**: Performance degrades on low-quality or non-standard X-rays

### Clinical Limitations
1. No prospective clinical validation
2. Not tested against radiologist performance on independent cohorts
3. Cannot account for clinical context (patient history, symptoms, prior imaging)
4. Multi-label outputs may not reflect clinical priority or urgency

---

## Bias Analysis

### Known Biases
- **Sex bias**: Dataset is 56.5% male — performance may differ for female patients, particularly for conditions with sex-specific prevalence (e.g. cardiomegaly)
- **Age bias**: Dataset skews toward working-age adults — performance on paediatric or elderly patients is uncertain
- **Geographic/equipment bias**: Single US institution — X-ray equipment, acquisition parameters, and patient demographics differ globally

### Bias Mitigation Applied
- Weighted BCE loss to address class imbalance
- Stratified train/val/test splits to maintain label distribution

### Recommended Further Work
- Evaluate on diverse external datasets (CheXpert, MIMIC-CXR, UK Biobank)
- Subgroup analysis by age, sex, and imaging equipment
- Calibration analysis to understand confidence reliability

---

## Ethical Considerations

### AI Safety
- **Not a medical device**: This model has not undergone regulatory review (FDA, CE marking)
- **Human oversight required**: Any deployment in a clinical-adjacent setting requires radiologist review of all outputs
- **Fail-safe design**: The model should be used as a "second reader" only — never as a primary decision-maker

### Transparency
- Full training code is open source
- Grad-CAM visualisations allow inspection of model reasoning
- Known limitations and biases are documented here

### Potential Harms
- False negatives (missed pathologies) could lead to delayed treatment if misused clinically
- False positives could cause unnecessary patient anxiety or follow-up procedures
- Overconfidence in model outputs by non-experts could displace appropriate clinical judgment

### Recommended Safeguards
- Display this disclaimer prominently in any user-facing application
- Log all predictions for audit
- Do not display predictions without confidence scores
- Include a clear "consult a radiologist" prompt

---

## How to Use

```python
from src.models.model import load_model_for_inference
from src.data.transforms import get_inference_transform
from src.utils.image_utils import load_xray
from src.utils.config import Paths

model     = load_model_for_inference(Paths.best_model)
transform = get_inference_transform()

image  = load_xray("chest_xray.png")
tensor = transform(image)
probs  = model.predict_single(tensor)
# → {"Pneumonia": 0.73, "Effusion": 0.12, ...}
```

For Grad-CAM:

```python
from src.explainability.gradcam import GradCAM
from src.explainability.visualise import generate_gradcam_overlay

with GradCAM(model, model.get_features_layer()) as cam:
    heatmap = cam.generate(tensor, "Pneumonia")

overlay = generate_gradcam_overlay(image, heatmap)
overlay.save("gradcam_pneumonia.png")
```

---

## Citation

If you use this implementation, please cite the original CheXNet paper:

```
Rajpurkar, P., Irvin, J., Ball, R.L., et al. (2017).
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
with Deep Learning. arXiv:1711.05225
```

And the NIH ChestX-ray14 dataset:

```
Wang, X., Peng, Y., Lu, L., et al. (2017).
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.
CVPR 2017.
```

---

**⚠️ This model is for research and educational purposes only.**
**It is not a medical device and must not be used for clinical diagnosis.**
