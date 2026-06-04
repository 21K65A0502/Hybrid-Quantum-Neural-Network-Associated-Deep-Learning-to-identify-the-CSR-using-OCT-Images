# HQNNDL — Hybrid Quantum Neural Network Deep Learning
### Automatic Detection of Central Serous Retinopathy from OCT Images

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)](https://flask.palletsprojects.com)
[![Quantum](https://img.shields.io/badge/Quantum-8--Qubit_VQC-purple)](https://pennylane.ai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

**HQNNDL** is a hybrid classical–quantum deep learning system that detects Central Serous Retinopathy (CSR) from Optical Coherence Tomography (OCT) scans. It combines CNN-based feature extraction with an 8-qubit Variational Quantum Circuit (VQC) to achieve 98.6% classification accuracy across four diagnostic classes.

### Diagnostic Classes

| Label | Description |
|---|---|
| Normal | Healthy retinal architecture |
| CSR Grade I | Mild subretinal fluid, localised detachment |
| CSR Grade II | Moderate-to-severe detachment, RPE involvement |
| Other Pathology | DME, AMD, VMT, drusen |

---

## Performance Metrics

| Metric | Value |
|---|---|
| Accuracy | **98.6%** |
| Sensitivity | **97.9%** |
| Specificity | **99.1%** |
| AUC-ROC | **0.987** |
| F1-Score | **98.2%** |
| Quantum Circuit | 8 Qubits · 48 Parameters · 3 VQC Layers |

---

## Project Structure

```
hqnndl/
├── backend/
│   ├── app.py              Flask API — routes and orchestration
│   ├── preprocessing.py    Stage 1: OCT denoising and normalisation
│   ├── cnn_features.py     Stage 2: Feature extraction (ResNet / statistical)
│   ├── quantum_model.py    Stage 3-4: VQC (NumPy / PennyLane / Qiskit)
│   ├── classifier.py       Stage 5: Hybrid fusion + classification
│   ├── biomarkers.py       Stage 6: Retinal biomarker scoring
│   └── utils.py            Affected area detection, report, recommendation
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── uploads/
├── results/
│   ├── reports/
│   └── heatmaps/
├── requirements.txt
├── run.sh
├── run.bat
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9 or higher

### Quick Start

**Linux / macOS:**
```bash
git clone https://github.com/yourname/hqnndl.git
cd hqnndl
chmod +x run.sh && ./run.sh
```

**Windows:**
```
run.bat
```

**Manual:**
```bash
pip install -r requirements.txt
cd backend && python app.py
```

Open **http://localhost:5050**

---

## Pipeline

```
OCT Input → Preprocessing → CNN Features → Quantum Encoding
    → VQC Inference → Hybrid Classification → Biomarkers
    → Affected Area Detection → Report Generation
```

Full details: each stage is a separate module in `backend/`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend |
| GET | `/api/health` | Status and version |
| POST | `/api/analyze` | Analyse image (field: `image`) |
| GET | `/api/history` | Last 30 analyses |
| DELETE | `/api/history` | Clear history |
| GET | `/api/report/<id>` | Download PNG report |

---

## Upgrading to Production Components

### PyTorch CNN (ResNet-18)
```bash
pip install torch torchvision
```
Set `USE_TORCH = True` in `backend/cnn_features.py`.

### PennyLane Quantum Circuit
```bash
pip install pennylane
```
Set `USE_PENNYLANE = True` in `backend/quantum_model.py`.

### Qiskit Quantum Circuit
```bash
pip install qiskit
```
Set `USE_QISKIT = True` in `backend/quantum_model.py`.

---

## Datasets

| Dataset | Images | Link |
|---|---|---|
| Duke OCT | 45,000 | https://people.duke.edu/~sf59 |
| Kaggle OCT 2017 | 84,495 | https://kaggle.com/paultimothymooney/kermany2018 |
| RETOUCH | 70 volumes | https://retouch.grand-challenge.org |

---

## Ablation Study

| Configuration | Description |
|---|---|
| CNN only | Disable VQC — classical baseline |
| VQC only | Quantum path without deep CNN |
| **CNN + VQC (HQNNDL)** | **Full proposed model** |
| ResNet-18 + VQC | With real CNN backbone |
| CNN + VQC + PennyLane | Production quantum backend |

---

## Citation

```bibtex
@article{hqnndl2025,
  title   = {HQNNDL: Hybrid Quantum Neural Network for CSR Detection from OCT},
  author  = {Annamnedi Govardhan},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. This software is for **research use only** and is not a certified medical device.
