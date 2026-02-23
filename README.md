# Refining DenseNet201 with Mixed Attention for High-Precision White Blood Cell Classification

> DenseNet201 with mixed spatial–channel attention for white blood cell classification on the Raabin-WBC dataset.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This is the official implementation of **DenseNet201 with mixed spatial–channel attention (SAM + CAM)** for white blood cell (WBC) classification on the [Raabin-WBC](https://raabindata.com/free-data/) dataset.

The model supports two classification tasks:

| Task | Classes |
|------|---------|
| **5-class** | Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil |
| **2-class** | Agranulocytes vs. Granulocytes |

> 📄 This repository accompanies a manuscript submitted to *The Visual Computer*. If you use this code, please cite the associated paper (see [Citation](#citation)).

---

## Architecture

The model combines **DenseNet201** as a backbone with:
- **SAM** — Spatial Attention Module
- **CAM** — Channel Attention Module

These two attention mechanisms are fused in a mixed strategy to improve feature discrimination across WBC morphology.

---

## Dataset Structure

Download the Raabin-WBC dataset and organize it as follows:

```
Raabin-WBC/
├── Basophil/
├── Eosinophil/
├── Lymphocyte/
├── Monocyte/
└── Neutrophil/
```

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/your-username/RaabinWBC-DenseNet201-MixedAttention.git
cd RaabinWBC-DenseNet201-MixedAttention
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py --data_dir "/path/to/Raabin-WBC"
```

Outputs are saved to:

```
outputs/
├── best_model.keras       # Best checkpoint by validation accuracy
├── training_log.csv       # Per-epoch metrics
├── splits/                # Train/val/test split indices
└── figures/               # Loss & accuracy curves
```

### Evaluation

```bash
python eval.py --data_dir "/path/to/Raabin-WBC" --model_path outputs/best_model.keras
```

This reports:
- 5-class classification metrics (accuracy, precision, recall, F1)
- 2-class (agranulocyte vs. granulocyte) metrics
- Confusion matrices for both tasks

---

## Results

*Results will be updated upon manuscript acceptance.*

---

## Citation

This repository accompanies a manuscript currently under review at *The Visual Computer*. Citation details will be provided upon publication.

If you use this code in the meantime, please acknowledge this repository:

```
Author(s). RaabinWBC-DenseNet201-MixedAttention. GitHub, 2025.
https://github.com/your-username/RaabinWBC-DenseNet201-MixedAttention
```

---

## License

This project is licensed under the [MIT License](LICENSE).
