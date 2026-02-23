# RaabinWBC-DenseNet201-MixedAttention

Official implementation of **DenseNet201 with mixed spatial–channel attention (SAM + CAM)** for white blood cell classification on the **Raabin-WBC** dataset.

This model performs:
- **5-class classification**: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil  
- **2-class classification**: Agranulocytes vs Granulocytes  

📌 This repository accompanies the manuscript submitted to *The Visual Computer*.  
If you use this code, please cite the associated manuscript.

---

## Dataset Structure
Raabin-WBC/
Basophil/
Eosinophil/
Lymphocyte/
Monocyte/
Neutrophil/

---

## Installation

```bash
pip install -r requirements.txt

## Training

python train.py --data_dir "/path/to/Raabin-WBC"

Outputs are saved in:

outputs/
  ├── best_model.keras
  ├── training_log.csv
  ├── splits/
  └── figures/
