# Meta-matching

## ❗️ Project Summary

---

1. **진행기간:** 2023.07 ~ 2023.12
2. **역할:** 주저자, 데이터 전처리, 모델 파이프라인 설계
3. **기술스택:**  **`Python`**, **`PyTorch`**, **`Pandas`**, **`NumPy`**, **`Scikit-learn`**
4. **결과 및 성과:** 
    - OHBM 논문 게재 [**[📄]**](https://drive.google.com/file/d/1d8s6HwLw67PCEArFbB57kWfpDAohtU9B/view)
    - https://github.com/hannn0403/transformer-based-meta-matching-stacking
5. **주요내용:** 본 연구는 Human Connectome Project의 750명 뇌 영상 데이터와 59개의 비영상 표현형을 활용하여, 다양한 이미지 특징 데이터를 기반으로 Basic DNN을 학습한 후, 5종 이미지 특징 데이터에서 예측한 58개의 비영상 표현형 값을 통합한 multimodal feature matrix를 transformer encoder에 입력함으로써 모달리티 간 상호관계를 self-attention 방식으로 학습하는 multimodal transformer stacking 기법을 제안하였다. 이 기법은 Train meta set과 Test meta set을 활용한 50회 반복 실험에서, 단일 이미지 특징 데이터를 이용한 advanced stacking과 stacking average 방법에 비해 평균 Pearson’s correlation 및 Coefficient of Determination(COD)이 각각 0.07, 0.12, 0.05, 0.08 씩 향상된 우수한 성능을 보이며, 다양한 모달리티의 정보를 효과적으로 융합하여 유동성 지능 예측에 기여함을 입증하였다.
---

# Transformer-based Meta-Matching Stacking

This repository contains the implementation of a multimodal transformer stacking framework for predicting fluid intelligence by integrating non-imaging phenotypes using self-attention (Meta-Matching stacking) on the Human Connectome Project (HCP) dataset.

## Features
- **Basic Predictors**: Train individual models (DNN, KRR) for each imaging modality feature set (sMRI, fMRI, dMRI).
- **Advanced Stacking**: Combine base predictor outputs across modalities to build a multimodal feature matrix.
- **Transformer Stacking**: Learn cross-modality relationships via a 9-layer Transformer encoder to predict fluid intelligence.
- **Modular Design**: Clean separation between data utilities (`functions.py`, `utils.py`), base learners (`basic_dnn.py`, `basic_krr.py`), stacking (`advanced_stacking.py`), and transformer workflow (`advanced_transformer.py`).

## Prerequisites
- Python 3.7+
- PyTorch
- NumPy, SciPy, pandas
- Scikit-learn
- tqdm

Install dependencies:
```bash
pip install torch numpy scipy pandas scikit-learn tqdm
```

## Repository Structure
```
transformer-based-meta-matching-stacking/
├── figure/                      # Final figures (PNG) for publication
│   ├── Final_figure_1.png
│   └── Final_figure_2.png
├── model/                       # Core Python modules
│   ├── basic_dnn.py             # Basic DNN predictor per modality
│   ├── basic_krr.py             # Kernel Ridge Regression predictor per modality
│   ├── advanced_stacking.py     # Script to perform multimodal stacking
│   ├── advanced_transformer.py  # Transformer-based stacking workflow
│   ├── advanced_finetuning.py   # Optional fine-tuning on small k-shot sets
│   ├── CBIG_model_pytorch.py    # Reference CBIG model wrapper
│   ├── functions.py             # Data loading & phenotype utilities
│   └── utils.py                 # Helper functions (metrics, I/O)
└── README.md                    # This overview file (detailed usage below)
```

## Data Preparation
1. **Extract Features**: Preprocess raw MRI (sMRI, fMRI, dMRI) to feature matrices (e.g., MIND, functional connectivity, TBSS). Save each modality as CSV or NumPy array.
2. **Structure**: Create a `data/` directory:
   ```
   data/
     ├── sMRI.npy      # shape: [n_subjects, n_features]
     ├── fMRI.npy      # shape: [n_subjects, n_features]
     └── dMRI.npy      # shape: [n_subjects, n_features]
   labels.csv         # columns: subject_id, fluid_intelligence, other phenotypes...
   ```
3. **Partition**: Split into meta-training and meta-testing sets (8:2 ratio). Within meta-testing, reserve k-shot and test subsets as described in the paper.

## Usage
All scripts in `model/` accept command-line arguments for input/output paths. Example workflows:

### 1. Train Base Predictors
```bash
python model/basic_dnn.py \
  --features data/sMRI.npy \
  --labels labels.csv \
  --output results/basic_dnn_sMRI

python model/basic_krr.py \
  --features data/fMRI.npy \
  --labels labels.csv \
  --output results/basic_krr_fMRI
```

### 2. Perform Advanced Stacking
Merge base predictions:
```bash
python model/advanced_stacking.py \
  --pred_dirs results/basic_dnn_sMRI,results/basic_krr_fMRI,results/basic_dnn_dMRI \
  --labels labels.csv \
  --output results/stacking
```
This generates a multimodal feature matrix for the Transformer.

### 3. Train Transformer Stacking Model
```bash
python model/advanced_transformer.py \
  --stacked_features results/stacking/features.npy \
  --labels results/stacking/labels.npy \
  --output results/transformer_stacking \
  --num_layers 9 \
  --hidden_dim 256 \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-4
```

### 4. Evaluate & Visualize
- Final prediction metrics (Pearson’s r, COD) will be saved to `results/transformer_stacking/metrics.csv`.
- Use `figure/Final_figure_1.png` and `figure/Final_figure_2.png` for publication-quality plots.
---
