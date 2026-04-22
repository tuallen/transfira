# TransFIRA: Transfer Learning for Face Image Recognizability Assessment

**FG 2026 — Oral Presentation**

**[Project Page](https://transfira.github.io/) | [arXiv](https://arxiv.org/abs/2510.06353) | [Paper](https://arxiv.org/pdf/2510.06353) | [OneDrive](https://umd0-my.sharepoint.com/:f:/g/personal/atu1_umd_edu/IgBq51obRcrbQ6x5C8mYh3iRAd9cwfOzS86NhC5QrAL424A?e=QSKMHr)**

[Allen Tu](https://tuallen.github.io),
[Kartik Narayan](https://kartik-3004.github.io/portfolio/),
[Joshua Gleason](https://scholar.google.com/citations?user=FUchtr4AAAAJ),
[Jennifer Xu](https://scholar.google.com/citations?user=iFy2JdkAAAAJ),
[Matthew Meyn](https://www.linkedin.com/in/matthew-meyn-579784157),
[Tom Goldstein](https://www.cs.umd.edu/~tomg/),
[Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)

<img src="assets/transfira.png" alt="TransFIRA Overview" />

This repository contains the official implementation of **TransFIRA: Transfer Learning for Face Image Recognizability Assessment ([FG 2026](https://fg2026.ieee-biometrics.org/))**.

**Abstract:** *Face recognition in unconstrained environments such as surveillance, video, and web imagery must contend with extreme variation in pose, blur, illumination, and occlusion, where conventional visual quality metrics fail to predict whether inputs are truly recognizable to the deployed encoder. Existing FIQA methods typically rely on visual heuristics, curated annotations, or computationally intensive generative pipelines, leaving their predictions detached from the encoder’s decision geometry. We introduce TransFIRA (Transfer Learning for Face Image Recognizability Assessment), a lightweight and annotation-free framework that grounds recognizability directly in embedding space. TransFIRA delivers three advances: (i) a definition of recognizability via class-center similarity (CCS) and class-center angular separation (CCAS), yielding the first natural, decision-boundary–aligned criterion for filtering and weighting; (ii) a recognizability-informed aggregation strategy that achieves state-of-the-art verification accuracy on BRIAR and IJB-C while nearly doubling correlation with true recognizability, all without external labels, heuristics, or backbone-specific training; and (iii) new extensions beyond faces, including encoder-grounded explainability that reveals how degradations and subject-specific factors affect recognizability, and the first method for body recognizability assessment. Experiments confirm state-of-the-art results on faces, strong performance on body recognition, and robustness under cross-dataset shifts and out-of-distribution evaluation. Together, these contributions establish TransFIRA as a unified, geometry-driven framework for recognizability assessment  that is encoder-specific, accurate, interpretable, and extensible across modalities, significantly advancing FIQA in accuracy, explainability, and scope.*



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tuallen/transfira.git
cd transfira
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate transfira
```

This installs PyTorch with CUDA 11.8 support. For other CUDA versions or CPU-only, modify the `pytorch-cuda` line in [environment.yml](environment.yml) (or remove it for CPU-only).

### 3. Download pretrained weights

**Note on Model Availability:** IARPA BRIAR training data and models are proprietary, so we cannot release the STR SwinTransformer, STR SemReID, or InsightFace ArcFace models finetuned on BRIAR training data used in the paper at this time. We provide the InsightFace ArcFace ResNet-50 model finetuned on WebFace12M (600K identities). See the [Paper Results](#paper-results) section for more information on reproducibility.

Download the following files and place them in `pretrained_weights/`:

- **`w600k_r50.onnx`**: InsightFace ArcFace ResNet-50 backbone (WebFace600K)
  - Download from [InsightFace HuggingFace](https://huggingface.co/yolkailtd/face-swap-models/blob/main/insightface/models/buffalo_l/w600k_r50.onnx)

- **`w600k_r50_transfira_0025.pt`**: TransFIRA recognizability predictor (epoch 25)
  - Download from [TransFIRA OneDrive](https://umd0-my.sharepoint.com/:f:/g/personal/atu1_umd_edu/IgBq51obRcrbQ6x5C8mYh3iRAd9cwfOzS86NhC5QrAL424A?e=QSKMHr)

```
pretrained_weights/
├── w600k_r50.onnx              # InsightFace ResNet-50 backbone (WebFace600K)
└── w600k_r50_transfira_0025.pt # TransFIRA recognizability predictor
```

### 4. Prepare your dataset

**Note on Dataset Availability:** IARPA BRIAR Protocol 3.1 is proprietary and cannot be released. IARPA is not currently distributing the IJB-C evaluation dataset. The full WebFace12M dataset is not currently being distributed. See the [Paper Results](#paper-results) section for more information on reproducibility.

We provide training and validation annotations for a subset of WebFace12M on the [TransFIRA OneDrive](https://umd0-my.sharepoint.com/:f:/g/personal/atu1_umd_edu/IgBq51obRcrbQ6x5C8mYh3iRAd9cwfOzS86NhC5QrAL424A?e=QSKMHr):
- **Training set**: 10,000 identities (train_10k.csv)
- **Validation set**: 2,000 identities (val_2k.csv)

If you have access to the full WebFace12M dataset in MXNet RecordIO format (train.rec/train.lst/train.idx), you can extract the subset using the provided script:

```bash
cd data/WebFace12M
python extract.py --rec_dir /path/to/your/webface12m
```

This will extract only the images specified in train_10k.csv and val_2k.csv and organize them in the following structure:

```
data/
├── WebFace12M/
│   ├── train_10k.csv       # Training annotations (10K identities)
│   ├── val_2k.csv          # Validation annotations (2K identities)
│   ├── train/                  # Extracted training images (created by extract.py)
│   │   ├── 000000/             # Subject ID directories
│   │   │   ├── 00000001.jpg    # Images named by RecordIO index
│   │   │   └── ...
│   │   └── ...
```

**CSV Format Requirements:**

For **gallery/probe datasets** (generate.py):
- Required columns: `path` (relative image path), `subject_id` (identity label)
- Example: `path,subject_id`

For **training datasets** (train.py):
- Required columns: `path`, `ccs`, `ccas` (generated by generate.py)
- The labels CSV is created automatically by generate.py
- Example: `path,ccs,ccas`

For **test datasets** (test.py):
- Required columns: `path`, `template_id` (for template aggregation)
- Example: `path,template_id`



## Quick Start

```bash
# 1. Generate features and recognizability labels
python generate.py --config configs/generate.toml

# 2. Train TransFIRA predictor
torchrun --nproc_per_node=4 train.py --config configs/train.toml

# 3. Test with TransFIRA aggregation
python test.py --config configs/test.toml --generate_features --predict_scores --aggregate_features --use_filter --use_weight
```

## Overview

TransFIRA is a framework to define recognizability through encoder-specific, geometry-grounded metrics:

- **Encoder-Specific**: Predictions reflect the actual discrimination ability of your deployed model, not generic visual quality
- **Geometry-Grounded**: Labels derived directly from embedding space (CCS, CCAS) align with decision boundaries
- **Annotation-Free**: No human quality labels, external IQA supervision, or recognition-specific training required
- **Lightweight**: Adds only a linear layer; works with any pretrained backbone

## Configuration

Example TOML configuration files are available in the `configs/` directory. Key parameters:
- **`backbone_type`**: Model architecture (e.g., `'onnx'`)
- **`model_path`**: Path to pretrained backbone weights
- **`checkpoint_path`**: Path to trained TransFIRA predictor checkpoint (testing only)
- **`outdir`**: Output directory for generated features, labels, and checkpoints
- **`[data]`**: Dataset paths, annotations CSV, column names for paths and labels



## 1. Feature Generation and Recognizability Label Computation

The `generate.py` script extracts backbone features and computes geometry-derived recognizability labels directly from embedding space:

- **CCS (Class Center Similarity)**: Cosine similarity to the correct class center (Equation 4)
- **NNCCS (Nearest Nonmatch Class Center Similarity)**: Maximum similarity to impostor class centers (Equation 5)
- **CCAS (Class Center Angular Separation)**: Decision-boundary margin, computed as CCS - NNCCS (Equation 6)

No human annotations or external quality labels required.

```bash
python generate.py --config configs/generate.toml
```

**Outputs** (saved to `outdir` in config):
- **`{split}_features.npy`**: Backbone embeddings (shape: [N, feature_dim])
- **`{split}_centers.npy`**: Class centers dictionary mapping subject_id → center vector
- **`{split}_ccs.npy`**: Class Center Similarity scores per image (shape: [N])
- **`{split}_nnccs.npy`**: Nearest Nonmatch Class Center Similarity scores per image (shape: [N])
- **`{split}_labels.csv`**: Updated CSV with `ccs`, `nnccs`, and `ccas` columns appended

Where `{split}` is the dataset name (e.g., `train_probes`, `val_gallery`).

### Datasets Without Separate Galleries

For datasets without gallery-probe structure (e.g., WebFace12M), define sections without suffixes (e.g., `[train]`, `[val]`). Class centers are computed from all samples of each identity (Equation 3), and CCS/NNCCS/CCAS are generated per image.

This is useful for creating training labels from any image collection without requiring separate gallery and probe sets.

### Datasets with Separate Galleries

For datasets with separate gallery and probe sets (e.g., BRIAR), define `[train_gallery]`, `[train_probes]`, `[val_gallery]`, `[val_probes]` in your config. Each section requires: `data_dir`, `annotations`, `path_col_name`, `label_col_name`. Set `annotations = ""` for unused splits.

**Class centers** are computed only from gallery embeddings (Equation 2), then probe embeddings are compared against these fixed centers to compute recognizability labels.

### Calibration (Optional)

For domains where cosine similarities saturate near 1.0 (e.g., body recognition), sigmoid calibration can restore discriminative variation. See Section IV-E and Figure 4 in the paper.

```bash
# Train calibration on training set
python calibrate.py --csv output/generated_labels/train_probes_labels.csv \
                    --outdir output/calibration/

# Apply calibration to validation/test sets
python calibrate.py --csv output/generated_labels/val_probes_labels.csv \
                    --outdir output/calibration/val/ \
                    --model output/calibration/calibration_model.pkl
```

This produces calibrated CCS/NNCCS/CCAS scores that spread across [0,1] with restored discriminative power. Face recognition typically does not require calibration.



## 2. Training the Recognizability Prediction Network

TransFIRA extends any pretrained backbone with a lightweight regression head, implemented as a linear layer, that predicts recognizability directly from images. The backbone and head are trained end-to-end using mean squared error loss (Equation 9) against precomputed CCS/CCAS labels.

```bash
torchrun --nproc_per_node=<num_gpus> train.py --config configs/train.toml
```

**Outputs** (saved to `checkpoint_outdir` in config):
- **`{run_name}/epoch_{N}.pth`**: Model checkpoints saved every `save_per_epoch` epochs
- **`{run_name}/run_params.json`**: Configuration parameters used for training
- **`runs/{run_name}/`**: TensorBoard logs for monitoring training progress


## 3. Testing and Recognizability-Informed Template Aggregation

Generate features, predict recognizability scores, and aggregate templates using trained models. This implements the recognizability-informed aggregation strategy (Section III-C) with two complementary operations:

1. **Filtering**: Natural cutoff at CCAS > 0 removes samples closer to impostor classes than to the correct class
2. **Weighting**: CCS-based weighting (Equation 10) emphasizes compact, reliable embeddings within each template

```bash
# Generate features
python test.py --config configs/test.toml --generate_features

# Predict CCS and CCAS scores
python test.py --config configs/test.toml --predict_scores

# Aggregate with filtering and weighting
python test.py --config configs/test.toml --aggregate_features --use_filter --use_weight

# Run all steps at once
python test.py --config configs/test.toml --generate_features --predict_scores --aggregate_features --use_filter --use_weight
```

### Aggregation Flags

- **`--use_weight`**: Apply CCS-based recognizability weighting (Equation 10). Each embedding is scaled by its predicted CCS before averaging, emphasizing samples with high class-center similarity.
- **`--use_filter`**: Apply CCAS > 0 filtering. Retains only samples predicted to lie on the correct side of the decision boundary.

### Output Files

**Outputs** (saved to `outdir` in config):
- **`test_features.npy`**: Backbone embeddings for each test image (shape: [N, feature_dim])
- **`test_ccs.npy`**: Predicted Class Center Similarity scores per image (shape: [N])
- **`test_ccas.npy`**: Predicted Class Center Angular Separation scores per image (shape: [N])
- **`templates_baseline.npy`**: Uniformly averaged templates (dict: subject_id → template)
- **`templates_weight.npy`**: CCS-weighted templates (Equation 10)
- **`templates_filter.npy`**: Templates aggregated from CCAS > 0 filtered samples only
- **`templates_filter_weight.npy`**: Filtered + weighted templates (best performance)

### Paper Results

**Note on Reproducibility:** The proprietary nature of IARPA BRIAR data and models, combined with WebFace12M and IJB-C no longer being actively distributed, presents challenges for directly reproducing the paper results. The results reported in the paper were computed using evaluation protocols that enable 1e-6 FMR measurements on BRIAR Protocol 3.1 and IJB-C, as well as direct comparison to numerous FIQA methods. The [test.py](test.py) script provided in this repository demonstrates a more practical example of template aggregation for general use cases.

The raw evaluation metric files, scripts, and Jupyter notebooks used to generate the paper results will eventually be uploaded to the [TransFIRA OneDrive](https://umd0-my.sharepoint.com/:f:/g/personal/atu1_umd_edu/IgBq51obRcrbQ6x5C8mYh3iRAd9cwfOzS86NhC5QrAL424A?e=QSKMHr). In the meantime, please reach out to [Allen Tu](https://tuallen.github.io) for access to these materials.



## BibTeX

```bibtex
@article{Tu2025TransFIRA,
    author  = {Tu, Allen and Narayan, Kartik and Gleason, Joshua and Xu, Jennifer and Meyn Matthew and Goldstein, Tom and Patel, Vishal M.},
    title   = {TransFIRA: Transfer Learning for Face Image Recognizability Assessment},
    journal = {arXiv preprint arXiv:2510.06353},
    year    = {2025},
    url     = {https://transfira.github.io/}
}
```



## Acknowledgements

This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via [2022-21102100005]. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The US Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
