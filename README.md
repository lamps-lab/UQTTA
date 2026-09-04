**Uncertainty Quantification in Table Structure Recognition**

You can download the model from this link: https://doi.org/10.5281/zenodo.11522093

# UQTTA: Uncertainty Quantification in Table Structure Recognition

[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FIRI62200.2024.00014-blue)](https://doi.org/10.1109/IRI62200.2024.00014)
[![Model DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11522093.svg)](https://doi.org/10.5281/zenodo.11522093)

This repository contains the research implementation, data resources, model configurations, and evaluation utilities accompanying the paper:

> **Kehinde Ajayi\*, Leizhen Zhang\*, Yi He, and Jian Wu.  
> “Uncertainty Quantification in Table Structure Recognition.”  
> 2024 IEEE International Conference on Information Reuse and Integration for Data Science (IRI), pp. 1–6, 2024.**  
> DOI: [10.1109/IRI62200.2024.00014](https://doi.org/10.1109/IRI62200.2024.00014)

\* Equal contribution.

## Overview

Table structure recognition (TSR) identifies the rows, columns, and individual cells in table images so that their contents can be processed by downstream systems such as optical character recognition and document-analysis pipelines. Although modern TSR models can detect cell locations, they commonly return predictions without a reliable indication of which cells are uncertain. This makes large-scale use difficult because domain experts may otherwise need to verify every extracted cell manually.

UQTTA implements **TTA-m**, an ensemble-based uncertainty-quantification pipeline derived from Test-Time Augmentation (TTA). The method generates multiple controlled representations of the same table, fine-tunes a TSR model on each representation, applies each model to the corresponding representation at inference time, and measures cross-model agreement among predicted cell bounding boxes. The resulting cell-level confidence scores help identify predictions that may require human review before extracted data are used downstream.

At the time of publication, to the authors’ knowledge, this was the first study to enable continuous uncertainty quantification for table structure recognition.

## TTA-m Method

For each table image, the implementation uses five related representations:

1. **NAT** — the original non-augmented table;
2. **NLT** — a table with detected lines removed;
3. **HLT** — a table with horizontal lines added;
4. **VLT** — a table with vertical lines added; and
5. **HLT + VLT** — a table with both horizontal and vertical lines added.

The workflow is:

1. Generate the four augmented representations from the original table image.
2. Fine-tune one CascadeTabNet-based TSR model for each of the five representations.
3. Apply each fine-tuned model to the matching representation at test time.
4. Match predicted cell bounding boxes across models using Intersection over Union (IoU).
5. Merge overlapping predictions and derive a confidence score from the fraction of model views that support the merged cell.

With four augmentation methods, each cell is evaluated using five model views. The agreement-based confidence levels are therefore:

```text
0.2, 0.4, 0.6, 0.8, and 1.0
```

A higher score indicates stronger agreement among the five model-view predictions. These scores are uncertainty indicators based on ensemble agreement; they should not be interpreted automatically as calibrated probabilities.

## Evaluating the Uncertainty Estimates

Directly labeled ground-truth uncertainty is generally unavailable for TSR. The paper therefore evaluates the confidence scores using two controllable surrogate gauges of recognition difficulty.

### 1. Pixel-Intensity Masking

The masking procedure makes selected table cells progressively fainter by increasing their pixel intensities. The implementation includes intensity-multiplication and full-whitening variants. If the uncertainty estimates are meaningful, increasingly degraded cells should receive lower confidence and should be detected correctly less often.

Relevant implementation:

```text
TTA/Image_mask.py
```

### 2. Cell-Complexity Quantification

A table is represented as an undirected graph:

- each node represents a table cell;
- edges represent top, bottom, left, or right adjacency between cells; and
- node degree measures the number of neighboring cells.

Cells with more complex structural relationships, including cells spanning multiple rows or columns, are generally more difficult for TSR models. The implementation computes adjacency degree and exports additional graph statistics for analysis.

Relevant implementation:

```text
TTA/structure_score.py
```

## Main Contributions

This artifact supports the paper’s three principal contributions:

1. **TTA-m**, an ensemble-based method for quantifying uncertainty in TSR predictions using both training-time and test-time table augmentations;
2. **two scalable uncertainty-evaluation strategies**, pixel-intensity masking and graph-based cell-complexity quantification; and
3. **an augmented TSR experimental resource** derived from the ICDAR 2019 table-recognition benchmark, including four controlled table-image transformations and corresponding analysis utilities.

## Results Reported in the Paper

The following final-output results are reported in Table I of the paper:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Vanilla TTA ensemble | 0.683 | 0.824 | 0.753 |
| TTA-m ensemble | 0.761 | 0.835 | 0.798 |
| TTA-tm ensemble | 0.778 | 0.831 | 0.806 |
| Original-model baseline | 0.899 | 0.659 | 0.760 |

`TTA-tm` combines TTA-m with the small-cell filtering heuristic studied in the paper.

The reported evaluation also shows that:

- the percentage of correctly predicted cells generally increases with the confidence level for TTA and TTA-m;
- TTA-m correctly predicts more than 80% of cells assigned a confidence score of `1.0`;
- confidence decreases as pixel-intensity degradation makes cells harder to recognize; and
- confidence generally decreases as cell-adjacency degree, used as a structural-complexity gauge, increases.

## Repository Structure

| Path | Purpose |
|---|---|
| `Augmentation/` | Generates the NLT, HLT, VLT, and HLT+VLT table-image variants. |
| `Augmentation/augmentation.py` | Runs the four table-line augmentation procedures over an image directory. |
| `Config/config_CascadeTabNet_table_input.py` | Full Cascade R-CNN/HRNet configuration for cell detection. |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train.py` | Training configuration for original table images. |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_nolines.py` | Training configuration for no-line images. |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_hori_lines.py` | Training configuration for horizontal-line images. |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_vert_lines.py` | Training configuration for vertical-line images. |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_both_lines.py` | Training configuration for images containing both horizontal and vertical lines. |
| `CascadaTabNet_infer_all_retrain.py` | Runs inference with the five separately fine-tuned models and writes cell predictions to JSON files. |
| `TTA/tta.py` | Matches and ensembles cell predictions across the five model views and assigns agreement-based confidence scores. |
| `TTA/Image_mask.py` | Generates pixel-intensity masking variants used to evaluate uncertainty behavior. |
| `TTA/structure_score.py` | Constructs table-cell adjacency graphs and exports structural-complexity measures. |
| `Evaluate/evaluate_structure.py` | Computes IoU-based precision, recall, and F1; groups predictions by confidence; and exports detailed results. |
| `Table_Structure_Recognition/` | Utilities for bordered and borderless table-structure post-processing. |
| `Data/` | Table images, annotations, example inference outputs, and the data organization used by the artifact. |
| `mmdetection/` | MMDetection-based training and inference implementation used by the project. |
| `requirements.txt` | Version-pinned dependency snapshot for the original experimental environment. |

## Dataset

The experiments reported in the paper use **543 real-world table images** selected from the ICDAR 2019 table-recognition benchmark:

- **443 images** for training; and
- **100 images** for testing.

Only the table-structure-recognition annotations are used. The project code expects COCO-style cell annotations and directories containing original and augmented table images. Researchers reproducing the experiments should obtain the underlying benchmark under its applicable terms and update the local data paths described below.

## Pretrained Model

The archived model resource is available through Zenodo:

**[Download the model from Zenodo](https://doi.org/10.5281/zenodo.11522093)**

After downloading the checkpoint resource, update the relevant `checkpoint_file` and `load_from` entries in the configuration and inference scripts.

## Environment

The original dependency snapshot is provided in `requirements.txt`. Core dependencies include:

- PyTorch and torchvision;
- MMDetection and MMCV/MMCV-Full;
- OpenCV;
- NumPy, pandas, and SciPy;
- NetworkX;
- pycocotools; and
- Matplotlib and OpenPyXL.

A basic setup is:

```bash
git clone https://github.com/lamps-lab/UQTTA.git
cd UQTTA

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

> **Compatibility note:** `mmcv-full` must match the installed PyTorch and CUDA versions. The repository records the versions used by the original project, but a compatible MMCV build may need to be installed separately for the target machine.

## Required Local Configuration

This repository preserves the original research implementation. Several scripts retain experiment-specific absolute paths and GPU selections. Replace them with paths valid on the local machine before running the pipeline.

| File or directory | Values to update |
|---|---|
| `Config/config_CascadeTabNet_table_input.py` | `ann_file`, `img_prefix`, `load_from`, and `work_dir` |
| `mmdetection/Config_Customer/config_CascadeTabNet_2_2_train*.py` | training/validation/test annotation paths, image prefixes, and pretrained-checkpoint path |
| `mmdetection/tools/train.py` | `CUDA_VISIBLE_DEVICES` and, optionally, the default configuration path |
| `Augmentation/augmentation.py` | `c_root` and the source image directory |
| `CascadaTabNet_infer_all_retrain.py` | `CUDA_VISIBLE_DEVICES`, `c_root`, five configuration paths, and five checkpoint paths |
| `TTA/tta.py` | `c_root` and prediction/output directory names |
| `TTA/Image_mask.py` | table-image, annotation, ground-truth, and output paths |
| `TTA/structure_score.py` | ground-truth cell directory and output root |
| `Evaluate/evaluate_structure.py` | prediction, ground-truth, structure-score, image, and spreadsheet-output paths |

The two top-level CascadeTabNet scripts currently use imports beginning with:

```python
from CascadeTabNet.Table_Structure_Recognition ...
```

If the cloned repository is not placed in a Python package named `CascadeTabNet`, either update those imports to reference the local `Table_Structure_Recognition` directory or adjust `PYTHONPATH`/the repository layout accordingly.

## Reproduction Workflow

### Step 1: Prepare the Original Data

Organize the original training and test images and their COCO-style cell annotations. The configuration files expect separate image roots for training and testing.

A representative layout is:

```text
<data-root>/
├── annotations/
│   ├── train.json
│   └── test.json
└── Cell_images/
    ├── output_image_only_cell_train/
    └── output_image_only_cell_test/
```

### Step 2: Generate the Four Augmented Views

Set `c_root` and the source directory in `Augmentation/augmentation.py`. Create the destination directories if they do not already exist, then run the script once for the training set and once for the test set:

```bash
python Augmentation/augmentation.py
```

The expected image directories are:

```text
output_image_only_cell_{train|test}_img_nolines/
output_image_only_cell_{train|test}_img_horizontal_lines/
output_image_only_cell_{train|test}_img_vertical_lines/
output_image_only_cell_{train|test}_img_both_lines/
```

### Step 3: Fine-Tune the Five TSR Models

After updating the data and checkpoint paths in each configuration, train one model for each representation:

```bash
python mmdetection/tools/train.py \
  --config mmdetection/Config_Customer/config_CascadeTabNet_2_2_train.py \
  --work-dir work_dirs/original

python mmdetection/tools/train.py \
  --config mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_nolines.py \
  --work-dir work_dirs/nolines

python mmdetection/tools/train.py \
  --config mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_hori_lines.py \
  --work-dir work_dirs/horizontal_lines

python mmdetection/tools/train.py \
  --config mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_vert_lines.py \
  --work-dir work_dirs/vertical_lines

python mmdetection/tools/train.py \
  --config mmdetection/Config_Customer/config_CascadeTabNet_2_2_train_both_lines.py \
  --work-dir work_dirs/both_lines
```

The original configurations train for 30 epochs. Pretrained checkpoints from the Zenodo resource may be used instead of retraining when applicable.

### Step 4: Generate Predictions from All Five Models

Update the five configuration/checkpoint mappings and image roots in `CascadaTabNet_infer_all_retrain.py`, then run:

```bash
python CascadaTabNet_infer_all_retrain.py
```

The script writes per-image cell predictions as JSON files for the original and four augmented representations.

### Step 5: Compute TTA-m Confidence Scores

Update `c_root` in `TTA/tta.py`, then run:

```bash
python TTA/tta.py
```

The script matches predicted cells across the five model views using IoU and produces merged predictions with cell-level confidence scores.

A typical output entry has the form:

```json
{
  "bbox": [x_min, y_min, x_max, y_max, confidence]
}
```

### Step 6: Compute Cell-Complexity Measures

Set the ground-truth cell and output paths in `TTA/structure_score.py`, then run:

```bash
python TTA/structure_score.py
```

The script builds cell-adjacency graphs and exports degree and other graph-based statistics to spreadsheet files.

### Step 7: Evaluate Cell Detection and Confidence Behavior

Update the paths in `Evaluate/evaluate_structure.py`, then run:

```bash
python Evaluate/evaluate_structure.py
```

The evaluation code:

- matches predicted and ground-truth cells by IoU;
- evaluates thresholds from `0.5` through `0.9`;
- computes precision, recall, and F1;
- separates correct, incorrect, and missed cells;
- groups predictions into confidence intervals; and
- exports summary and cell-level analyses to Excel files.

### Step 8: Reproduce the Masking Analysis

Generate progressively degraded cell images with:

```bash
python TTA/Image_mask.py
```

Then point the inference and ensemble scripts to the generated masking directories and repeat Steps 4, 5, and 7. The implemented masking resources include doubled pixel intensity, tripled pixel intensity, and fully whitened-cell variants.

## Expected Outputs

Depending on the selected workflow, the artifact produces:

- augmented table-image directories;
- fine-tuned model checkpoints;
- per-model JSON cell predictions;
- TTA-m JSON predictions containing agreement-based confidence scores;
- visualized cell detections and prediction errors;
- graph-based cell-complexity spreadsheets; and
- Excel summaries of precision, recall, F1, confidence, correctness, and structural complexity.

## Scope and Limitations

- The paper does not use directly annotated ground-truth uncertainty; masking and cell complexity are controlled surrogate gauges.
- The four table-line transformations do not cover every possible source of variation in real-world documents.
- TTA-m requires multiple fine-tuned models and therefore has greater training cost than vanilla TTA.
- The released code is a research snapshot with experiment-specific paths rather than a packaged command-line application.
- The confidence score measures cross-model agreement and is not, by itself, a formal probability-calibration guarantee.

## Citation

Please cite the paper when using this repository:

```bibtex
@inproceedings{ajayi2024uncertainty,
  title     = {Uncertainty Quantification in Table Structure Recognition},
  author    = {Ajayi, Kehinde and Zhang, Leizhen and He, Yi and Wu, Jian},
  booktitle = {2024 IEEE International Conference on Information Reuse and Integration for Data Science (IRI)},
  pages     = {1--6},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/IRI62200.2024.00014}
}
```

## Acknowledgements

This work was supported in part by the U.S. National Science Foundation under Grant Nos. **IIS-2245946** and **IIS-2236578**, the Commonwealth Cyber Initiative, and the Research Institute of Digital Innovation in Learning.

## Questions

For questions about the artifact, please open a GitHub issue.
