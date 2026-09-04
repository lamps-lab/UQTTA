**Uncertainty Quantification in Table Structure Recognition**

You can download the model from this link: https://doi.org/10.5281/zenodo.11522093

## Overview

This repository provides the implementation and experimental resources for
uncertainty quantification in table structure recognition. The project evaluates
the reliability of table-structure-recognition predictions and develops methods
for identifying uncertain or potentially unreliable outputs.

The implementation accompanies our paper:
"Uncertainty Quantification in Table Structure Recognition."

## Main Functionality

The repository includes implementations for:

- table structure recognition;
- test-time augmentation (TTA);
- uncertainty estimation;
- prediction evaluation;
- model inference and validation;
- experimental datasets and configuration files.

## Motivation

Table structure recognition systems may produce incorrect structural predictions,
particularly for complex or noisy documents. Uncertainty quantification helps
identify predictions that may require additional review before they are used in
downstream document-processing or decision systems.

## Repository Structure

- `TTA/` – test-time augmentation methods
- `Evaluate/` – evaluation utilities
- `Table_Structure_Recognition/` – table recognition implementation
- `Augmentation/` – data augmentation
- `Config/` – experiment configuration

