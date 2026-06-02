# Broadband Illumination Spectroscopy

Repository accompanying the manuscript:

> **Broadband illumination spectroscopy for plastic classification: a step towards real-time microplastic monitoring**

This repository contains the complete analysis workflow, source code, and spectral datasets used for the classification of plastic and non-plastic materials using broadband UV–visible illumination spectroscopy.

The repository is intended as supplementary material and a reproducibility resource for the associated publication.

---

## Repository overview

The workflow comprises:

1. Selection of sample-related spectra from continuous measurements
2. Spectral preprocessing
3. Dimensionality reduction using Principal Component Analysis (PCA)
4. Classification using Support Vector Classification (SVC)
5. Validation using Leave-One-Group-Out cross-validation (LOGO-CV)

The analysis is implemented as a sequential notebook workflow supported by reusable Python modules located in `src/`.

---

## Requirements

To reproduce the analysis workflow locally, the following software is required:

* Python 3.12
* Conda
* Jupyter Notebook or JupyterLab

All required Python dependencies are specified in `environment.yml`.

The repository was developed and tested using the Conda environment defined in this file.

---

## Usage

### Clone the repository

```bash
git clone https://github.com/Nico-Merck/broadband-illumination-spectroscopy.git
cd broadband-illumination-spectroscopy
```

### Create the Conda environment

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate bis_venv
```

### Launch Jupyter

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

### Run the analysis workflow

Execute the notebooks sequentially:

1. `01_data_selection.ipynb`
2. `02_preprocessing.ipynb`
3. `03_classification_validation.ipynb`

The notebooks are designed to run consecutively without the need for manual modification.

Intermediate datasets and cached results are generated automatically during execution.

---

## Repository structure

```
.
├── data/                                             # Spectral datasets and light source spectrum
│   ├── continuous_spectra/                           # Continuous translation-stage measurements
│   │   ├── c_01/
│   │   ├── ...
│   │   └── c_37/
│   │
│   ├── fixed_spectra/                                # Fixed-position feasibility measurements
│   │   ├── f_001/
│   │   ├── ...
│   │   └── f_504/
│   │
│   └── light_source_spectrum/                        # Recorded broadband light source spectrum
│
├── notebooks/                                        # Sequential analysis workflow notebooks
│   ├── 01_data_selection.ipynb                       # Selection of sample-related spectra
│   ├── 02_preprocessing.ipynb                        # Spectral preprocessing workflow
│   └── 03_classification_validation.ipynb            # PCA, SVC classification, and validation
│
├── results/                                          # Generated local analysis outputs
│   └── cache/                                        # Automatically generated intermediate datasets
│
├── src/                                              # Reusable analysis modules
│   ├── cache.py                                      # Local dataset caching utilities
│   ├── classification.py                             # Classification and validation routines
│   ├── data_selection.py                             # Spectrum selection functions
│   ├── loaders.py                                    # Data loading utilities
│   ├── plot_style.py                                 # Global plotting configuration
│   ├── plotting.py                                   # Plotting functions
│   └── preprocessing.py                              # Spectral preprocessing functions
│
├── environment.yml                                   # Conda environment specification
├── LICENSE                                           # Repository license
└── README.md                                         # Repository documentation
```

---

## Data organisation

The repository contains both fixed-position and continuous spectral measurements.

### `data/fixed_spectra/`

Contains spectra acquired during the initial feasibility measurements at fixed sample positions.

### `data/continuous_spectra/`

Contains continuous translation-stage measurements used for the classification workflow described in the manuscript.

### `data/light_source_spectrum/`

Contains the recorded emission spectrum of the broadband illumination source.

---

## Preprocessing pipeline

The preprocessing workflow includes:

* background correction,
* detector artefact interpolation using PCHIP interpolation,
* Standard Normal Variate (SNV) normalisation,
* Savitzky–Golay smoothing,
* first-derivative Savitzky–Golay filtering.

The resulting spectra are subsequently used for PCA-based dimensionality reduction and SVC classification.

---

## Classification workflow

The repository evaluates three classification tasks:

1. Full material discrimination between all investigated classes
2. Reduced polymer-family-level classification
3. Binary plastic-versus-non-plastic classification

Classification is performed using a Support Vector Classifier (SVC) with a radial basis function (RBF) kernel.

Validation is performed using Leave-One-Group-Out cross-validation (LOGO-CV) to avoid information leakage between measurement sequences.

---

## Cache and generated files

The directory `results/cache/` is used for locally generated intermediate files and cached datasets.

Cache contents are intentionally excluded from version control and are regenerated automatically when the notebooks are executed.

---

## Reproducibility

The repository is designed as a reproducibility resource for the associated publication.

All preprocessing, classification, and evaluation steps used in the manuscript are included in this repository.

---

## Citation

If you use this repository, please cite the associated publication.

Publication information and DOI will be added upon publication.

---

## License

This project is licensed under the terms of the **Apache 2.0 License**.
See the `LICENSE` file for details.
