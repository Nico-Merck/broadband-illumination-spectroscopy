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
│   └── 03_classification_validation.ipynb            # PCA, SVC, and validation
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

## Contents

* **Data folder**

  * `continuous_spectra/`: Continuous translation-stage measurements used for the main classification workflow.

    * `c_01/` to `c_37/`: Individual continuous measurement sequences.
  * `fixed_spectra/`: Fixed-position feasibility measurements acquired during the initial validation of the setup.
    These spectra are included for completeness and transparency but are not used in the subsequent classification workflow.

    * `f_001/` to `f_504/`: Individual fixed-position measurement folders.
  * `light_source_spectrum/`: Recorded emission spectrum of the broadband illumination source.

* **Jupyter notebooks**

  * `01_data_selection.ipynb`: Selects sample-related spectra from the continuous measurement sequences.
  * `02_preprocessing.ipynb`: Applies background correction, artefact interpolation, SNV transformation, and Savitzky–Golay derivative preprocessing.
  * `03_classification_validation.ipynb`: Performs PCA-based dimensionality reduction, SVC classification, and LOGO-CV validation.

* **Python modules**

  * `src/cache.py`: Utilities for generating and loading locally cached intermediate datasets.
  * `src/classification.py`: Classification, validation, and evaluation routines.
  * `src/data_selection.py`: Functions for selecting sample-related spectra based on intensity criteria.
  * `src/loaders.py`: Data loading utilities for spectral datasets and metadata.
  * `src/plot_style.py`: Global plotting configuration.
  * `src/plotting.py`: Plotting functions used by the notebooks.
  * `src/preprocessing.py`: Spectral preprocessing functions.

* **Generated outputs**

  * `results/cache/`: Locally generated intermediate datasets. Cache contents are excluded from version control and regenerated automatically when the notebooks are executed.

* **Environment and metadata**

  * `environment.yml`: Conda environment file specifying the dependencies required to reproduce the analysis.
  * `LICENSE`: Licence information.
  * `README.md`: Project description, installation instructions, and usage guide.

---

## Data selection workflow

The notebook `01_data_selection.ipynb` identifies sample-related spectra from continuous translation-stage measurements.

For each spectrum, the total intensity is calculated as the sum over all wavelength channels. Spectra are retained if their total intensity exceeds a threshold relative to the median intensity of the corresponding measurement sequence.

The selected spectra are subsequently used as input for the preprocessing and classification workflows.

---

## Preprocessing pipeline

The notebook `02_preprocessing.ipynb` applies the preprocessing workflow to the selected spectra.

The preprocessing pipeline includes:

* background correction,
* detector artefact interpolation using PCHIP interpolation,
* Standard Normal Variate (SNV) transformation,
* Savitzky–Golay smoothing,
* first-derivative Savitzky–Golay filtering.

The resulting spectra are subsequently used for PCA-based dimensionality reduction and SVC classification.

---

## Classification workflow

The notebook `03_classification_validation.ipynb` performs dimensionality reduction, classification, and validation of the preprocessed spectra.

The repository evaluates three classification tasks:

1. Full 18-class material discrimination between all investigated classes
2. Reduced polymer-type classification
3. Binary plastic-versus-non-plastic differentiation

Classification is performed using a Support Vector Classifier (SVC) with a radial basis function (RBF) kernel.

Validation is performed using Leave-One-Group-Out cross-validation (LOGO-CV) to avoid information leakage between measurement sequences.

---

## Cache and generated files

The directory `results/cache/` is used for locally generated intermediate files and cached datasets.

Cache contents are intentionally excluded from version control and are regenerated automatically when the notebooks are executed.

---

## Citation

If you use this repository, please cite the associated publication.

Publication information and DOI will be added upon publication.

---

## License

This project is licensed under the terms of the **Apache 2.0 License**.
See the `LICENSE` file for details.
