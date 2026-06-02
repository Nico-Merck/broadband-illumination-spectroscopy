from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_CSV = PROJECT_ROOT / "data" / "samples.csv"

CONTINUOUS_DIR = PROJECT_ROOT / "data" / "continuous_spectra"
FIXED_DIR = PROJECT_ROOT / "data" / "fixed_spectra"
LIGHT_SOURCE_DIR = PROJECT_ROOT / "data" / "lightsource_spectrum"

CONTINUOUS_CSV = CONTINUOUS_DIR / "continuous_measurements.csv"
FIXED_CSV = FIXED_DIR / "fixed_measurements.csv"

VALID_SPECTRUM_PREFIX = "MAYP1114611"
VALID_SPECTRUM_SUFFIX = ".txt"


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration of a spectral dataset.

    Attributes
    ----------
    name : str
        Name of the dataset.
    data_dir : Path
        Directory containing the spectrum files.
    metadata_csv : Path
        Path to the metadata CSV file.
    id_column : str
        Column containing the measurement identifiers.
    """

    name: str
    data_dir: Path
    metadata_csv: Path
    id_column: str


@dataclass(frozen=True)
class SpectralDataset:
    """
    Spectral dataset with a shared wavelength axis.

    Attributes
    ----------
    wavelengths : np.ndarray
        Shared wavelength axis.
    intensities : np.ndarray
        Spectral intensity matrix with shape
        (n_spectra, n_wavelengths).
    metadata : pd.DataFrame
        Metadata table with one row per spectrum.
    """

    wavelengths: np.ndarray
    intensities: np.ndarray
    metadata: pd.DataFrame


CONTINUOUS_DATASET = DatasetConfig(
    name="continuous_spectra",
    data_dir=CONTINUOUS_DIR,
    metadata_csv=CONTINUOUS_CSV,
    id_column="sequence_id",
)

FIXED_DATASET = DatasetConfig(
    name="fixed_spectra",
    data_dir=FIXED_DIR,
    metadata_csv=FIXED_CSV,
    id_column="measurement_id",
)

LIGHT_SOURCE_DATASET = DatasetConfig(
    name="light_source_spectrum",
    data_dir=LIGHT_SOURCE_DIR,
    metadata_csv=LIGHT_SOURCE_DIR / "light_source_measurements.csv",
    id_column="measurement_id",
)


def load_samples() -> pd.DataFrame:
    """
    Load the sample catalogue.

    Returns
    -------
    pd.DataFrame
        Sample metadata table.
    """

    return pd.read_csv(SAMPLES_CSV)


def load_metadata(dataset: DatasetConfig) -> pd.DataFrame:
    """
    Load the metadata table of a spectral dataset.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset configuration.

    Returns
    -------
    pd.DataFrame
        Measurement metadata table.

    Raises
    ------
    ValueError
        If the required measurement identifier column is missing.
    """

    metadata = pd.read_csv(dataset.metadata_csv)

    if dataset.id_column not in metadata.columns:
        raise ValueError(
            f"Missing required column: {dataset.id_column}"
        )

    return metadata


def get_measurement_folder(
    measurement_id: str,
    dataset: DatasetConfig,
) -> Path:
    """
    Return the folder of a measurement.

    Parameters
    ----------
    measurement_id : str
        Measurement identifier.
    dataset : DatasetConfig
        Dataset configuration.

    Returns
    -------
    Path
        Measurement folder.
    """

    return dataset.data_dir / measurement_id


def spectrum_sort_key(path: Path) -> int:
    """
    Return the numeric acquisition index of a spectrum file.

    Parameters
    ----------
    path : Path
        Spectrum file path.

    Returns
    -------
    int
        Numeric acquisition index extracted from the file name.
    """

    return int(path.stem.split("__")[-1])


def list_spectrum_files(folder: Path) -> list[Path]:
    """
    List valid spectrum files in acquisition order.

    Parameters
    ----------
    folder : Path
        Folder containing spectrum files.

    Returns
    -------
    list[Path]
        Sorted list of valid spectrum files.
    """

    return sorted(
        (
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.name.startswith(VALID_SPECTRUM_PREFIX)
            and file.suffix.lower() == VALID_SPECTRUM_SUFFIX
        ),
        key=spectrum_sort_key,
    )


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a spectrum from a text file.

    Parameters
    ----------
    path : Path
        Spectrum file path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Wavelength values and intensity values.

    Raises
    ------
    ValueError
        If no valid spectral data can be extracted.
    """

    spectrum = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["wavelength", "intensity"],
        decimal=",",
    )

    spectrum = spectrum.dropna()

    spectrum["wavelength"] = pd.to_numeric(
        spectrum["wavelength"],
        errors="coerce",
    )
    spectrum["intensity"] = pd.to_numeric(
        spectrum["intensity"],
        errors="coerce",
    )

    spectrum = spectrum.dropna()

    if spectrum.empty:
        raise ValueError(f"No valid spectral data found in file: {path}")

    wavelengths = spectrum["wavelength"].to_numpy(dtype=float)
    intensities = spectrum["intensity"].to_numpy(dtype=float)

    return wavelengths, intensities


def load_measurement_spectra(
    measurement_id: str,
    dataset: DatasetConfig,
    reference_wavelengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load all spectra belonging to one measurement.

    Parameters
    ----------
    measurement_id : str
        Measurement identifier.
    dataset : DatasetConfig
        Dataset configuration.
    reference_wavelengths : np.ndarray | None, default=None
        Reference wavelength axis used for consistency checking.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Shared wavelength axis, intensity matrix, and spectrum metadata.

    Raises
    ------
    ValueError
        If no spectra are found or if wavelength axes are inconsistent.
    """

    folder = get_measurement_folder(measurement_id, dataset)
    spectrum_files = list_spectrum_files(folder)

    intensity_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []

    wavelengths_out = reference_wavelengths

    for spectrum_file in spectrum_files:
        wavelengths, intensities = load_spectrum(spectrum_file)

        if wavelengths_out is None:
            wavelengths_out = wavelengths

        else:
            if len(wavelengths) != len(wavelengths_out):
                raise ValueError(
                    f"Wavelength axis length mismatch in {spectrum_file}. "
                    f"Expected {len(wavelengths_out)}, got {len(wavelengths)}."
                )

            if not np.allclose(wavelengths, wavelengths_out):
                raise ValueError(
                    f"Wavelength axis mismatch in file: {spectrum_file}"
                )

        intensity_rows.append(intensities)

        spectrum_number = len(intensity_rows)

        metadata_rows.append(
            {
                dataset.id_column: measurement_id,
                "spectrum_index_in_sequence": spectrum_number,
                "spectrum_id": f"{measurement_id}-{spectrum_number:03d}",
                "spectrum_file": spectrum_file.name,
            }
        )

    if wavelengths_out is None:
        raise ValueError(
            f"No valid spectrum files found in folder: {folder}"
        )

    intensities_matrix = np.vstack(intensity_rows)
    spectrum_metadata = pd.DataFrame(metadata_rows)

    return wavelengths_out, intensities_matrix, spectrum_metadata


def load_dataset_spectra(dataset: DatasetConfig) -> SpectralDataset:
    """
    Load all spectra belonging to a spectral dataset.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset configuration.

    Returns
    -------
    SpectralDataset
        Spectral dataset containing wavelengths, intensities,
        and merged metadata.

    Raises
    ------
    ValueError
        If no spectra are found for the dataset.
    """

    measurement_metadata = load_metadata(dataset)

    reference_wavelengths: np.ndarray | None = None
    intensity_blocks: list[np.ndarray] = []
    spectrum_metadata_blocks: list[pd.DataFrame] = []

    for measurement_id in measurement_metadata[dataset.id_column]:
        wavelengths, intensities, spectrum_metadata = load_measurement_spectra(
            measurement_id=str(measurement_id),
            dataset=dataset,
            reference_wavelengths=reference_wavelengths,
        )

        if reference_wavelengths is None:
            reference_wavelengths = wavelengths

        intensity_blocks.append(intensities)
        spectrum_metadata_blocks.append(spectrum_metadata)

    if reference_wavelengths is None:
        raise ValueError(f"No spectra found for dataset: {dataset.name}")

    intensity_matrix = np.vstack(intensity_blocks)

    spectrum_metadata = pd.concat(
        spectrum_metadata_blocks,
        ignore_index=True,
    )

    merged_metadata = spectrum_metadata.merge(
        measurement_metadata,
        on=dataset.id_column,
        how="left",
    )

    return SpectralDataset(
        wavelengths=reference_wavelengths,
        intensities=intensity_matrix,
        metadata=merged_metadata,
    )


def load_continuous_spectra() -> SpectralDataset:
    """
    Load the continuous spectral dataset.

    Returns
    -------
    SpectralDataset
        Continuous spectral dataset.
    """

    return load_dataset_spectra(CONTINUOUS_DATASET)


def load_fixed_spectra() -> SpectralDataset:
    """
    Load the fixed-position spectral dataset.

    Returns
    -------
    SpectralDataset
        Fixed-position spectral dataset.
    """

    return load_dataset_spectra(FIXED_DATASET)


def load_all_spectra() -> SpectralDataset:
    """
    Load continuous and fixed spectra into one combined dataset.

    Returns
    -------
    SpectralDataset
        Combined spectral dataset.

    Raises
    ------
    ValueError
        If the continuous and fixed datasets use different wavelength axes.
    """

    continuous = load_continuous_spectra()
    fixed = load_fixed_spectra()

    if len(continuous.wavelengths) != len(fixed.wavelengths):
        raise ValueError(
            "Continuous and fixed datasets have different wavelength axis lengths."
        )

    if not np.allclose(continuous.wavelengths, fixed.wavelengths):
        raise ValueError(
            "Continuous and fixed datasets have different wavelength axes."
        )

    intensities = np.vstack(
        [
            continuous.intensities,
            fixed.intensities,
        ]
    )

    metadata = pd.concat(
        [
            continuous.metadata,
            fixed.metadata,
        ],
        ignore_index=True,
    )

    return SpectralDataset(
        wavelengths=continuous.wavelengths,
        intensities=intensities,
        metadata=metadata,
    )


def load_mean_light_source_spectrum() -> SpectralDataset:
    """
    Load light-source spectra and calculate their mean spectrum.

    Returns
    -------
    SpectralDataset
        Dataset containing one mean light-source spectrum.

    Raises
    ------
    ValueError
        If no light-source spectra are found or if wavelength axes
        are inconsistent.
    """

    spectrum_files = list_spectrum_files(LIGHT_SOURCE_DIR)

    wavelengths_out = None
    intensity_rows = []

    for spectrum_file in spectrum_files:
        wavelengths, intensities = load_spectrum(spectrum_file)

        if wavelengths_out is None:
            wavelengths_out = wavelengths

        elif not np.allclose(wavelengths, wavelengths_out):
            raise ValueError(
                f"Inconsistent wavelength axis in {spectrum_file}"
            )

        intensity_rows.append(intensities)

    if not intensity_rows:
        raise ValueError("No light-source spectra found.")

    mean_spectrum = np.mean(
        np.vstack(intensity_rows),
        axis=0,
        keepdims=True,
    )

    metadata = pd.DataFrame(
        [
            {
                "spectrum_id": "light_source_mean",
                "n_averaged_spectra": len(intensity_rows),
            }
        ]
    )

    return SpectralDataset(
        wavelengths=wavelengths_out,
        intensities=mean_spectrum,
        metadata=metadata,
    )