from __future__ import annotations

import numpy as np

from src.loaders import SpectralDataset


DEFAULT_THRESHOLD_FACTOR = 1.05


def calculate_total_intensity(intensities: np.ndarray) -> np.ndarray:
    """
    Calculate the total intensity of each spectrum.

    Parameters
    ----------
    intensities : np.ndarray
        Spectral intensity matrix with shape
        (n_spectra, n_wavelengths).

    Returns
    -------
    np.ndarray
        Total intensity value for each spectrum.
    """

    return np.sum(intensities, axis=1)


def select_sample_related_spectra(
    dataset: SpectralDataset,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
) -> SpectralDataset:
    """
    Select sample-related spectra from continuous measurements.

    Parameters
    ----------
    dataset : SpectralDataset
        Continuous spectral dataset.
    threshold_factor : float, default=1.05
        Multiplicative factor applied to the median total intensity
        of each measurement sequence.

    Returns
    -------
    SpectralDataset
        Dataset containing selected sample-related spectra and
        background reference spectra.

    Raises
    ------
    ValueError
        If the metadata table does not contain a ``sequence_id`` column.

    Notes
    -----
    For each measurement sequence, the total intensity is calculated
    for every spectrum. A spectrum is marked as sample-related if its
    total intensity is greater than or equal to ``threshold_factor``
    times the median total intensity of the corresponding sequence.

    The first ten spectra of each sequence are retained as background
    reference spectra independently of the intensity threshold.
    """

    if "sequence_id" not in dataset.metadata.columns:
        raise ValueError(
            "Data selection requires a continuous dataset with a "
            "'sequence_id' column in the metadata."
        )

    total_intensity = calculate_total_intensity(dataset.intensities)

    metadata = dataset.metadata.copy()
    metadata["total_intensity"] = total_intensity
    metadata["selection_threshold"] = np.nan
    metadata["selected"] = False

    metadata["spectrum_index_in_sequence"] = (
        metadata
        .groupby("sequence_id")
        .cumcount()
        .add(1)
    )

    metadata["is_background_reference"] = (
        metadata["spectrum_index_in_sequence"] <= 10
    )

    metadata["is_sample_related"] = False

    selected_mask = np.zeros(len(metadata), dtype=bool)

    for sequence_id, sequence_indices in metadata.groupby("sequence_id").groups.items():
        sequence_indices = np.array(list(sequence_indices))

        sequence_total_intensity = total_intensity[sequence_indices]
        median_total_intensity = np.median(sequence_total_intensity)
        threshold = threshold_factor * median_total_intensity

        sequence_mask = sequence_total_intensity >= threshold

        selected_mask[sequence_indices] = (
            sequence_mask
            | metadata.loc[sequence_indices, "is_background_reference"].to_numpy()
        )
        metadata.loc[sequence_indices, "selection_threshold"] = threshold
        metadata.loc[sequence_indices, "is_sample_related"] = sequence_mask
        metadata.loc[sequence_indices, "selected"] = (
            sequence_mask
            | metadata.loc[sequence_indices, "is_background_reference"].to_numpy()
        )

    selected_metadata = metadata.loc[selected_mask].reset_index(drop=True)
    selected_intensities = dataset.intensities[selected_mask]

    return SpectralDataset(
        wavelengths=dataset.wavelengths,
        intensities=selected_intensities,
        metadata=selected_metadata,
    )


def summarise_selection(
    original_dataset: SpectralDataset,
    selected_dataset: SpectralDataset,
) -> None:
    """
    Print a summary of the spectrum selection result.

    Parameters
    ----------
    original_dataset : SpectralDataset
        Dataset before spectrum selection.
    selected_dataset : SpectralDataset
        Dataset after spectrum selection.

    Returns
    -------
    None
        The summary is printed to the console.
    """

    n_original = len(original_dataset.metadata)
    n_selected = len(selected_dataset.metadata)

    print("Data selection")
    print("==============")
    print(f"Original spectra: {n_original}")
    print(f"Selected spectra: {n_selected}")
    print(f"Retained fraction: {n_selected / n_original:.3f}")

    print("\nSelected spectra per sequence:")

    summary = (
        selected_dataset.metadata
        .groupby(
            [
                "sequence_id",
                "sample_id",
            ]
        )
        .size()
        .rename("n_selected")
        .reset_index()
    )

    print(summary)