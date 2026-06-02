from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def apply_background_correction(
    intensities: np.ndarray,
    metadata: pd.DataFrame,
    sequence_column: str = "sequence_id",
    background_column: str = "is_background_reference",
) -> np.ndarray:
    """
    Apply sequence-specific background correction.

    Parameters
    ----------
    intensities : np.ndarray
        Spectral intensity matrix with shape
        (n_spectra, n_wavelengths).
    metadata : pd.DataFrame
        Metadata table containing one row per spectrum.
    sequence_column : str, default="sequence_id"
        Column identifying the measurement sequence.
    background_column : str, default="is_background_reference"
        Boolean column identifying background reference spectra.

    Returns
    -------
    np.ndarray
        Background-corrected intensity matrix.

    Raises
    ------
    ValueError
        If a sequence contains no background reference spectra.
    """

    corrected = intensities.copy()

    for sequence_id, sequence_metadata in metadata.groupby(sequence_column):
        sequence_indices = sequence_metadata.index.to_numpy()

        background_indices = sequence_metadata.index[
            sequence_metadata[background_column]
        ].to_numpy()

        if len(background_indices) == 0:
            raise ValueError(
                f"No background reference spectra found for sequence "
                f"'{sequence_id}'."
            )

        background_reference = intensities[background_indices].mean(axis=0)

        corrected[sequence_indices] = (
            intensities[sequence_indices] - background_reference
        )

    return corrected


class SpectraNormaliser(BaseEstimator, TransformerMixin):
    """
    Spectrum-wise normalisation transformer.

    Parameters
    ----------
    method : str, default="snv"
        Normalisation method. Currently supported: "snv".
    """

    def __init__(self, method: str = "snv"):
        self.method = method

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X :
            Spectral intensity matrix.
        y : optional
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        SpectraNormaliser
            Fitted transformer.
        """

        X_array = self._to_array(X)
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X):
        """
        Transform spectral data.

        Parameters
        ----------
        X :
            Spectral intensity matrix.

        Returns
        -------
        np.ndarray
            Normalised intensity matrix.

        Raises
        ------
        ValueError
            If the number of features differs from the fitted data or
            if the normalisation method is unknown.
        """

        X_array = self._to_array(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {X_array.shape[1]}."
            )

        method = self.method.lower()

        if method == "snv":
            return self._apply_snv(X_array)

        raise ValueError(f"Unknown normalisation method: {self.method}")

    @staticmethod
    def _apply_snv(X: np.ndarray) -> np.ndarray:
        """
        Apply standard normal variate normalisation.

        Parameters
        ----------
        X : np.ndarray
            Spectral intensity matrix.

        Returns
        -------
        np.ndarray
            SNV-normalised intensity matrix.

        Raises
        ------
        ValueError
            If at least one spectrum has zero standard deviation.
        """

        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)

        if np.any(std == 0):
            raise ValueError("At least one spectrum has zero standard deviation.")

        return (X - mean) / std

    @staticmethod
    def _to_array(X) -> np.ndarray:
        """
        Convert input data to a two-dimensional float array.

        Parameters
        ----------
        X :
            Input data.

        Returns
        -------
        np.ndarray
            Two-dimensional float array.

        Raises
        ------
        ValueError
            If the input is not two-dimensional.
        """

        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_array.shape}.")

        return X_array.astype(float, copy=False)


class ArtifactInterpolator(BaseEstimator, TransformerMixin):
    """
    Local spectral artefact interpolation transformer.

    Parameters
    ----------
    start_bin : int
        First spectral bin of the artefact region.
    end_bin : int
        First spectral bin after the artefact region.
    method : str, default="pchip"
        Interpolation method passed to ``pandas.DataFrame.interpolate``.
    """

    def __init__(
        self,
        start_bin: int,
        end_bin: int,
        method: str = "pchip",
    ):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.method = method

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X :
            Spectral intensity matrix.
        y : optional
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        ArtifactInterpolator
            Fitted transformer.
        """

        X_array = self._to_array(X)
        self.n_features_in_ = X_array.shape[1]
        self._validate_bins(self.n_features_in_)
        return self

    def transform(self, X):
        """
        Interpolate the specified artefact region.

        Parameters
        ----------
        X :
            Spectral intensity matrix.

        Returns
        -------
        np.ndarray
            Intensity matrix with interpolated artefact region.

        Raises
        ------
        ValueError
            If the number of features differs from the fitted data or
            if interpolation produces NaN values.
        """

        X_array = self._to_array(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {X_array.shape[1]}."
            )

        X_df = pd.DataFrame(X_array.copy())
        X_df.iloc[:, self.start_bin:self.end_bin] = np.nan

        X_interpolated = X_df.interpolate(
            method=self.method,
            axis=1,
            limit_direction="both",
        )

        if X_interpolated.isna().any().any():
            raise ValueError(
                "Interpolation produced NaN values. Check artefact region "
                "and interpolation support."
            )

        return X_interpolated.to_numpy()

    def _validate_bins(self, n_features: int) -> None:
        """
        Validate artefact-region bin indices.

        Parameters
        ----------
        n_features : int
            Number of spectral features.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the bin interval is invalid.
        """

        if self.start_bin < 0:
            raise ValueError("start_bin must be non-negative.")

        if self.end_bin <= self.start_bin:
            raise ValueError("end_bin must be larger than start_bin.")

        if self.end_bin > n_features:
            raise ValueError(
                f"end_bin ({self.end_bin}) exceeds number of features "
                f"({n_features})."
            )

        if self.start_bin == 0 or self.end_bin == n_features:
            raise ValueError(
                "Artefact interpolation should not touch the spectral edges."
            )

    @staticmethod
    def _to_array(X) -> np.ndarray:
        """
        Convert input data to a two-dimensional float array.

        Parameters
        ----------
        X :
            Input data.

        Returns
        -------
        np.ndarray
            Two-dimensional float array.

        Raises
        ------
        ValueError
            If the input is not two-dimensional.
        """

        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_array.shape}.")

        return X_array.astype(float, copy=False)


class SavgolTransformer(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filtering transformer.

    Parameters
    ----------
    window_length : int, default=95
        Filter window length.
    polyorder : int, default=2
        Polynomial order used for filtering.
    deriv : int, default=0
        Derivative order.
    delta : float, default=1.0
        Spacing of the spectral axis.
    """

    def __init__(
        self,
        window_length: int = 95,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X :
            Spectral intensity matrix.
        y : optional
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        SavgolTransformer
            Fitted transformer.
        """

        X_array = self._to_array(X)
        self.n_features_in_ = X_array.shape[1]
        self._validate_params(self.n_features_in_)
        return self

    def transform(self, X):
        """
        Apply Savitzky-Golay filtering.

        Parameters
        ----------
        X :
            Spectral intensity matrix.

        Returns
        -------
        np.ndarray
            Filtered intensity matrix.

        Raises
        ------
        ValueError
            If the number of features differs from the fitted data.
        """

        X_array = self._to_array(X)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {X_array.shape[1]}."
            )

        return savgol_filter(
            X_array,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=1,
        )

    def _validate_params(self, n_features: int) -> None:
        """
        Validate Savitzky-Golay filter parameters.

        Parameters
        ----------
        n_features : int
            Number of spectral features.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the filter parameters are invalid.
        """

        if self.window_length <= 0:
            raise ValueError("window_length must be positive.")

        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd.")

        if self.window_length > n_features:
            raise ValueError(
                f"window_length ({self.window_length}) exceeds number of "
                f"features ({n_features})."
            )

        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be smaller than window_length.")

        if self.deriv < 0:
            raise ValueError("deriv must be non-negative.")

        if self.delta <= 0:
            raise ValueError("delta must be positive.")

    @staticmethod
    def _to_array(X) -> np.ndarray:
        """
        Convert input data to a two-dimensional float array.

        Parameters
        ----------
        X :
            Input data.

        Returns
        -------
        np.ndarray
            Two-dimensional float array.

        Raises
        ------
        ValueError
            If the input is not two-dimensional.
        """

        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_array.shape}.")

        return X_array.astype(float, copy=False)


def wavelength_step(wavelengths: np.ndarray) -> float:
    """
    Calculate the mean wavelength spacing.

    Parameters
    ----------
    wavelengths : np.ndarray
        One-dimensional wavelength axis.

    Returns
    -------
    float
        Mean spacing between adjacent wavelength values.

    Raises
    ------
    ValueError
        If the wavelength axis is not one-dimensional or contains
        fewer than two values.
    """

    wavelengths = np.asarray(wavelengths, dtype=float)

    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional.")

    if wavelengths.size < 2:
        raise ValueError("wavelengths must contain at least two values.")

    return float(np.mean(np.diff(wavelengths)))


def bins_from_wavelength_range(
    wavelengths: np.ndarray,
    wavelength_range: tuple[float, float],
) -> tuple[int, int]:
    """
    Convert a wavelength interval to positional bin indices.

    Parameters
    ----------
    wavelengths : np.ndarray
        One-dimensional wavelength axis.
    wavelength_range : tuple[float, float]
        Lower and upper wavelength limits.

    Returns
    -------
    tuple[int, int]
        Start bin and end bin following Python slicing convention.

    Raises
    ------
    ValueError
        If the wavelength axis is invalid, if the interval is invalid,
        or if the interval does not overlap with the wavelength axis.
    """

    wavelengths = np.asarray(wavelengths, dtype=float)

    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional.")

    lower, upper = wavelength_range

    if upper <= lower:
        raise ValueError("Upper wavelength limit must be larger than lower limit.")

    mask = (wavelengths >= lower) & (wavelengths <= upper)

    if not np.any(mask):
        raise ValueError("Wavelength range does not overlap with wavelength axis.")

    indices = np.flatnonzero(mask)

    return int(indices[0]), int(indices[-1] + 1)


def create_preprocessing_pipeline(
    wavelengths: np.ndarray,
    artefact_wavelength_range: tuple[float, float] | None = (548.0, 554.0),
    artefact_bin_range: tuple[int, int] | None = None,
    savgol_window_length: int = 95,
    savgol_polyorder: int = 2,
    delta: float | None = 1.0,
) -> Pipeline:
    """
    Create the spectral preprocessing pipeline.

    Parameters
    ----------
    wavelengths : np.ndarray
        One-dimensional wavelength axis.
    artefact_wavelength_range : tuple[float, float] | None, default=(548.0, 554.0)
        Wavelength interval used for artefact interpolation.
    artefact_bin_range : tuple[int, int] | None, default=None
        Positional bin interval used for artefact interpolation. If provided,
        it takes precedence over ``artefact_wavelength_range``.
    savgol_window_length : int, default=95
        Window length used for Savitzky-Golay filtering.
    savgol_polyorder : int, default=2
        Polynomial order used for Savitzky-Golay filtering.
    delta : float | None, default=1.0
        Spacing between adjacent wavelength values. 

    Returns
    -------
    Pipeline
        Preprocessing pipeline containing artefact interpolation,
        SNV normalisation, and first-derivative Savitzky-Golay filtering.

    Raises
    ------
    ValueError
        If neither an artefact wavelength range nor an artefact bin range
        is provided.
    """

    if artefact_bin_range:
        start_bin, end_bin = artefact_bin_range
    elif artefact_wavelength_range:
        start_bin, end_bin = bins_from_wavelength_range(
            wavelengths=wavelengths,
            wavelength_range=artefact_wavelength_range,
        )
    else:
        raise ValueError(
            "Either artefact_bin_range or artefact_wavelength_range "
            "must be provided."
        )

    return Pipeline(
        steps=[
            (
                "artefact_interpolation",
                ArtifactInterpolator(
                    start_bin=start_bin,
                    end_bin=end_bin,
                    method="pchip",
                ),
            ),
            (
                "snv",
                SpectraNormaliser(method="snv"),
            ),
            (
                "savgol_derivative",
                SavgolTransformer(
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                    deriv=1,
                    delta=delta,
                ),
            ),
        ]
    )