from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.loaders import SpectralDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "results" / "cache"


def save_spectral_dataset(
    dataset: SpectralDataset,
    name: str,
) -> None:
    """
    Save a spectral dataset to the cache directory.

    Parameters
    ----------
    dataset : SpectralDataset
        Spectral dataset to save.
    name : str
        Base name used for the cached array and metadata files.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        CACHE_DIR / f"{name}.npz",
        wavelengths=dataset.wavelengths,
        intensities=dataset.intensities,
    )

    dataset.metadata.to_csv(
        CACHE_DIR / f"{name}_metadata.csv",
        index=False,
    )


def load_spectral_dataset_cache(name: str) -> SpectralDataset:
    """
    Load a spectral dataset from the cache directory.

    Parameters
    ----------
    name : str
        Base name of the cached array and metadata files.

    Returns
    -------
    SpectralDataset
        Cached spectral dataset.
    """

    if not cache_exists(name):
        raise FileNotFoundError(
            f"Cache files for '{name}' not found in {CACHE_DIR}."
        )

    arrays = np.load(CACHE_DIR / f"{name}.npz")
    metadata = pd.read_csv(CACHE_DIR / f"{name}_metadata.csv")

    print(f"Loading cache: {name}")

    return SpectralDataset(
        wavelengths=arrays["wavelengths"],
        intensities=arrays["intensities"],
        metadata=metadata,
    )


def cache_exists(name: str) -> bool:
    """
    Check whether all cache files for a spectral dataset exist.

    Parameters
    ----------
    name : str
        Base name of the cached array and metadata files.

    Returns
    -------
    bool
        True if both cache files exist, otherwise False.
    """

    return (
        (CACHE_DIR / f"{name}.npz").exists()
        and (CACHE_DIR / f"{name}_metadata.csv").exists()
    )


def latest_modified_time(paths: list[Path]) -> float:
    """
    Return the latest modification time among existing paths.

    Parameters
    ----------
    paths : list[Path]
        Paths to inspect.

    Returns
    -------
    float
        Latest modification time. Returns 0.0 if none of the paths exists.
    """

    existing_paths = [path for path in paths if path.exists()]

    if not existing_paths:
        return 0.0

    return max(path.stat().st_mtime for path in existing_paths)


def collect_source_files(source_paths: list[Path]) -> list[Path]:
    """
    Collect source files used for cache freshness checking.

    Parameters
    ----------
    source_paths : list[Path]
        Files or directories used to generate a cached dataset.
        Directories are searched recursively.

    Returns
    -------
    list[Path]
        Collected source files.
    """

    files: list[Path] = []

    for source_path in source_paths:
        if source_path.is_file():
            files.append(source_path)

        elif source_path.is_dir():
            files.extend(
                path
                for path in source_path.rglob("*")
                if path.is_file()
            )

    return files


def cache_is_current(
    name: str,
    source_paths: list[Path],
) -> bool:
    """
    Check whether a cached spectral dataset is up to date.

    Parameters
    ----------
    name : str
        Base name of the cached array and metadata files.
    source_paths : list[Path]
        Files or directories used to check cache freshness.

    Returns
    -------
    bool
        True if the cache exists and is newer than all source files,
        otherwise False.
    """

    if not cache_exists(name):
        return False

    cache_files = [
        CACHE_DIR / f"{name}.npz",
        CACHE_DIR / f"{name}_metadata.csv",
    ]

    source_files = collect_source_files(source_paths)

    cache_time = min(path.stat().st_mtime for path in cache_files)
    source_time = latest_modified_time(source_files)

    return cache_time >= source_time


def load_or_build_spectral_dataset(
    name: str,
    builder: Callable[[], SpectralDataset],
    source_paths: list[Path],
    force_rebuild: bool = False,
) -> SpectralDataset:
    """
    Load a cached spectral dataset or rebuild it if required.

    Parameters
    ----------
    name : str
        Base name of the cached array and metadata files.
    builder : Callable[[], SpectralDataset]
        Function used to build the dataset if the cache is missing
        or outdated.
    source_paths : list[Path]
        Files or directories used to check cache freshness.
    force_rebuild : bool, default=False
        If True, ignore the existing cache and rebuild the dataset.

    Returns
    -------
    SpectralDataset
        Loaded or rebuilt spectral dataset.

    Raises
    ------
    ValueError
        If the cache is missing and no builder function is provided.
    """

    if not force_rebuild and cache_is_current(name, source_paths):
        return load_spectral_dataset_cache(name)

    if builder is None:
        raise ValueError(
            f"Cache '{name}' is missing and no builder function is provided."
        )

    print(f"Rebuilding cache: {name}")
    dataset = builder()
    save_spectral_dataset(dataset, name)

    return dataset