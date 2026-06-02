from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

from src.loaders import SpectralDataset, load_samples
from src.plot_style import (
    PRIMARY_BLUE,
    GRID_MAJOR,
    GRID_MINOR,
    CONFUSION_MATRIX_SCALE,
    register_templates
)

from sklearn.metrics import confusion_matrix

register_templates()


def plot_number_spectra_per_sample(
    selected_dataset: SpectralDataset,
    sort_by: str = "n_spectra",
    height: int = 500,
    ascending: bool = False,
    display_order: list[str] | None = None,
    print_summary: bool = True,
):
    """
    Create an interactive bar plot showing the number of selected
    spectra per sample class.

    Uses ``display_label`` and ``display_colour`` from ``samples.csv``.

    Parameters
    ----------
    selected_dataset:
        Spectral dataset after data selection.
    sort_by:
        Column used for automatic sorting of the x-axis categories.
        Ignored if ``display_order`` is provided.
    ascending:
        Whether automatic sorting is performed in ascending order.
    display_order:
        Manual ordering of x-axis categories using the corresponding
        ``display_label`` entries. If provided, this order overrides
        automatic sorting.
    print_summary:
        Whether to print a summary of the plotted data.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly bar chart showing the number of selected
        spectra per sample class.
    """

    samples = load_samples()

    summary = (
        selected_dataset.metadata
        .groupby("sample_id", as_index=False)
        .size()
        .rename(columns={"size": "n_spectra"})
        .merge(
            samples[
                [
                    "sample_id",
                    "display_label",
                    "display_colour",
                    "material_group",
                    "material_type",
                    "polymer_type",
                ]
            ],
            on="sample_id",
            how="left",
        )
    )

    if print_summary:
        class_totals = summary["n_spectra"]

        print(f"Total selected spectra: {int(class_totals.sum())}")
        print(
            "Mean selected spectra per sample class: "
            f"{class_totals.mean():.2f} ± {class_totals.std(ddof=1):.2f}"
        )

        if "sequence_id" in selected_dataset.metadata.columns:
            print(
                "Number of measurement sequences: "
                f"{selected_dataset.metadata['sequence_id'].nunique()}"
            )


    summary["material_group"] = (
        summary["material_group"]
        .str.replace("_", "-", regex=False)
        .str.capitalize()
    )

    if display_order is not None:
        missing_labels = set(display_order) - set(summary["display_label"])

        if missing_labels:
            raise ValueError(
                "Unknown display labels in display_order: "
                f"{sorted(missing_labels)}"
            )

        order_map = {
            label: index
            for index, label in enumerate(display_order)
        }

        summary["_display_order"] = summary["display_label"].map(order_map)
        summary = summary.sort_values("_display_order")

        x_order = display_order

    else:
        if sort_by not in summary.columns:
            raise ValueError(
                f"Unknown sort column '{sort_by}'. "
                f"Available columns are: {list(summary.columns)}"
            )

        summary = summary.sort_values(
            sort_by,
            ascending=ascending,
        )

        x_order = summary["display_label"].tolist()

    fig = px.bar(
        summary,
        x="display_label",
        y="n_spectra",
        color="display_label",
        color_discrete_map=dict(
            zip(summary["display_label"], summary["display_colour"])
        ),
        category_orders={
            "display_label": x_order,
        },
        hover_data=[
            "material_group",
            "n_spectra",
        ],
        labels={
            "display_label": "Sample class",
            "n_spectra": "Number of selected spectra",
            "material_group": "Material group",
        },
        title="Number of selected continuous spectra per sample class",
    )

    fig.update_layout(
        template="broadband_histogram",
        xaxis_tickangle=-45,
        showlegend=False,
        bargap=0.25,
        height=height,
    )

    fig.update_traces(
        marker_line_width=0,
        hovertemplate=(
            "Sample class: %{x}<br>"
            "Number of selected spectra: %{y}<br>"
            "Material group: %{customdata[0]}"
            "<extra></extra>"
        ),
    )

    return fig


def plot_spectra_per_sample(
    dataset: SpectralDataset,
    intensities: np.ndarray | None = None,
    title: str = "Spectra",
    yaxis_title: str = "Intensity / a.u.",
    height: int = 800,
    template: str = "broadband_spectra",
    max_spectra_per_sequence: int | None = 3,
    sample_order: list[str] | None = None,
    opacity: float = 0.55,
):
    """
    Create an interactive line plot of spectra grouped by sample class.

    The figure contains one legend entry per sample class. Clicking a legend
    entry hides or shows all spectra belonging to that sample class. Double
    clicking isolates the selected class.

    Parameters
    ----------
    dataset:
        Spectral dataset containing wavelengths, intensities, and metadata.
    intensities:
        Optional intensity matrix to plot. If omitted, ``dataset.intensities``
        is used. This is useful for showing intermediate preprocessing steps
        while keeping the original metadata.
    title:
        Figure title.
    yaxis_title:
        Label of the y-axis.
    height:
        Figure height in pixels.
    template:
        Plotly template name.
    max_spectra_per_sequence:
        Maximum number of spectra shown per measurement sequence within each
        sample class. Set to None to plot all spectra.
    sample_order:
        Optional manual ordering of sample classes based on ``display_label``.
    opacity:
        Trace opacity.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly line figure.
    """

    wavelengths = np.asarray(dataset.wavelengths, dtype=float)
    spectra = (
        np.asarray(dataset.intensities, dtype=float)
        if intensities is None
        else np.asarray(intensities, dtype=float)
    )

    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape {spectra.shape}.")

    if spectra.shape[1] != wavelengths.size:
        raise ValueError(
            "Number of intensity columns does not match wavelength axis length."
        )

    metadata = dataset.metadata.copy().reset_index(drop=True)

    if len(metadata) != spectra.shape[0]:
        raise ValueError(
            "Number of metadata rows does not match number of spectra."
        )

    samples = load_samples()[
        [
            "sample_id",
            "display_label",
            "display_colour",
            "material_group",
            "material_type",
            "polymer_type",
        ]
    ]

    plot_metadata = metadata.merge(
        samples,
        on="sample_id",
        how="left",
    )

    if plot_metadata["display_label"].isna().any():
        missing = sorted(
            plot_metadata.loc[
                plot_metadata["display_label"].isna(),
                "sample_id",
            ].astype(str).unique()
        )
        raise ValueError(
            "No display metadata found for sample_id values: "
            f"{missing}"
        )

    if sample_order is None:
        sample_labels = (
            plot_metadata["display_label"]
            .drop_duplicates()
            .tolist()
        )
    else:
        missing_labels = set(sample_order) - set(plot_metadata["display_label"])

        if missing_labels:
            raise ValueError(
                "Unknown display labels in sample_order: "
                f"{sorted(missing_labels)}"
            )

        sample_labels = [
            label
            for label in sample_order
            if label in set(plot_metadata["display_label"])
        ]

    colour_map = (
        plot_metadata
        .drop_duplicates("display_label")
        .set_index("display_label")["display_colour"]
        .to_dict()
    )

    fig = go.Figure()

    for sample_label in sample_labels:
        sample_mask = plot_metadata["display_label"] == sample_label
        sample_indices = plot_metadata.index[sample_mask].to_numpy()

        if max_spectra_per_sequence != 'All':
            selected_indices = []

            for sequence_id, sequence_metadata in (
                plot_metadata.loc[sample_indices]
                .groupby("sequence_id", sort=False)
            ):
                n_available = len(sequence_metadata)

                if max_spectra_per_sequence > n_available:
                    sample_label_for_warning = sequence_metadata["display_label"].iloc[0]

                    warnings.warn(
                        f"Requested {max_spectra_per_sequence} spectra for "
                        f"sample class '{sample_label_for_warning}' and sequence "
                        f"'{sequence_id}', but only {n_available} spectra are available. "
                        "All available spectra for this sequence will be displayed.",
                        UserWarning,
                        stacklevel=2,
                    )

                selected_indices.extend(
                    sequence_metadata
                    .head(max_spectra_per_sequence)
                    .index
                    .tolist()
                )

            sample_indices = np.asarray(selected_indices)

        for trace_number, spectrum_index in enumerate(sample_indices):
            row = plot_metadata.loc[spectrum_index].copy()
            row["material_group"] = (
                row["material_group"]
                .replace("_", "-")
                .capitalize()
            )

            fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=spectra[spectrum_index],
                    mode="lines",
                    name=sample_label,
                    legendgroup=sample_label,
                    showlegend=trace_number == 0,
                    line={
                        "color": colour_map[sample_label],
                        "width": 1.2,
                    },
                    opacity=opacity,
                    customdata=np.repeat(
                        [[
                            row["spectrum_id"],
                            row["material_group"],
                        ]],
                        wavelengths.size,
                        axis=0,
                    ),

                    hovertemplate=(
                        "Sample class: " + sample_label + "<br>"
                        "Spectrum ID: %{customdata[0]}<br>"
                        "Wavelength: %{x:.2f} nm<br>"
                        "Value: %{y:.4g}<br>"
                        "Material group: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title=yaxis_title,
        height=height,
        legend_title_text="Sample class",
        legend={
            "groupclick": "togglegroup",
            "itemsizing": "constant",
        },
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.12)",
        title_standoff=10,
        minor=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor=GRID_MINOR,
        ),
        hoverformat=".2f",
        range=[200, 650],
    )

    fig.update_yaxes(
        title_standoff=10,
        minor={
            "showgrid": True,
            "gridwidth": 0.5,
            "gridcolor": GRID_MINOR,
        },
    )

    return fig

def plot_pca_component_distribution(
    pc_counts,
    title: str = (
        "Distribution of number of principal components "
        "selected by PCA during cross-validation"
    ),
    height: int = 800,
    template: str = "broadband_histogram",
):
    """
    Plot the distribution of the number of retained principal
    components across cross-validation folds.

    Parameters
    ----------
    pc_counts:
        Iterable containing the number of retained principal
        components per fold.
    title:
        Figure title.
    height:
        Figure height in pixels.
    template:
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive histogram figure.
    """

    pc_counts = np.asarray(pc_counts)

    bins = np.arange(
        pc_counts.min() - 0.5,
        pc_counts.max() + 1.5,
        1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pc_counts,
            xbins=dict(
                start=bins.min(),
                end=bins.max(),
                size=1,
            ),
            marker_color=PRIMARY_BLUE,
            marker_line_color="black",
            marker_line_width=1,
            hovertemplate=(
                "Number of components: %{x}<br>"
                "Frequency: %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template=template,
        title=title,
        height=height,
        xaxis_title="Number of principal components",
        yaxis_title="Frequency",
        bargap=0.1,
    )

    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_MAJOR,
    )

    return fig

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    normalize: str | None = None,
    title: str = "Confusion matrix",
    height: int = 700,
    template: str = "broadband_confusion_matrix",
):
    """
    Create an interactive Plotly confusion matrix.

    Parameters
    ----------
    y_true:
        True class labels.
    y_pred:
        Predicted class labels.
    labels:
        Optional class order.
    normalize:
        Normalisation mode passed to sklearn.metrics.confusion_matrix.
        Use None, "true", "pred", or "all".
    title:
        Figure title.
    height:
        Figure height in pixels.
    template:
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive confusion matrix.
    """

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=".2f" if normalize else True,
        aspect="square",
        color_continuous_scale=CONFUSION_MATRIX_SCALE,
        zmin=0,
        zmax=25,
        labels={
            "x": "Predicted label",
            "y": "True label",
            "color": "Fraction" if normalize else "Count",
        },
        title=title,
    )

    fig.update_layout(
        template=template,
        height=height,
        width=height,
    )

    fig.update_xaxes(
        side="bottom",
        tickangle=-45,
        automargin=True,
        title_standoff=10,
    )

    fig.update_yaxes(
        autorange="reversed",
        automargin=True,
        title_standoff=10,
    )

    return fig
