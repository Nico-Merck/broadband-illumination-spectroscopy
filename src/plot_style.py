from __future__ import annotations

import plotly.io as pio
import copy

PRIMARY_BLUE = "#004A99"
GRID_MAJOR = "rgba(0,0,0,0.12)"
GRID_MINOR = "rgba(0,0,0,0.06)"
CONFUSION_MATRIX_SCALE = [
    [0.0, "#ffffff"],
    [1.0, PRIMARY_BLUE],
]

def register_templates() -> None:
    """
    Register all Plotly templates used throughout the project.
    """

    # ------------------------------------------------------------------
    # Base template
    # ------------------------------------------------------------------

    base_template = {
        "layout": {
            "template": "simple_white",

            "font": {
                "family": "Arial Narrow",
                "size": 14,
                "color": "black",
            },

            "title": {
                "x": 0.5,
                "xanchor": "center",
            },

            "plot_bgcolor": "white",
            "paper_bgcolor": "white",

            "margin": {
                "l": 80,
                "r": 40,
                "t": 80,
                "b": 100,
            },

            "xaxis": {
                "showline": True,
                "linewidth": 1.5,
                "linecolor": "black",

                "showgrid": False,

                "ticks": "outside",
                "tickwidth": 1.5,
                "tickcolor": "black",
                "ticklen": 6,
            },

            "yaxis": {
                "showline": True,
                "linewidth": 1.5,
                "linecolor": "black",

                "showgrid": True,
                "gridwidth": 1,
                "gridcolor": GRID_MAJOR,

                "ticks": "outside",
                "tickwidth": 1.5,
                "tickcolor": "black",
                "ticklen": 6,

                "zeroline": False,
            },

            "legend": {
                "borderwidth": 0,
            },
        }
    }

    pio.templates["broadband_base"] = base_template

    # ------------------------------------------------------------------
    # Spectra template
    # ------------------------------------------------------------------

    pio.templates["broadband_spectra"] = copy.deepcopy(pio.templates["broadband_base"])

    # ------------------------------------------------------------------
    # Confusion matrix template
    # ------------------------------------------------------------------

    pio.templates["broadband_confusion_matrix"] = copy.deepcopy(pio.templates["broadband_base"])

    pio.templates[
        "broadband_confusion_matrix"
    ].layout.update(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    # ------------------------------------------------------------------
    # Histogram / spectra count template
    # ------------------------------------------------------------------

    pio.templates["broadband_histogram"] = copy.deepcopy(pio.templates["broadband_base"])

    pio.templates["broadband_histogram"].layout.update(
        bargap=0.25,
        yaxis=dict(showgrid=True),
    )

    # Default template
    pio.templates.default = "broadband_spectra"