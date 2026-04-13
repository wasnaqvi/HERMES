# src/__init__.py
from .Model import (
    Model,
    MetModel,
    _met_model,
    _met_model_no_scatter,
    _met_model_no_stellar,
    make_met_model_fixed_scatter,
)
from .Survey import Survey, SurveySampler
from .data import HermesData
from .scatter_threshold import (
    ScatterThresholdConfig,
    make_synthetic_catalog,
    plot_scatter_threshold_results,
    print_scatter_threshold_summary,
    run_scatter_threshold,
    summarize_scatter_threshold,
)

HermesModel = MetModel  # legacy alias

__all__ = [
    "Model",
    "MetModel",
    "HermesModel",
    "Survey",
    "SurveySampler",
    "HermesData",
    "ScatterThresholdConfig",
    "_met_model",
    "_met_model_no_scatter",
    "_met_model_no_stellar",
    "make_met_model_fixed_scatter",
    "make_synthetic_catalog",
    "run_scatter_threshold",
    "summarize_scatter_threshold",
    "plot_scatter_threshold_results",
    "print_scatter_threshold_summary",
]
__version__ = "1.1.0"
