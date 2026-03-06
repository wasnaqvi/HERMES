# src/RockyUtils.py
"""Data loading and preprocessing for HERMES4Rocky (CMF regression)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROCKY_DIR = Path(__file__).resolve().parent.parent / "dataset" / "Rocky"
DEFAULT_STAR_CMF_ERR = 0.03   # from Brandt & Zarka paper
DEFAULT_PLANET_CMF_ERR = 0.3  # from Brandt & Zarka paper


def load_rocky_surveys(data_dir: Path = ROCKY_DIR) -> Dict[str, pd.DataFrame]:
    """Load all 5 Rocky CSV files, return {survey_name: df}."""
    surveys: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        name = csv_path.stem  # e.g. "Behmard"
        df = pd.read_csv(csv_path)
        df = df.dropna(how="all")  # drop fully empty rows
        surveys[name] = df
    return surveys


def build_pooled_dataset(
    surveys: Dict[str, pd.DataFrame],
    star_cmf_err: float = DEFAULT_STAR_CMF_ERR,
    planet_cmf_err: float = DEFAULT_PLANET_CMF_ERR,
    survey_subset: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Filter to rows with both Star_CMF and Planet CMF present.

    Parameters
    ----------
    surveys : dict
        {survey_name: DataFrame} from load_rocky_surveys().
    star_cmf_err, planet_cmf_err : float
        Default measurement uncertainties.
    survey_subset : sequence of str or None
        If given, only include these surveys (e.g. ["Behmard", "Brinkman"]).
        Surveys without CMF data are silently skipped.
        If None (default), all surveys with valid CMF data are included.

    Returns unified DataFrame with columns:
        pl_name, survey, star_cmf, planet_cmf, star_cmf_err, planet_cmf_err
    """
    frames = []
    for name, df in surveys.items():
        if survey_subset is not None and name not in survey_subset:
            continue
        if "Star_CMF" not in df.columns or "Planet CMF" not in df.columns:
            continue
        sub = df.dropna(subset=["Star_CMF", "Planet CMF"])[["pl_name", "Star_CMF", "Planet CMF"]].copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={"Star_CMF": "star_cmf", "Planet CMF": "planet_cmf"})
        sub["star_cmf"] = pd.to_numeric(sub["star_cmf"], errors="coerce")
        sub["planet_cmf"] = pd.to_numeric(sub["planet_cmf"], errors="coerce")
        sub = sub.dropna(subset=["star_cmf", "planet_cmf"])
        sub["survey"] = name
        sub["star_cmf_err"] = star_cmf_err
        sub["planet_cmf_err"] = planet_cmf_err
        frames.append(sub)

    pooled = pd.concat(frames, ignore_index=True)
    return pooled[["pl_name", "survey", "star_cmf", "planet_cmf", "star_cmf_err", "planet_cmf_err"]]


def build_partial_pooling_dataset(pooled_df: pd.DataFrame) -> dict:
    """
    For the partial-pooling model, build index arrays mapping
    each observation to a unique planet.

    Returns dict with:
        unique_planets  – list of unique planet names
        planet_idx      – int array (N_obs,) mapping each obs -> unique planet index
        star_cmf_obs    – float array (N_obs,)
        planet_cmf_obs  – float array (N_obs,)
        star_cmf_err    – float array (N_obs,)
        planet_cmf_err  – float array (N_obs,)
        survey_labels   – array of survey names per observation
        n_planets       – number of unique planets
        n_obs           – total observations
    """
    unique_planets = sorted(pooled_df["pl_name"].unique())
    planet_to_idx = {name: i for i, name in enumerate(unique_planets)}
    planet_idx = np.array([planet_to_idx[n] for n in pooled_df["pl_name"]], dtype=np.int32)

    return {
        "unique_planets": unique_planets,
        "planet_idx": planet_idx,
        "star_cmf_obs": pooled_df["star_cmf"].to_numpy(dtype=np.float64),
        "planet_cmf_obs": pooled_df["planet_cmf"].to_numpy(dtype=np.float64),
        "star_cmf_err": pooled_df["star_cmf_err"].to_numpy(dtype=np.float64),
        "planet_cmf_err": pooled_df["planet_cmf_err"].to_numpy(dtype=np.float64),
        "survey_labels": pooled_df["survey"].to_numpy(),
        "n_planets": len(unique_planets),
        "n_obs": len(pooled_df),
    }
