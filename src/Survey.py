from __future__ import annotations

import math
from typing import Dict, List, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .data import HermesData  # relative import


def compute_leverage(arr: np.ndarray) -> float:
    """
    1D leverage proxy:
        L = sqrt( sum_i (x_i - mean(x))^2 )
    computed on finite values only.

    Notes
    -----
    - This is essentially sqrt(n) * std(x).
      Grows with both spread and sample size.
    """
    arr = np.asarray(arr, float)
    m = np.isfinite(arr)
    arr = arr[m]
    if arr.size < 2:
        return 0.0
    return float(np.sqrt(np.sum((arr - arr.mean()) ** 2)))


def _infer_name_col(df: pd.DataFrame) -> Optional[str]:
    """
    Best-effort inference of a planet-name column.
    Returns the column name if found, else None.

    Expand this list as your pipeline standardizes.
    """
    candidates = [
        "Planet Name",
        "planet_name",
        "pl_name",
        "Name",
        "name",
        "Planet",
        "planet",
        "Target",
        "target",
        "TOI",
        "toi",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_names(df: pd.DataFrame, name_col: Optional[str]) -> List[str]:
    """
    Row-aligned extraction of planet names from df.

    If name_col is None or missing, falls back to "row_{i}".
    """
    if name_col is not None and name_col in df.columns:
        # make robust strings; preserve ordering aligned to df rows
        s = df[name_col].astype(str).fillna("")
        # optional: strip whitespace
        names = [x.strip() for x in s.to_list()]
        # if any are empty, replace those with row_i identifiers
        out: List[str] = []
        for i, nm in enumerate(names):
            out.append(nm if nm else f"row_{i}")
        return out

    # fallback: stable identifiers even if there is no name column
    return [f"row_{i}" for i in range(len(df))]



class Survey:
    """
    Survey
    ------
    A `Survey` represents *one sampled subset* of the parent Hermes dataset.

    Think of it as the atomic unit you fit models to:
      - It contains the DataFrame slice (`df`) that your model will ingest
      - It contains metadata about how it was drawn (survey_id, class_label)
      - NEW: it also carries planet/target names forward so you can:
          * inspect "what was actually sampled"
          * label/annotate plots per-survey
          * later overlay posterior predictive / fitted curves with planet names

    What is stored?
    ---------------
    Nothing permanent is written anywhere — names live only in-memory inside
    each Survey instance:

      - `planet_names`: list[str] aligned with df rows
      - `planet_index`: dict[str, list[int]] mapping name -> row indices
        (list because duplicates can happen; e.g., repeated identifiers)
    """

    def __init__(
        self,
        survey_id: int,
        class_label: str,
        df: pd.DataFrame,
        *,
        name_col: Optional[str] = None,
    ):
        self.survey_id = int(survey_id)
        self.class_label = str(class_label)
        self.df = df.reset_index(drop=True)

        # NEW: carry forward names (row-aligned)
        self.name_col = name_col if (name_col in self.df.columns) else None
        self.planet_names: List[str] = _extract_names(self.df, self.name_col)

        # NEW: quick lookup from name -> rows inside this survey
        self.planet_index: Dict[str, List[int]] = {}
        for i, nm in enumerate(self.planet_names):
            self.planet_index.setdefault(nm, []).append(i)

    @property
    def n(self) -> int:
        return len(self.df)

    # --------- convenience: "what is this survey made of?" -------------------

    def targets(self) -> List[str]:
        """Return the planet/target names in this survey (row-aligned)."""
        return list(self.planet_names)

    def target_table(self, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Return a small inspection table with names + selected columns.

        Examples
        --------
        survey.target_table(["logM", "Star Metallicity", "Planet Radius"])
        """
        out = pd.DataFrame({"planet_name": self.planet_names})
        if cols:
            cols = [c for c in cols if c in self.df.columns]
            out = pd.concat([out, self.df.loc[:, cols].reset_index(drop=True)], axis=1)
        return out

    def row_for_target(self, name: str) -> List[int]:
        """Return row indices in `df` for a given target name (may be multiple)."""
        return self.planet_index.get(str(name), [])

    # leverage and metrics testing.

    def leverage(self, col: str = "logM") -> float:
        """
        Leverage of the specified column.
        Default: leverage of logM.
        """
        return compute_leverage(self.df[col].to_numpy(float))

    def leverage_2D(self, col_x: str = "logM", col_y: str = "Star Metallicity") -> float:
        """
        2D leverage proxy as quadrature sum of 1D leverages.
        """
        return float(np.sqrt(self.leverage(col_x) ** 2 + self.leverage(col_y) ** 2))

    def leverage_3D(
        self,
        col_x: str = "logM",
        col_y: str = "Star Metallicity",
        col_z: str = "Planet Radius",
    ) -> float:
        """
        3D leverage proxy.

        NOTE:
        Your current definition uses cube-root of sum-of-squares:
            cbrt(Lx^2 + Ly^2 + Lz^2)

        which is somewhat unconventional (most "quadrature" norms use sqrt).
        I kept it as-is to avoid changing semantics across your pipeline.
        """
        return float(math.cbrt(self.leverage(col_x) ** 2 + self.leverage(col_y) ** 2 + self.leverage(col_z) ** 2))

    def mahalanobis_3D(
        self,
        col_x: str = "logM",
        col_y: str = "Star Metallicity",
        col_z: str = "Planet Radius",
    ) -> float:
        """
        Mean 3D Mahalanobis distance of points in the specified columns,
        computed within this survey.

        Returns 0 if there are <2 finite rows.
        """
        data = self.df[[col_x, col_y, col_z]].to_numpy(float)
        data = data[np.all(np.isfinite(data), axis=1)]
        if data.shape[0] < 2:
            return 0.0
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        diff = data - mean
        m_dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
        return float(np.mean(m_dist))


class SurveySampler:
    """
    SurveySampler
    -------------
    Builds nested mass classes (S1..S4) from HermesData and draws many Survey
    objects from them over an N-grid.

    Core idea:
      - HermesData is the "parent population" (your ARIEL MCS or synthetic set)
      - SurveySampler constructs *class-conditional subsets* (S1..S4)
      - sample_grid draws many Survey realizations without replacement
        for each (class, N) combination.

    NEW:
      - You can specify `name_col` (planet-name column) once in the sampler.
      - Each Survey produced will carry those names forward in-memory.

    Why this matters:
      - You can compute survey-level metrics (leverage, WAIC diffs, etc.)
        and still have full traceability to the *exact targets* that drove
        that result.
    """

    def __init__(
        self,
        hermes: HermesData,
        rng_seed: Optional[int] = None,
        *,
        name_col: Optional[str] = None,
    ):
        self.hermes = hermes
        self.rng = np.random.default_rng(rng_seed)

        # NEW: choose / infer planet name column once (used for all surveys)
        if name_col is None:
            name_col = _infer_name_col(self.hermes.df)
        self.name_col = name_col

        # build nested mass classes based on logM quantiles
        self.mass_classes: Dict[str, pd.DataFrame] = self._build_mass_classes()

    def _build_mass_classes(self) -> Dict[str, pd.DataFrame]:
        df = self.hermes.df
        q25, q50, q75 = df["logM"].quantile([0.25, 0.5, 0.75])

        classes: Dict[str, pd.DataFrame] = {}
        classes["S1"] = df.copy()
        classes["S2"] = df[df["logM"] >= q25].copy()
        classes["S3"] = df[df["logM"] >= q50].copy()
        classes["S4"] = df[df["logM"] >= q75].copy()
        return classes

    def sample_grid(
        self,
        N_grid: Iterable[int],
        n_reps_per_combo: int = 10,
        class_order: Optional[List[str]] = None,
    ) -> List[Survey]:
        """
        For each class in class_order and each N in N_grid,
        draw n_reps_per_combo surveys without replacement.

        Returns
        -------
        surveys : list[Survey]
            Flat list of Survey objects. Each Survey contains:
              - df: sampled targets
              - planet_names, planet_index: NEW traceability layer
        """
        if class_order is None:
            class_order = ["S1", "S2", "S3", "S4"]

        surveys: List[Survey] = []
        survey_id = 1

        for label in class_order:
            if label not in self.mass_classes:
                continue
            subset = self.mass_classes[label]
            n_available = len(subset)

            for N in N_grid:
                if N > n_available:
                    continue

                for _ in range(n_reps_per_combo):
                    rs = int(self.rng.integers(0, 2**32 - 1))
                    sample_df = subset.sample(n=N, replace=False, random_state=rs)
                    surveys.append(
                        Survey(
                            survey_id,
                            label,
                            sample_df,
                            name_col=self.name_col,
                        )
                    )
                    survey_id += 1

        return surveys
