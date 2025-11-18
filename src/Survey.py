from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Iterable, Optional

from .data import HermesData  # relative import


def compute_leverage(arr: np.ndarray) -> float:
    """
    L = sqrt( sum_i (x_i - mean(x))^2 ) on finite values.
    """
    arr = np.asarray(arr, float)
    m = np.isfinite(arr)
    arr = arr[m]
    if arr.size < 2:
        return 0.0
    return float(np.sqrt(np.sum((arr - arr.mean())**2)))


class Survey:
    """
    One survey sample: a subset of hermes_data.
    Not a dataclass on purpose.
    """

    def __init__(self, survey_id: int, class_label: str, df: pd.DataFrame):
        self.survey_id = int(survey_id)
        self.class_label = str(class_label)
        self.df = df.reset_index(drop=True)

    @property
    def n(self) -> int:
        return len(self.df)

    def leverage(self, col: str = "logM") -> float:
        """
        Leverage of the specified column.
        Default: leverage of logM.
        """
        return compute_leverage(self.df[col].to_numpy(float))


class SurveySampler:
    """
    Build nested mass classes (S1..S4) from HermesData
    and draw many Survey objects from them on an N-grid.
    """

    def __init__(self, hermes: HermesData, rng_seed: Optional[int] = None):
        self.hermes = hermes
        self.rng = np.random.default_rng(rng_seed)
        # build nested mass classes based on logM quantiles
        self.mass_classes: Dict[str, pd.DataFrame] = self._build_mass_classes()

    def _build_mass_classes(self) -> Dict[str, pd.DataFrame]:
        df = self.hermes.df
        q25, q50, q75 = df["logM"].quantile([0.25, 0.5, 0.75])

        classes: Dict[str, pd.DataFrame] = {}
        # S1: all planets
        classes["S1"] = df.copy()
        # S2: logM >= 25th percentile
        classes["S2"] = df[df["logM"] >= q25].copy()
        # S3: logM >= 50th percentile
        classes["S3"] = df[df["logM"] >= q50].copy()
        # S4: logM >= 75th percentile
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

        Returns flat list of Survey objects.
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
                    surveys.append(Survey(survey_id, label, sample_df))
                    survey_id += 1

        return surveys


# Backwards-compatible alias if you ever used SurveyFactory
SurveyFactory = SurveySampler
