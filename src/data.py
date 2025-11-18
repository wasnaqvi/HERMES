# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class HermesData:
    """
    Thin wrapper around the HERMES dataframe.
    """
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str) -> "HermesData":
        """
        Load the synthetic HERMES CSV and wrap it.
        """
        df = pd.read_csv(path)
        return cls(df=df)

    @property
    def n(self) -> int:
        return len(self.df)

    def describe_columns(self) -> pd.DataFrame:
        """
        Convenience: get basic stats on key columns.
        """
        cols = ["logM", "log(X_H2O)", "uncertainty_lower", "uncertainty_upper"]
        cols = [c for c in cols if c in self.df.columns]
        return self.df[cols].describe()

    def mass_quantile_classes(
        self,
        mass_col: str = "logM",
    ) -> Dict[str, "HermesData"]:
        """
        Return four nested mass classes S1..S4 based on quartiles of logM,
        matching what SurveySampler does.

        - S1: full sample
        - S2: logM >= 25th percentile
        - S3: logM >= 50th percentile
        - S4: logM >= 75th percentile
        """
        df = self.df.copy()
        logM = df[mass_col].to_numpy(float)

        q25, q50, q75 = np.quantile(logM, [0.25, 0.5, 0.75])

        mask2 = logM >= q25
        mask3 = logM >= q50
        mask4 = logM >= q75

        return {
            "S1": HermesData(df.reset_index(drop=True)),
            "S2": HermesData(df.loc[mask2].reset_index(drop=True)),
            "S3": HermesData(df.loc[mask3].reset_index(drop=True)),
            "S4": HermesData(df.loc[mask4].reset_index(drop=True)),
        }
