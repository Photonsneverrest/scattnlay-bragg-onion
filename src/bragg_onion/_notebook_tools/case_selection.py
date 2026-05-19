# case_selection.py

import numpy as np
import pandas as pd


def select_single_case_by_filters(df: pd.DataFrame, filters: dict) -> pd.Series:
    subset = df.copy()

    for col, val in filters.items():
        if col not in subset.columns:
            raise KeyError(f"Column '{col}' is not in dataframe.")

        if np.issubdtype(subset[col].dtype, np.number):
            subset = subset[np.isclose(subset[col].astype(float), float(val))]
        else:
            subset = subset[subset[col] == val]

    if len(subset) == 0:
        raise ValueError(f"No match found for filters: {filters}")

    if len(subset) > 1:
        raise ValueError(
            f"Filters not unique ({len(subset)} matches). "
            f"Refine filters: {filters}"
        )

    return subset.iloc[0]


def select_best_etaC(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("eta_C", ascending=False).iloc[0]