from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


@dataclass(frozen=True)
class _PreparedData:
    long_df: pd.DataFrame
    groups: List[str]


def _prepare_data(df: pd.DataFrame, res_var: str | None, xfac_var: str | None) -> _PreparedData:
    if (res_var is None) != (xfac_var is None):
        raise ValueError("Provide both `res_var` and `xfac_var`, or neither for wide-format input.")

    if res_var is None and xfac_var is None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        long_df = df.melt(var_name="factor", value_name="value").dropna()
        return _PreparedData(long_df=long_df, groups=sorted(long_df["factor"].astype(str).unique().tolist()))

    if res_var not in df.columns or xfac_var not in df.columns:
        raise ValueError("`res_var` and `xfac_var` must exist in DataFrame columns.")

    long_df = df[[res_var, xfac_var]].copy().dropna()
    long_df.columns = ["value", "factor"]
    long_df["factor"] = long_df["factor"].astype(str)
    return _PreparedData(long_df=long_df, groups=sorted(long_df["factor"].unique().tolist()))


def _build_significance_lookup(tukey_result) -> Dict[Tuple[str, str], bool]:
    sig: Dict[Tuple[str, str], bool] = {}
    for row in tukey_result._results_table.data[1:]:
        g1, g2, _meandiff, _p_adj, _lower, _upper, reject = row
        pair = tuple(sorted((str(g1), str(g2))))
        sig[pair] = bool(reject)
    return sig


def _is_significant(sig_lookup: Dict[Tuple[str, str], bool], a: str, b: str) -> bool:
    if a == b:
        return False
    return sig_lookup.get(tuple(sorted((a, b))), False)



def _letter_for_index(idx: int) -> str:
    """Convert 0-based index to a, b, ..., z, aa, ab, ..."""
    idx += 1
    out = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _assign_letters(order: Sequence[str], sig_lookup: Dict[Tuple[str, str], bool]) -> Dict[str, str]:
    letter_sets: List[Set[str]] = []

    for group in order:
        assigned = False
        for s in letter_sets:
            if all(not _is_significant(sig_lookup, group, member) for member in s):
                s.add(group)
                assigned = True
        if not assigned:
            letter_sets.append({group})

    codes: Dict[str, str] = {g: "" for g in order}
    for idx, members in enumerate(letter_sets):
        letter = _letter_for_index(idx)
        for group in members:
            codes[group] += letter

    return codes


def tukey(
    df: pd.DataFrame,
    res_var: str | None = None,
    xfac_var: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Tukey HSD and return means with compact letter groups.

    Parameters
    ----------
    df
        Input dataframe. For wide format, each column is a group and rows are replicates.
        For long format, pass both `res_var` and `xfac_var`.
    res_var
        Response variable column (required for long format).
    xfac_var
        Factor/group column (required for long format).
    alpha
        Significance threshold for Tukey HSD.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: factor, mean, group. Means are sorted descending.
    """

    if not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    prepared = _prepare_data(df=df, res_var=res_var, xfac_var=xfac_var)
    if len(prepared.groups) < 2:
        raise ValueError("Need at least two groups for Tukey HSD.")

    means = (
        prepared.long_df.groupby("factor", as_index=False)["value"]
        .mean()
        .sort_values("value", ascending=False)
        .rename(columns={"value": "mean"})
        .reset_index(drop=True)
    )

    tukey_result = pairwise_tukeyhsd(
        endog=prepared.long_df["value"],
        groups=prepared.long_df["factor"],
        alpha=alpha,
    )
    sig_lookup = _build_significance_lookup(tukey_result)

    order = means["factor"].tolist()
    letters = _assign_letters(order=order, sig_lookup=sig_lookup)

    summary_df = means.copy()
    summary_df["group"] = summary_df["factor"].map(letters)
    return summary_df
