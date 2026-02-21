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
        long_df["factor"] = long_df["factor"].astype(str)
        return _PreparedData(long_df=long_df, groups=sorted(long_df["factor"].unique().tolist()))

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


def _letter_for_index(idx: int) -> str:
    """Convert 0-based index to a, b, ..., z, aa, ab, ..."""
    idx += 1
    out = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _absorb_columns(columns: List[Set[str]]) -> List[Set[str]]:
    unique: List[Set[str]] = []
    for col in columns:
        if col and col not in unique:
            unique.append(col)

    keep: List[Set[str]] = []
    for i, col in enumerate(unique):
        if any(col < other for j, other in enumerate(unique) if i != j):
            continue
        keep.append(col)
    return keep


def _assign_letters(order: Sequence[str], sig_lookup: Dict[Tuple[str, str], bool]) -> Dict[str, str]:
    """Assign compact-letter-display groups using a split/absorb algorithm.

    Ensures no significant pair shares a letter and non-significant chains can receive
    overlapping letters (e.g., A~B, B~C, A!=C => B can be `ab`).
    """

    rank = {name: idx for idx, name in enumerate(order)}
    sig_pairs = [pair for pair, is_sig in sig_lookup.items() if is_sig]
    sig_pairs.sort(key=lambda p: (min(rank.get(p[0], 10**9), rank.get(p[1], 10**9)), p))

    columns: List[Set[str]] = [set(order)]

    for a, b in sig_pairs:
        updated: List[Set[str]] = []
        for col in columns:
            if a in col and b in col:
                left = set(col)
                right = set(col)
                left.discard(a)
                right.discard(b)
                updated.extend([left, right])
            else:
                updated.append(set(col))
        columns = _absorb_columns(updated)

    columns.sort(key=lambda c: min(rank[g] for g in c))

    codes: Dict[str, str] = {g: "" for g in order}
    for idx, col in enumerate(columns):
        letter = _letter_for_index(idx)
        for group in order:
            if group in col:
                codes[group] += letter

    return codes


def tukey(
    df: pd.DataFrame,
    res_var: str | None = None,
    xfac_var: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Tukey HSD and return summary stats with compact letter groups."""

    if not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    prepared = _prepare_data(df=df, res_var=res_var, xfac_var=xfac_var)
    if len(prepared.groups) < 2:
        raise ValueError("Need at least two groups for Tukey HSD.")

    summary_df = (
        prepared.long_df.groupby("factor", as_index=False)["value"]
        .agg(Count="size", Sum="sum", Mean="mean", Variance="var")
        .sort_values("Mean", ascending=False)
        .rename(columns={"factor": "Groups"})
        .reset_index(drop=True)
    )

    tukey_result = pairwise_tukeyhsd(
        endog=prepared.long_df["value"],
        groups=prepared.long_df["factor"],
        alpha=alpha,
    )
    sig_lookup = _build_significance_lookup(tukey_result)

    order = summary_df["Groups"].tolist()
    letters = _assign_letters(order=order, sig_lookup=sig_lookup)

    summary_df["group"] = summary_df["Groups"].map(letters)
    return summary_df
