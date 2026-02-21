from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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


def _share_any_token(a: List[int], b: List[int]) -> bool:
    return bool(set(a).intersection(set(b)))


def _close_letter_gaps(tokens: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """If a group has non-contiguous letters (e.g. a,c), add missing in-between (b)."""
    out = {k: sorted(set(v)) for k, v in tokens.items()}
    for group, vals in out.items():
        if len(vals) < 2:
            continue
        start, end = min(vals), max(vals)
        out[group] = list(range(start, end + 1))
    return out


def _assign_letters(order: Sequence[str], sig_lookup: Dict[Tuple[str, str], bool]) -> Dict[str, str]:
    """Assign letters in the requested sequential style, with gap-closing."""

    token_codes: Dict[str, List[int]] = {g: [] for g in order}
    next_letter_idx = 0

    for i, current in enumerate(order):
        if not token_codes[current]:
            token_codes[current].append(next_letter_idx)
            for candidate in order[i + 1 :]:
                pair = tuple(sorted((current, candidate)))
                if not sig_lookup.get(pair, False):
                    token_codes[candidate].append(next_letter_idx)
            next_letter_idx += 1
            continue

        to_link: List[str] = []
        for candidate in order[i + 1 :]:
            if _share_any_token(token_codes[current], token_codes[candidate]):
                continue
            pair = tuple(sorted((current, candidate)))
            if not sig_lookup.get(pair, False):
                to_link.append(candidate)

        if to_link:
            token_codes[current].append(next_letter_idx)
            for candidate in to_link:
                token_codes[candidate].append(next_letter_idx)
            next_letter_idx += 1

    token_codes = _close_letter_gaps(token_codes)

    return {
        group: "".join(_letter_for_index(i) for i in sorted(set(idx_list)))
        for group, idx_list in token_codes.items()
    }


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
