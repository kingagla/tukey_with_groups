import random

import pandas as pd

from tukey_with_groups import tukey


def test_returns_summary_columns_and_sorted_means_wide():
    df = pd.DataFrame(
        {
            "A": [10, 11, 10, 12],
            "B": [8, 9, 7, 8],
            "C": [5, 6, 4, 5],
        }
    )

    out = tukey(df)

    assert list(out.columns) == ["factor", "mean", "group"]
    assert out["mean"].is_monotonic_decreasing
    assert set(out["factor"]) == {"A", "B", "C"}


def test_long_and_wide_input_match():
    wide = pd.DataFrame(
        {
            "A": [3, 4, 5, 4],
            "B": [2, 2, 3, 2],
            "C": [1, 1, 2, 1],
        }
    )
    long = wide.melt(var_name="grp", value_name="val")

    out_wide = tukey(wide)
    out_long = tukey(long, res_var="val", xfac_var="grp")

    pd.testing.assert_frame_equal(out_wide, out_long)


def test_overlap_groups_can_produce_ab_pattern():
    random.seed(123)
    n = 40

    def draw(mu, sigma, size):
        return [random.gauss(mu, sigma) for _ in range(size)]

    df = pd.DataFrame(
        {
            "A": draw(10.0, 0.55, n),
            "B": draw(9.8, 0.55, n),
            "C": draw(9.55, 0.55, n),
        }
    )

    out = tukey(df)
    groups = dict(zip(out["factor"], out["group"]))

    assert groups["B"] in {"ab", "ba"}
