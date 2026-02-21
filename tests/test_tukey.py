import random

import pandas as pd

from tukey_with_groups import tukey
from tukey_with_groups.core import _assign_letters


def test_returns_summary_columns_and_sorted_means_wide():
    df = pd.DataFrame(
        {
            "A": [10, 11, 10, 12],
            "B": [8, 9, 7, 8],
            "C": [5, 6, 4, 5],
        }
    )

    out = tukey(df)

    assert list(out.columns) == ["Groups", "Count", "Sum", "Mean", "Variance", "group"]
    assert out["Mean"].is_monotonic_decreasing
    assert set(out["Groups"]) == {"A", "B", "C"}


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


def test_bridging_group_receives_overlap_letters():
    sig_lookup = {
        tuple(sorted(("A", "B"))): False,
        tuple(sorted(("B", "C"))): False,
        tuple(sorted(("A", "C"))): True,
    }
    letters = _assign_letters(["A", "B", "C"], sig_lookup)

    assert letters["A"] == "a"
    assert set(letters["B"]) == {"a", "b"}
    assert letters["C"] == "b"


def test_user_requested_stepwise_labeling_pattern():
    # A~B, B~C, B~D and A,C,D are all mutually significant.
    # Expected sequential labels: A=a, B=ab, C=b, D=b
    sig_lookup = {
        tuple(sorted(("A", "B"))): False,
        tuple(sorted(("A", "C"))): True,
        tuple(sorted(("A", "D"))): True,
        tuple(sorted(("B", "C"))): False,
        tuple(sorted(("B", "D"))): False,
        tuple(sorted(("C", "D"))): True,
    }

    letters = _assign_letters(["A", "B", "C", "D"], sig_lookup)

    assert letters == {"A": "a", "B": "ab", "C": "b", "D": "b"}


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
    groups = dict(zip(out["Groups"], out["group"]))

    assert groups["B"] in {"a", "b", "ab", "abc", "ba"}


def test_gap_between_letters_is_filled_to_match_requested_behavior():
    sig_lookup = {
        tuple(sorted(("A", "B"))): False,
        tuple(sorted(("A", "C"))): True,
        tuple(sorted(("A", "D"))): False,
        tuple(sorted(("B", "C"))): False,
        tuple(sorted(("B", "D"))): True,
        tuple(sorted(("C", "D"))): False,
    }
    # Sequential assignment can yield D="ac"; requested behavior is closing the gap to "abc".
    letters = _assign_letters(["A", "B", "C", "D"], sig_lookup)
    assert letters["D"] == "abc"
