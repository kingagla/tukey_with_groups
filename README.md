# tukey_with_groups

A small Python package that runs **Tukey HSD** and returns a ready-to-use summary table with compact letter groups.

## What the group letters mean

The returned `group` column follows the convention you described:

- `a` = highest mean group bucket.
- `ab` = this group is **not significantly different** from both `a` and `b` groups.
- Different letters with no overlap (for example `a` vs `c`) indicate a statistically significant difference at `alpha`.

In short: groups sharing at least one letter are not significantly different from each other.

## Installation

```bash
pip install .
```

To build a distributable wheel/sdist:

```bash
python -m build
```

## Usage

### 1) Wide input (each column is a treatment/group)

```python
import pandas as pd
from tukey_with_groups import tukey

wide_df = pd.DataFrame({
    "A": [10, 11, 10, 12],
    "B": [8, 9, 7, 8],
    "C": [5, 6, 4, 5],
})

summary_df = tukey(wide_df)
print(summary_df)
```

### 2) Long input (`res_var` + `xfac_var`)

```python
long_df = wide_df.melt(var_name="factor", value_name="value")
summary_df = tukey(long_df, res_var="value", xfac_var="factor")
```

## API

```python
tukey(df, res_var=None, xfac_var=None, alpha=0.05) -> pandas.DataFrame
```

Returns a dataframe with:

- `factor`: group/treatment name
- `mean`: group mean (sorted descending)
- `group`: compact letters (`a`, `ab`, `b`, ...)
