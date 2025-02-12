from typing import Any
import numpy as np
import pandas as pd


def _convert_dict_to_matrix(data_dict: dict[str, dict[str, Any]]) -> np.ndarray:
    """
    Convert a nested dictionary to a numpy matrix with float entries.
    Will respect the order of the dictionaries.

    Args:
      data_dict (dict[str, dict[str, Any]]): Top level keys are rows, bottom level keys are columns.
        Values must be convertable to float.

    Returns
      np.ndarray: Matrix representing data in dictionary.
    """
    inner_keys = set()
    for i, v in enumerate(data_dict.values()):
        if i == 0:
            inner_keys = set(v.keys())
        else:
            assert inner_keys == set(
                v.keys()
            ), "Inner keys do not match across all rows"

    df = pd.DataFrame.from_dict(data_dict).T

    # ignoring mypy error, mypy not up to date with pandas deprecating applymap
    df = df.map(
        lambda x: float(x) if x not in ("NaN", "nan", None, np.nan) else np.nan
    )  # type: ignore[operator]

    return df.to_numpy()
