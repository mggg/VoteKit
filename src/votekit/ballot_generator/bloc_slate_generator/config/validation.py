"""Validation helpers and data-shape conversion utilities for bloc-slate config."""

import math
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from numbers import Real
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd

from votekit.pref_interval import PreferenceInterval

UNSET_VALUE = -1.0
FLOAT_TOL = 1e-8


class ConfigurationWarning(UserWarning):
    """Raised when adjusting a setting in BlocSlateConfig may cause a conflict."""


_original_formatwarning = warnings.formatwarning


def _config_warning_format(
    message: Union[Warning, str],
    category: type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:  # pragma: no cover
    if issubclass(category, ConfigurationWarning):
        return f"{category.__name__}: {message}\n"
    return _original_formatwarning(message, category, filename, lineno, line)


warnings.formatwarning = cast(Any, _config_warning_format)


BlocProportionMapping = Union[Mapping[str, Union[int, float]], pd.Series]
# Backward compatibility alias; keep the original misspelling available.
BlocPropotionMapping = BlocProportionMapping
CohesionMapping = Union[Mapping[str, Union[Mapping[str, float], pd.Series]], pd.DataFrame]
PreferenceIntervalLike = Union[Mapping[str, Union[float, int]], PreferenceInterval]
PreferenceMapping = Union[
    Mapping[str, Mapping[str, PreferenceIntervalLike]],
    pd.DataFrame,
]


def _is_finite_real(x: object) -> bool:
    """
    Return True if x is a finite real number (int or float), False otherwise.

    Args:
        x (object): The object to check.
    Returns:
        bool: True if x is a finite real number, False otherwise.
    """
    # Reject bools (subclass of int) and non-finite numbers
    if isinstance(x, bool) or not isinstance(x, Real):
        return False
    try:
        return math.isfinite(float(x))
    except Exception:  # pragma: no cover
        return False


def _to_finite_float_array(
    values: Union[pd.Series, pd.DataFrame],
    *,
    context: str,
) -> np.ndarray:
    """
    Convert values to float ndarray and enforce finiteness.

    Args:
        values (Union[pd.Series, pd.DataFrame]): The values to convert.
        context (str): Description of the values for error messages.

    Returns:
        np.ndarray: The converted values as a float ndarray.
    """
    try:
        arr = values.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{context} must contain numeric values.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{context} contains non-finite values.")
    return arr


def _unset_mask(values: np.ndarray) -> np.ndarray:
    """
    Return boolean mask for unset sentinel values.

    Args:
        values (np.ndarray): The array of values to check.

    Returns:
        np.ndarray: A boolean array where True indicates the value is an unset sentinel value.
    """
    return np.isclose(values, UNSET_VALUE, atol=FLOAT_TOL)


def _invalid_negative_values(values: np.ndarray) -> list[float]:
    """
    Return sorted distinct negative values that are not unset sentinel values.

    Args:
        values (np.ndarray): The array of values to check.

    Returns:
        list[float]: A sorted list of distinct negative values that are not unset sentinel values.
    """
    invalid_mask = (values < 0) & (~_unset_mask(values))
    return sorted({float(v) for v in values[invalid_mask]})


def _sum_differs_from_one(value: float) -> bool:
    """
    Return True when value is not within tolerance of 1.0.

    Args:
        value (float): The value to check.

    Returns:
        bool: True if value is not within tolerance of 1.0, False otherwise.
    """
    return abs(float(value) - 1.0) > FLOAT_TOL


def _first_error_probability_row(
    row_vals: Union[pd.Series, pd.DataFrame],
    *,
    context: str,
    invalid_negative_error: Callable[[list[float]], Exception],
    unset_error: Callable[[], Exception],
    sum_error: Callable[[float], Exception],
) -> Optional[Exception]:
    """
    Return the first validation error for a probability row, if any.

    Validation order is:
    1) numeric + finite
    2) invalid negative values (anything < 0 except UNSET_VALUE)
    3) unset values (UNSET_VALUE present)
    4) row sum equals 1 within tolerance


    Args:
        row_vals (Union[pd.Series, pd.DataFrame]): The values in the row to validate.
        context (str): Description of the values for error messages.
        invalid_negative_error (Callable[[list[float]], Exception]): Function that takes a list of
            invalid negative values and returns an Exception to raise.
        unset_error (Callable[[], Exception]): Function that returns an Exception to raise when
            unset sentinel values are present.
        sum_error (Callable[[float], Exception]): Function that takes the row sum and returns an
            Exception to raise when the sum does not equal 1 within tolerance.

    Returns:
        Optional[Exception]: The first validation error encountered, or None if no errors.
    """
    try:
        vals = _to_finite_float_array(row_vals, context=context)
    except (TypeError, ValueError) as e:
        return e

    invalid_negatives = _invalid_negative_values(vals)
    if invalid_negatives:
        return invalid_negative_error(invalid_negatives)

    if (vals < 0).any():
        return unset_error()

    total = float(vals.sum())
    if _sum_differs_from_one(total):
        return sum_error(total)

    return None


def typecheck_bloc_proportion_mapping(
    bloc_prop_mapping: BlocProportionMapping,
) -> None:
    """
    Checks to make sure that the values that are stored in params are of the expected type.

    Args:
        bloc_prop_mapping (BlocProportionMapping): The bloc proportion mapping to check.

    Raises:
        TypeError: If params is not a Mapping[str, float] or pd.Series with string index
            and numeric dtype.
        ValueError: If params contains non-finite values.
    """
    if isinstance(bloc_prop_mapping, pd.Series):
        ser = bloc_prop_mapping
        if not all(isinstance(i, str) for i in ser.index):
            raise TypeError("Bloc keys must be a 'str'.")
        if not pd.api.types.is_numeric_dtype(ser.dtype):
            raise TypeError("Bloc proportions must be numeric.")
        if not np.isfinite(ser.to_numpy()).all():
            raise ValueError("Bloc proportions contain non-finite values.")
        return

    if not isinstance(cast(object, bloc_prop_mapping), Mapping):  # keep Pyright happy
        raise TypeError(
            f"Bloc proportions must be a mapping or a dataframe, got '{type(bloc_prop_mapping).__name__}'"
        )

    for bloc, v in bloc_prop_mapping.items():
        if not isinstance(bloc, str):
            raise TypeError(f"Bloc keys must be a 'str', got '{bloc!r}' of '{type(bloc).__name__}'")
        if not _is_finite_real(v):
            raise TypeError(
                f"Bloc '{bloc!r}': proportion must be a finite real (int|float), got '{v!r}' of "
                f"type '{type(v).__name__}'"
            )


def convert_bloc_proportion_map_to_series(
    bloc_prop_mapping: BlocProportionMapping,
) -> pd.Series:
    """
    Convert a dictionary of bloc proportions to a Series.

    Args:
        bloc_prop_mapping (BlocProportionMapping): The bloc proportion mapping to convert.

    Returns:
        pd.Series: A pandas Series with bloc names as the index and proportions as values.

    Raises:
        TypeError: If bloc_prop is not a Mapping[str, float] or pd.Series with string index
            and numeric dtype.
        ValueError: If bloc_prop contains non-finite values, negative values, or does not sum to 1.
    """
    typecheck_bloc_proportion_mapping(bloc_prop_mapping)

    # basically a no_op if already a Series
    if isinstance(bloc_prop_mapping, pd.Series):
        if len(set(bloc_prop_mapping.index)) != len(bloc_prop_mapping.index):
            raise ValueError("Bloc proportions index (blocs) contains duplicates.")
        if bloc_prop_mapping.dtype != float:
            bloc_prop_mapping = bloc_prop_mapping.astype(float)
        values = bloc_prop_mapping.to_numpy(dtype=float)
        if np.any(values < 0):
            raise ValueError("Bloc proportions must be non-negative.")
        if np.any(values > 1):
            raise ValueError("Bloc proportions cannot be greater than 1.")
        return bloc_prop_mapping

    bloc_series = pd.Series(bloc_prop_mapping)

    if any(bloc_series < 0):
        raise ValueError("Bloc proportions must be non-negative.")
    if _sum_differs_from_one(bloc_series.sum()):
        raise ValueError(
            "Bloc proportions currently sum to "
            f"{bloc_series.sum():0.6f} when they should sum to 1 within tolerance "
            f"{FLOAT_TOL:g}."
        )

    # Quick normalize in case of fp errors
    bloc_series = bloc_series / bloc_series.sum()
    return bloc_series


def typecheck_cohesion_mapping(cohesion_mapping: CohesionMapping) -> None:
    """
    Raise TypeError if 'params' is not a mapping of the expected nested shape.

    Args:
        cohesion_mapping (CohesionMapping): The cohesion mapping to check.

    Raises:
        TypeError: If params is not a Mapping[str, Mapping[str, float]] or pd.DataFrame
            with string index and float dtypes.
        ValueError: If params contains non-finite values.
    """

    if isinstance(cohesion_mapping, pd.DataFrame):
        df = cohesion_mapping
        if not all(isinstance(c, str) for c in df.columns):
            raise TypeError("cohesion_df columns (blocs) must be a 'str'.")
        if not all(isinstance(i, str) for i in df.index):
            raise TypeError("cohesion_df index (slates) must be a 'str'.")
        if not all(pd.api.types.is_float_dtype(dt) for dt in df.dtypes):
            raise TypeError("cohesion_df must have float dtypes in every column.")
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError("cohesion_df contains non-finite values.")
        return

    if not isinstance(cast(object, cohesion_mapping), Mapping):  # keep Pyright happy
        raise TypeError(
            f"Cohesion parameters must be a mapping, got '{type(cohesion_mapping).__name__}'"
        )

    for bloc, inner in cohesion_mapping.items():
        if not isinstance(bloc, str):
            raise TypeError(f"Bloc keys must be a 'str', got '{type(bloc).__name__}'")
        if not isinstance(inner, Mapping):
            raise TypeError(
                f"Bloc '{bloc!r}' value must be a mapping, got '{type(inner).__name__}'"
            )

        for slate, v in inner.items():
            if not isinstance(slate, str):
                raise TypeError(
                    f"In bloc '{bloc!r}': slate keys must be a 'str', got '{type(slate).__name__}'"
                )
            if not _is_finite_real(v):
                raise TypeError(
                    f"In bloc '{bloc!r}', slate '{slate!r}': value must be a finite real "
                    f"(int|float), got '{v!r}' of type '{type(v).__name__}'"
                )


def convert_cohesion_map_to_cohesion_df(
    cohesion_mapping: CohesionMapping,
) -> pd.DataFrame:
    """
    Convert a dictionary of cohesion parameters to a DataFrame to pass to BlocSlateConfig.

    Args:
        cohesion_mapping (CohesionMapping): The cohesion mapping to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame with blocs as the index and slates as columns.

    Raises:
        TypeError: If cohesion_map is not a Mapping[str, Mapping[str, float]] or pd.DataFrame
            with string index and float dtypes.
        ValueError: If cohesion_map contains non-finite values.
        ValueError: If cohesion_map contains duplicate blocs or slates.
    """
    typecheck_cohesion_mapping(cohesion_mapping)

    # basically a no_op if already a DataFrame
    if isinstance(cohesion_mapping, pd.DataFrame):
        ret = cohesion_mapping.copy()
        if len(set(ret.index)) != len(ret.index):
            raise ValueError("cohesion_df index (blocs) contains duplicates.")
        if len(set(ret.columns)) != len(ret.columns):
            raise ValueError("cohesion_df columns (slates) contains duplicates.")
        return ret

    blocs_to_slate: MutableMapping[str, MutableMapping[str, float]] = {
        bloc: {} for bloc in cohesion_mapping
    }

    for bloc, slate_dict in cohesion_mapping.items():
        slate_series = pd.Series(slate_dict)
        blocs_to_slate[bloc].update(
            {str(slate): float(value) for slate, value in slate_series.items()}
        )

    return pd.DataFrame(blocs_to_slate).fillna(UNSET_VALUE).T


def typecheck_preference(pref_mapping: PreferenceMapping) -> None:
    """
    Raise TypeError if 'pref_mapping' is not a mapping of the expected nested shape.

    Args:
        pref_mapping (PreferenceMapping): The preference mapping to check.

    Raises:
        TypeError: If pref_mapping is not a Mapping[str, Mapping[str, PreferenceIntervalLike]]
            or pd.DataFrame with string index and numeric dtypes. Note that PreferenceIntervalLike
            is either a Mapping[str, float|int] or PreferenceInterval.
        ValueError: If pref_mapping contains non-finite values.
    """
    if isinstance(pref_mapping, pd.DataFrame):
        df = pref_mapping
        if not all(isinstance(c, str) for c in df.columns):
            raise TypeError("preference_df columns (candidates) must be a 'str'.")
        if not all(isinstance(i, str) for i in df.index):
            raise TypeError("preference_df index (blocs) must be a 'str'.")
        if not all(pd.api.types.is_numeric_dtype(dt) for dt in df.dtypes):
            raise TypeError("preference_df columns must be numeric.")
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError("preference_df contains non-finite values.")
        return

    if not isinstance(cast(object, pref_mapping), Mapping):  # cast gets around Pyright
        raise TypeError(f"preference_dict must be a mapping, got '{type(pref_mapping).__name__}'")

    for bloc, slate_dict in pref_mapping.items():
        if not isinstance(bloc, str):
            raise TypeError(f"Bloc keys must be a 'str', got '{type(bloc).__name__}'")
        if not isinstance(slate_dict, Mapping):
            raise TypeError(
                f"Value for bloc '{bloc!r}' must be a mapping, got '{type(slate_dict).__name__}'"
            )

        for slate, item in slate_dict.items():
            if not isinstance(slate, str):
                raise TypeError(
                    f"In bloc '{bloc!r}': slate keys must be a 'str', got '{type(slate).__name__}'"
                )

            if isinstance(item, Mapping):
                for name, score in item.items():
                    if not isinstance(name, str):
                        raise TypeError(
                            f"In bloc '{bloc!r}', slate '{slate!r}': "
                            f"candidate names must be a 'str', got '{type(name).__name__}'"
                        )
                    if not _is_finite_real(score):
                        raise TypeError(
                            f"In bloc '{bloc!r}', slate '{slate!r}', candidate '{name!r}': "
                            f"score must be a finite real (int|float), got '{score!r}'"
                        )

            elif isinstance(item, PreferenceInterval):
                interval = item.interval
                for name, score in interval.items():
                    if not isinstance(name, str):
                        raise TypeError(
                            f"In bloc '{bloc!r}', slate '{slate!r}': candidate names must be "
                            f"a 'str', got '{type(name).__name__}'"
                        )
                    if not _is_finite_real(score):
                        raise TypeError(
                            f"In bloc '{bloc!r}', slate '{slate!r}', candidate '{name!r}': "
                            f"score must be a finite real (int|float), got '{score!r}'"
                        )
            else:
                raise TypeError(
                    f"In bloc '{bloc!r}', slate '{slate!r}': expected Mapping[str, float|int] "
                    f"or PreferenceInterval, got '{type(item).__name__}'"
                )


def convert_preference_map_to_preference_df(
    preference_mapping: PreferenceMapping,
) -> pd.DataFrame:
    """
    Convert a dictionary of preference mappings to a DataFrame to pass to BlocSlateConfig.

    Args:
        preference_mapping (PreferenceMapping): The preference mapping to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame with blocs as the index and candidates as columns.

    Raises:
        TypeError: If preference_map is not a Mapping[str, Mapping[str, PreferenceIntervalLike]]
            or pd.DataFrame with string index and numeric dtypes. Note that PreferenceIntervalLike
            is either a Mapping[str, float|int] or PreferenceInterval.

        ValueError: If preference_map contains non-finite values.
        ValueError: If preference_map contains duplicate blocs or candidates.
    """
    typecheck_preference(preference_mapping)

    # basically a no_op if already a DataFrame
    if isinstance(preference_mapping, pd.DataFrame):
        if len(set(preference_mapping.index)) != len(preference_mapping.index):
            raise ValueError("preference_df index (blocs) contains duplicates.")
        if len(set(preference_mapping.columns)) != len(preference_mapping.columns):
            raise ValueError("preference_df columns (candidates) contains duplicates.")
        return preference_mapping

    blocs_to_cand: MutableMapping[str, MutableMapping[str, float]] = {
        bloc: {} for bloc in preference_mapping
    }
    for bloc, slate_dict in preference_mapping.items():
        for cand_item in slate_dict.values():
            cand_map = (
                cand_item.interval if isinstance(cand_item, PreferenceInterval) else cand_item
            )
            cand_series = pd.Series(cand_map)
            blocs_to_cand[bloc].update(
                {str(candidate): float(value) for candidate, value in cand_series.items()}
            )

    return pd.DataFrame(blocs_to_cand).fillna(UNSET_VALUE).T
