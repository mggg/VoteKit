"""
This file only contains the model for BlocSlateConfig and related helper classes and functions.

BlocSlateConfig is a configuration object for BlocSlateGenerator that holds all the
parameters needed to generate a set of ballots using one of our ballot generation algorithms
involving both voter blocs and slates of candidates.
"""

from collections.abc import Mapping, MutableMapping, MutableSequence, Iterator
from typing import Sequence, Optional, Union, cast, Any, overload, Iterable
from votekit.pref_interval import combine_preference_intervals, PreferenceInterval
from numbers import Real
import weakref
import math
import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
from warnings import warn
from pprint import pformat
from textwrap import indent


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


warnings.formatwarning = _config_warning_format


BlocPropotionMapping = Union[Mapping[str, Union[int, float]], pd.Series]
CohesionMapping = Union[
    Mapping[str, Union[Mapping[str, float], pd.Series]], pd.DataFrame
]
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


def typecheck_bloc_proportion_mapping(
    params: BlocPropotionMapping,
) -> None:
    """
    Checks to make sure that the values that are stored in params are of the expected type.

    Args:
        params (BlocPropotionMapping): The bloc proportion mapping to check.

    Raises:
        TypeError: If params is not a Mapping[str, float] or pd.Series with string index
            and numeric dtype.
        ValueError: If params contains non-finite values.
    """
    if isinstance(params, pd.Series):
        ser = params
        if not all(isinstance(i, str) for i in ser.index):
            raise TypeError("Bloc keys must be a 'str'.")
        if not pd.api.types.is_numeric_dtype(ser.dtype):
            raise TypeError("Bloc proportions must be numeric.")
        if not np.isfinite(ser.to_numpy()).all():
            raise ValueError("Bloc proportions contain non-finite values.")
        return

    if not isinstance(cast(object, params), Mapping):  # keep Pyright happy
        raise TypeError(
            f"Bloc proportions must be a mapping or a dataframe, got '{type(params).__name__}'"
        )

    for bloc, v in params.items():
        if not isinstance(bloc, str):
            raise TypeError(
                f"Bloc keys must be a 'str', got '{bloc!r}' of '{type(bloc).__name__}'"
            )
        if not _is_finite_real(v):
            raise TypeError(
                f"Bloc '{bloc!r}': proportion must be a finite real (int|float), got '{v!r}' of "
                f"type '{type(v).__name__}'"
            )


def convert_bloc_proportion_map_to_series(
    bloc_prop: BlocPropotionMapping,
) -> pd.Series:
    """
    Convert a dictionary of bloc proportions to a Series.

    Args:
        bloc_prop (BlocPropotionMapping): The bloc proportion mapping to convert.

    Returns:
        pd.Series: A pandas Series with bloc names as the index and proportions as values.

    Raises:
        TypeError: If bloc_prop is not a Mapping[str, float] or pd.Series with string index
            and numeric dtype.
        ValueError: If bloc_prop contains non-finite values, negative values, or does not sum to 1.
    """
    typecheck_bloc_proportion_mapping(bloc_prop)

    # basically a no_op if already a Series
    if isinstance(bloc_prop, pd.Series):
        if len(set(bloc_prop.index)) != len(bloc_prop.index):
            raise ValueError("Bloc proportions index (blocs) contains duplicates.")
        if bloc_prop.dtype != float:
            bloc_prop = bloc_prop.astype(float)
        if any(bloc_prop < 0):
            raise ValueError("Bloc proportions must be non-negative.")
        if any(bloc_prop > 1):
            raise ValueError("Bloc proportions cannot be greater than 1.")
        return bloc_prop

    bloc_series = pd.Series(bloc_prop)

    if any(bloc_series < 0):
        raise ValueError("Bloc proportions must be non-negative.")
    if abs(bloc_series.sum() - 1) > 1e-8:
        raise ValueError(
            f"Bloc proportions currently sum to {bloc_series.sum():0.6f} when they "
            "should sum to 1."
        )

    # Quick normalize in case of fp errors
    bloc_series = bloc_series / bloc_series.sum()
    return bloc_series


def typecheck_cohesion_mapping(params: CohesionMapping) -> None:
    """
    Raise TypeError if 'params' is not a mapping of the expected nested shape.

    Args:
        params (CohesionMapping): The cohesion mapping to check.

    Raises:
        TypeError: If params is not a Mapping[str, Mapping[str, float]] or pd.DataFrame
            with string index and float dtypes.
        ValueError: If params contains non-finite values.
    """

    if isinstance(params, pd.DataFrame):
        df = params
        if not all(isinstance(c, str) for c in df.columns):
            raise TypeError("cohesion_df columns (blocs) must be a 'str'.")
        if not all(isinstance(i, str) for i in df.index):
            raise TypeError("cohesion_df index (slates) must be a 'str'.")
        if not all(pd.api.types.is_float_dtype(dt) for dt in df.dtypes):
            raise TypeError("cohesion_df must have float dtypes in every column.")
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError("cohesion_df contains non-finite values.")
        return

    if not isinstance(cast(object, params), Mapping):  # keep Pyright happy
        raise TypeError(
            f"Cohesion parameters must be a mapping, got '{type(params).__name__}'"
        )

    for bloc, inner in params.items():
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
    cohesion_map: CohesionMapping,
) -> pd.DataFrame:
    """
    Convert a dictionary of cohesion parameters to a DataFrame to pass to BlocSlateConfig.

    Args:
        cohesion_map (CohesionMapping): The cohesion mapping to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame with blocs as the index and slates as columns.

    Raises:
        TypeError: If cohesion_map is not a Mapping[str, Mapping[str, float]] or pd.DataFrame
            with string index and float dtypes.
        ValueError: If cohesion_map contains non-finite values.
        ValueError: If cohesion_map contains duplicate blocs or slates.
    """
    typecheck_cohesion_mapping(cohesion_map)

    # basically a no_op if already a DataFrame
    if isinstance(cohesion_map, pd.DataFrame):
        ret = cohesion_map.copy()
        if len(set(ret.index)) != len(ret.index):
            raise ValueError("cohesion_df index (blocs) contains duplicates.")
        if len(set(ret.columns)) != len(ret.columns):
            raise ValueError("cohesion_df columns (slates) contains duplicates.")
        return ret

    blocs_to_slate: MutableMapping[str, MutableMapping[str, float]] = {
        bloc: {} for bloc in cohesion_map
    }

    for bloc, slate_dict in cohesion_map.items():
        slate_series = pd.Series(slate_dict)
        blocs_to_slate[bloc].update(slate_series.to_dict())

    return pd.DataFrame(blocs_to_slate).fillna(-1.0).T


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
        raise TypeError(
            f"preference_dict must be a mapping, got '{type(pref_mapping).__name__}'"
        )

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
    preference_map: PreferenceMapping,
) -> pd.DataFrame:
    """
    Convert a dictionary of preference mappings to a DataFrame to pass to BlocSlateConfig.

    Args:
        preference_map (PreferenceMapping): The preference mapping to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame with blocs as the index and candidates as columns.

    Raises:
        TypeError: If preference_map is not a Mapping[str, Mapping[str, PreferenceIntervalLike]]
            or pd.DataFrame with string index and numeric dtypes. Note that PreferenceIntervalLike
            is either a Mapping[str, float|int] or PreferenceInterval.

        ValueError: If preference_map contains non-finite values.
        ValueError: If preference_map contains duplicate blocs or candidates.
    """
    typecheck_preference(preference_map)

    # basically a no_op if already a DataFrame
    if isinstance(preference_map, pd.DataFrame):
        if len(set(preference_map.index)) != len(preference_map.index):
            raise ValueError("preference_df index (blocs) contains duplicates.")
        if len(set(preference_map.columns)) != len(preference_map.columns):
            raise ValueError("preference_df columns (candidates) contains duplicates.")
        return preference_map

    blocs_to_cand: MutableMapping[str, MutableMapping[str, float]] = {
        bloc: {} for bloc in preference_map
    }
    for bloc, slate_dict in preference_map.items():
        seen_candidates: set[str] = set()
        for cand_item in slate_dict.values():
            cand_map = (
                cand_item.interval
                if isinstance(cand_item, PreferenceInterval)
                else cand_item
            )
            cand_series = pd.Series(cand_map)
            seen_candidates.update(list(cand_series.index))

            blocs_to_cand[bloc].update(cand_series.to_dict())

    return pd.DataFrame(blocs_to_cand).fillna(0.0).T


class _CandListProxy(MutableSequence[str]):
    """
    A proxy for a list of candidates in a slate that routes all changes through the
    owning SlateCandMap to ensure validation.

    Args:
        owner (SlateCandMap): The owning SlateCandMap.
        key (str): The slate name.
    """

    __slots__ = ("__owner", "__key")

    def __init__(self, owner: "SlateCandMap", key: str):
        self.__owner = owner
        self.__key = key

    def __len__(self) -> int:
        return len(self.__owner._data[self.__key])

    @overload
    def __getitem__(self, i: int) -> str: ...

    @overload
    def __getitem__(self, i: slice) -> MutableSequence[str]: ...

    def __getitem__(self, i: Union[int, slice]) -> Union[str, MutableSequence[str]]:
        data = self.__owner._data[self.__key]
        if isinstance(i, slice):
            return data[i]
        return data[i]

    @overload
    def __setitem__(self, i: int, v: str) -> None: ...

    @overload
    def __setitem__(self, i: slice, v: Iterable[str]) -> None: ...

    def __setitem__(self, i: Union[int, slice], v: Union[str, Iterable[str]]) -> None:
        new = list(self.__owner._data[self.__key])
        if isinstance(i, slice):  # pragma: no cover
            if isinstance(v, (str, bytes)) or not isinstance(v, Iterable):
                raise TypeError("Slice assignment requires an iterable of str")
            new[i] = [str(x) for x in v]
        else:
            new[i] = str(v)
        self.__owner[self.__key] = new

    @overload
    def __delitem__(self, i: int) -> None: ...

    @overload
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: Union[int, slice]) -> None:
        new = list(self.__owner._data[self.__key])
        del new[i]
        self.__owner[self.__key] = new

    def _current(self) -> list[str]:
        return list(self.__owner._data[self.__key])

    def insert(self, i: int, v: str) -> None:
        """Insert candidate v at index i if not already present."""
        if not isinstance(cast(object, v), str):
            raise TypeError("Slate candidates must be a 'str'")
        if not isinstance(cast(object, i), int):
            raise TypeError("Index must be an 'int'")
        new = list(self.__owner._data[self.__key])
        if v not in new:
            new.insert(i, str(v))
        self.__owner[self.__key] = new

    def extend(self, it: Iterable[str]) -> None:
        """Extend the candidate list by appending elements from the iterable."""
        rollback = self._current().copy()
        try:
            for v in it:
                self.insert(len(self), v)

        except Exception as e:
            self.__owner[self.__key] = rollback
            raise e

    def __iadd__(self, it: Iterable[str]) -> "_CandListProxy":
        self.extend(it)
        return self

    def append(self, v: str) -> None:
        """Append candidate v to the end of the list if not already present."""
        self.insert(len(self), v)

    def sort(self) -> None:
        """Sort the candidate list in place."""
        new = self._current()
        new.sort()
        self.__owner[self.__key] = new

    def __eq__(self, other: Union[Sequence[str], Any]):
        if not isinstance(other, Sequence):
            return False
        if len(self) != len(other):
            return False
        for v1, v2 in zip(self._current(), other):
            if v1 != v2:
                return False

        return True

    def __repr__(self) -> str:  # pragma: no cover
        return str(self._current())


class SlateCandMap(MutableMapping[str, Sequence[str]]):
    """
    Mapping[str, Sequence[str]] that enforces slate to candidate list rules:

    - Each slate must have a non-empty list of candidates
    - No candidate may appear in more than one slate
    - Allows item assignment with warnings if candidates are duplicated
    - Routes candidate list mutations through a proxy to ensure validation

    Args:
        parent (BlocSlateConfig): The owning BlocSlateConfig.
        init (Optional[Mapping[str, Sequence[str]]]): Initial mapping of slate names to
            sequences of candidate names. If None, defaults to an empty mapping.
    """

    __slots__ = ("__parent", "_data")

    def __init__(
        self,
        parent: "BlocSlateConfig",
        init: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        try:
            self.__parent = weakref.proxy(parent)
        except TypeError:
            # parent is already a weakref.ProxyType
            self.__parent = parent
        self._data: dict[str, list[str]] = {}
        if init is not None:
            try:
                for k, v in init.items():
                    if len(v) == 0:
                        raise ValueError(
                            f"Slate '{k}' has empty candidate list. "
                            "Candidate lists must be non-empty."
                        )
                    self._data.update({k: [str(c) for c in v]})
            except AttributeError as e:
                raise AttributeError(
                    f"SlateCandMap 'init' variable is of type '{type(init).__name__}' which "
                    "does not implement the '.items()' method."
                ) from e

    def to_dict(self) -> dict[str, list[str]]:
        """
        Return a deep copy of the internal slate to candidates mapping as a standard dict.

        Returns:
            dict[str, list[str]]: A deep copy of the internal mapping.
        """
        return {k: deepcopy(v) for k, v in self._data.items()}

    def __getitem__(self, key: str) -> _CandListProxy:
        return _CandListProxy(self, key)

    def __setitem__(self, key: str, value: Sequence[str]) -> None:
        if not isinstance(cast(object, key), str):
            raise TypeError("Slate name must be a 'str'")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("Slate candidates must be a sequence of str")
        if len(value) == 0:
            raise ValueError(
                f"Slate '{key}' has empty candidate list. Candidate lists must be non-empty."
            )
        val_list = [str(c) for c in value]

        # Prevent adding candidates that already exist in *other* slates
        existing_cands = set(self.__parent.candidates)
        existing_cands -= set(self._data.get(key, ()))  # allow replacing same slate
        clashing_candidates = existing_cands.intersection(val_list)
        if clashing_candidates == set() or clashing_candidates.issubset(
            set(self._data.get(key, ()))
        ):
            rollback = self._data.get(key, None)
            rollback_slate_dict = (
                self.__parent._current_preference_df_slate_cand_mapping
            )
            self._data[key] = val_list
            try:
                self.__parent._update_preference_and_cohesion_slates()
            except KeyError as e:
                if rollback is None:
                    del self._data[key]
                else:
                    self._data[key] = rollback

                self.__parent._current_preference_df_slate_cand_mapping = (
                    rollback_slate_dict
                )
                raise KeyError(
                    f"{e.args[0]}. "
                    "You may have tried to modify the candidate list directly. "
                    "Please modify the entire slate at once instead using "
                    "config.slate_to_candidates[slate] = [...]. "
                    "If renaming a candidate, please use config.rename_candidates({...}) "
                ) from e
            return

        clash_cand_keys = []
        clash_cand_list = list(clashing_candidates)
        for k, v_list in self._data.items():
            for v in clash_cand_list:
                if v in v_list:
                    clash_cand_keys.append(k)
        if len(clash_cand_keys) != 0:
            raise ValueError(
                f"Candidates {clash_cand_list} already exist in slates {clash_cand_keys}"
            )

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:  # noqa
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def update(self, other=(), /, **kw) -> None:
        items = (
            dict(other, **kw).items()
            if not isinstance(other, Mapping)
            else other.items()
        )
        for k, v in items:
            self[k] = v  # route through __setitem__

    def __or__(self, other: Mapping[str, Sequence[str]]) -> "SlateCandMap":
        new = SlateCandMap(self.__parent, self._data)
        new.update(other)
        return new

    def __ror__(self, other: Mapping[str, Sequence[str]]) -> "SlateCandMap":
        full_map = dict(other) | self._data.copy()
        return SlateCandMap(self.__parent, full_map)

    def __ior__(self, other: Mapping[str, Sequence[str]]) -> "SlateCandMap":
        self.update(other)
        return self

    def __repr__(self) -> str:  # pragma: no cover
        return pformat(self._data, indent=2, width=40, sort_dicts=False, compact=True)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MutableMapping):
            return False

        for k1, v1 in self._data.items():
            if k1 not in other or other[k1] != v1:
                return False

        for k2, v2 in other.items():
            if k2 not in self._data or self._data[k2] != v2:
                return False

        return True

    def copy(self):
        """
        Return a deep copy of the internal slate to candidates mapping.

        Returns:
            dict[str, list[str]]: A deep copy of the internal mapping.
        """
        return deepcopy(self._data)


class BlocProportions(MutableMapping[str, float]):
    """
    Mapping[str, float] that enforces bloc proportion rules:
    - Each bloc must have a non-negative proportion
    - The proportions must sum to 1

    Args:
        parent (BlocSlateConfig): The owning BlocSlateConfig.
        init (Optional[BlocPropotionMapping]): Initial mapping of bloc names to their
            proportions in the electorate. If None, defaults to an empty mapping.
    """

    __slots__ = ("__parent", "__data")

    def __init__(
        self,
        parent: "BlocSlateConfig",
        init: Optional[BlocPropotionMapping] = None,
    ) -> None:
        self.__parent = weakref.proxy(parent)
        self.__data: dict[str, float] = {}

        if init is not None:
            ser = convert_bloc_proportion_map_to_series(init)
            self.__data.update(ser.to_dict())

        self._validate()

    def _validate(self) -> None:
        """
        Validate that the bloc proportions are non-negative and sum to 1.

        Raises:
            ValueError: If any bloc proportion is negative.
            Warning: If the bloc proportions do not sum to 1 and the parent config is not silent.
        """
        ser = pd.Series(self.__data, dtype=float)

        if (ser < 0).any():  # pragma: no cover
            raise ValueError("Bloc proportions must be non-negative.")

        typecheck_bloc_proportion_mapping(ser)
        total = ser.sum()
        if abs(total - 1.0) > 1e-8:
            if not self.__parent.silent:
                warn(
                    f"Bloc proportions currently sum to {total:.6f} when they should sum to 1.",
                    ConfigurationWarning,
                )

    def __getitem__(self, key: str) -> float:
        return self.__data[key]

    def __setitem__(self, key: str, value: float) -> None:
        raise RuntimeError(
            "Cannot set bloc proportions directly. Please provide a full mapping "
            "and set using config.bloc_proportions = {...}"
        )

    def __delitem__(self, key: str) -> None:
        rollback = self.__data[key]
        del self.__data[key]
        try:
            self._validate()
        except Exception as e:  # pragma: no cover
            self.__data[key] = rollback
            raise e

    def __iter__(self) -> Iterator[str]:  # pragma: no cover  # noqa
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:  # pragma: no cover
        return pformat(self.__data, indent=2, width=40, sort_dicts=False, compact=True)

    def to_series(self) -> pd.Series:
        """
        Return a Series representation of the bloc proportions.

        Returns:
            pd.Series: A pandas Series with bloc names as the index and proportions as values.
        """
        return pd.Series(self.__data, dtype=float)

    def copy(self) -> dict[str, float]:
        """
        Return a deep copy of the internal bloc proportions mapping as a standard dict.

        Returns:
            dict[str, float]: A deep copy of the internal mapping.
        """
        return deepcopy(self.__data)


class BlocSlateConfig:
    """
    Configuration object for BlocSlateGenerator that holds all the parameters needed to
    generate a set of ballots using one of our ballot generation algorithms involving both
    voter blocs and slates of candidates.

    Args:
        n_voters (int): The total number of voters to simulate. Must be > 0.
        slate_to_candidates (Optional[Mapping[str, Sequence[str]]]): A mapping of slate names
            to sequences of candidate names. Each slate must have a non-empty list of candidates,
            and no candidate may appear in more than one slate. If None, defaults to an empty
            mapping.
        bloc_proportions (Optional[BlocPropotionMapping]): A mapping of voter bloc names to
            their proportions in the electorate. The proportions must be non-negative and sum to 1.
            If None, defaults to an empty mapping.
        preference_mapping (Optional[PreferenceMapping]): A nested mapping of voter bloc names
            to slate names to either PreferenceInterval or Mapping[str, float|int] representing
            the preference scores for each candidate in the slate. Each bloc must have a mapping
            for every slate defined in slate_to_candidates, and no candidate may appear in more
            than one slate. The scores for each bloc and slate must be non-negative and sum to 1.
            If None, defaults to an empty mapping.
        cohesion_mapping (Optional[CohesionMapping]): A mapping of voter bloc names to
            mappings of slate names to their cohesion parameters. The cohesion parameters must be
            non-negative and sum to 1 for each bloc. If None, defaults to an empty mapping.
        silent (bool): If True, suppresses warnings about configuration issues. Defaults to False.

    Attributes:
        n_voters (int): The total number of voters to simulate.
        slate_to_candidates (SlateCandMap): A mapping of slate names to sequences of
            names. Behaves like a MutableMapping[str, Sequence[str]] (think dictionary).
        bloc_proportions (BlocProportions): A mapping of voter bloc names to their proportions
            in the electorate. Behaves like a MutableMapping[str, float] (think dictionary).
        preference_df (pd.DataFrame): A DataFrame with blocs as the index and candidates as columns,
            containing the preference scores for each candidate in each bloc.
        cohesion_df (pd.DataFrame): A DataFrame with blocs as the index and slates as columns,
            containing the cohesion parameters for each slate in each bloc.
        silent (bool): If True, suppresses warnings about configuration issues.

    Warns:
        ConfigurationWarning: If there is anything in the configuration that, when passed to
            a ballot generator, would cause an error.

    Raises:
        TypeError: On invalid types for any of the arguments or assignments
            (e.g., non-string keys, non-mapping shapes).
        ValueError: On invalid values (e.g., empty candidate lists, duplicate
            keys, non-finite numbers, negative proportions/scores).
    """

    __slots__ = (
        "n_voters",
        "slate_to_candidates",
        "bloc_proportions",
        "preference_df",
        "cohesion_df",
        "silent",
        "__alphas",
        "__clear_alpha_bool",
        "_current_preference_df_slate_cand_mapping",
        "__weakref__",
    )
    n_voters: int
    slate_to_candidates: SlateCandMap
    bloc_proportions: BlocProportions
    preference_df: pd.DataFrame
    cohesion_df: pd.DataFrame
    silent: bool
    __alphas: Optional[pd.DataFrame]

    # Similar to mutex so are warned about what is writing to preference_df
    __clear_alpha_bool: bool

    # Something to help make sure that the preference_df is updated correctly when the candidates
    # or slates change
    _current_preference_df_slate_cand_mapping: Optional[dict[str, list[str]]]

    def __init__(
        self,
        *,
        n_voters: int,
        slate_to_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        bloc_proportions: Optional[BlocPropotionMapping] = None,
        preference_mapping: Optional[PreferenceMapping] = None,
        cohesion_mapping: Optional[CohesionMapping] = None,
        silent: bool = False,
    ) -> None:
        object.__setattr__(self, "silent", silent)

        object.__setattr__(self, "_BlocSlateConfig__clear_alpha_bool", True)

        object.__setattr__(self, "_current_preference_df_slate_cand_mapping", None)

        self.__validate_voters(n_voters)
        object.__setattr__(self, "n_voters", n_voters)

        if bloc_proportions is None:
            bloc_voter_series: pd.Series = pd.Series(dtype=float)
        else:
            bloc_voter_series = convert_bloc_proportion_map_to_series(bloc_proportions)
        object.__setattr__(
            self, "bloc_proportions", BlocProportions(self, bloc_voter_series)
        )

        if slate_to_candidates is None:
            slate_map = SlateCandMap(self, dict())
        else:
            slate_map = SlateCandMap(self, slate_to_candidates)
        object.__setattr__(self, "slate_to_candidates", slate_map)

        if cohesion_mapping is None:
            cohesion_df = pd.DataFrame()
        else:
            self.__validate_cohesion_df_mapping_keys_ok_in_config(cohesion_mapping)
            cohesion_df = convert_cohesion_map_to_cohesion_df(cohesion_mapping)
        object.__setattr__(self, "cohesion_df", cohesion_df)

        if preference_mapping is None or (
            isinstance(preference_mapping, pd.DataFrame) and preference_mapping.empty
        ):
            preference_df = pd.DataFrame()
        else:
            self.__validate_pref_df_mapping_keys_ok_in_config(preference_mapping)
            preference_df = convert_preference_map_to_preference_df(preference_mapping)
            preference_df = preference_df[self.candidates]  # ensure column order
        object.__setattr__(self, "preference_df", preference_df)

        object.__setattr__(self, "_BlocSlateConfig__alphas", None)

    @property
    def candidates(self) -> list[str]:
        """
        Computed property: A flat list of all candidates in all slate. Derived from the values of
        slate_to_candidates.
        """
        return [c for clist in self.slate_to_candidates.values() for c in clist]

    @property
    def slates(self) -> list[str]:
        """
        Computed property: A list of all slates. Derived from the keys of slate_to_candidates.
        """
        return list(self.slate_to_candidates.keys())

    @property
    def blocs(self) -> list[str]:
        """
        Computed property: A list of all voter blocs. Derived from the keys of bloc_proportions.
        """
        return list(self.bloc_proportions.keys())

    def __validate_voters(self, voters) -> None:
        """
        Validate that the number of voters is a positive integer.

        Args:
            voters (int): The number of voters to validate.

        Raises:
            TypeError: If voters is not cleanly convertible to an int.
            ValueError: If voters is not > 0.
        """
        if not int(voters) == voters:
            raise TypeError("Number of voters must be cleanly convertible to an int.")
        if voters <= 0:
            raise ValueError("Number of voters must be > 0.")

    def __validate_pref_df_mapping_keys_ok_in_config(
        self, pref_mapping: PreferenceMapping
    ) -> bool:
        """
        Validate that the keys in the preference mapping are compatible with the
        current configuration.

        Args:
            pref_mapping (PreferenceMapping): The preference mapping to validate.

        Returns:
            bool: True if the keys are compatible, False otherwise.

        Raises:
            TypeError: If pref_mapping is not a Mapping[str, Mapping[str, PreferenceIntervalLike]]
                or pd.DataFrame with string index and numeric dtypes. Note that
                PreferenceIntervalLike is either a Mapping[str, float|int] or PreferenceInterval.
            ValueError: If pref_mapping contains non-finite values.

        Warns:
            ConfigurationWarning: If there are any issues with the configuration that would
                cause an error when passed to a ballot generator.
        """
        messages: list[str] = []

        blocs: set[str]
        candidates: set[str]

        if isinstance(pref_mapping, pd.DataFrame):
            blocs = {str(x) for x in pref_mapping.index.tolist()}
            candidates = {str(x) for x in pref_mapping.columns.tolist()}
        else:
            blocs = set(pref_mapping.keys())
            candidates = set()
            for bloc, slate_dict in pref_mapping.items():
                slate_cand_set: set[str] = set()
                slate_set: set[str] = set(slate_dict.keys())
                if slate_set != set(self.slates):
                    messages.append(
                        f"Preference mapping for bloc '{bloc}' has slates "
                        f"{sorted(list(slate_set))} but "
                        f"config has slates {sorted(self.slates)}"
                    )
                for slate, preference_like in slate_dict.items():
                    if not isinstance(preference_like, (Mapping, PreferenceInterval)):
                        raise TypeError(
                            f"Preference mapping for bloc '{bloc}', slate '{slate}' "
                            f"must be Mapping[str, float|int] or PreferenceInterval, "
                            f"got '{type(preference_like).__name__}'"
                        )

                    if isinstance(preference_like, PreferenceInterval):
                        new_cands = set(preference_like.interval.keys())
                    else:
                        new_cands = set(preference_like.keys())

                    if new_cands.intersection(slate_cand_set) != set():
                        raise ValueError(
                            f"Preference interval for bloc '{bloc}' and slate '{slate}' has "
                            f"candidates {sorted(list(new_cands.intersection(slate_cand_set)))} "
                            f"which appear in other slates in the same bloc."
                        )

                    slate_cand_set.update(new_cands)
                candidates.update(slate_cand_set)

        if set(blocs) != set(self.blocs):
            if self.blocs == []:
                messages.append(
                    "Preference mapping has voter blocs but no blocs are defined in "
                    "bloc_proportions."
                )
            else:
                messages.append(
                    f"Preference mapping expected exactly the blocs "
                    f"{sorted(self.blocs)}, got {sorted(list(blocs))}"
                )
        if set(candidates) != set(self.candidates):
            if self.candidates == []:
                messages.append(
                    "Preference mapping has candidates but no candidates are defined "
                    "in slate_to_candidates."
                )
            else:
                if set(self.candidates).issubset(set(candidates)):
                    messages.append(
                        f"Preference contains candidates not present in slate_to_candidates. "
                        f"Ignoring: {set(candidates) - set(self.candidates)}"
                    )
                else:
                    messages.append(
                        f"Preference mapping expected exactly the candidates "
                        f"'{sorted(self.candidates)}', got {sorted(list(candidates))}"
                    )

        valid = len(messages) == 0
        if not self.silent:
            for msg in messages:
                warn(msg, ConfigurationWarning)

        return valid

    def __validate_cohesion_df_mapping_keys_ok_in_config(
        self, cohesion_mapping: CohesionMapping
    ) -> bool:
        """
        Validate that the keys in the cohesion mapping are compatible with the
        current configuration.

        Args:
            cohesion_mapping (CohesionMapping): The cohesion mapping to validate.

        Returns:
            bool: True if the keys are compatible, False otherwise.

        Raises:
            TypeError: If cohesion_mapping is not a Mapping[str, Mapping[str, float]]
                or pd.DataFrame with string index and float dtypes.
            ValueError: If cohesion_mapping contains non-finite values.
            ValueError: If cohesion_mapping contains duplicate blocs or slates.

        Warns:
            ConfigurationWarning: If there are any issues with the configuration that would
                cause an error when passed to a ballot generator.
        """

        messages: list[str] = []

        blocs: set[str]
        slates: set[str]

        if isinstance(cohesion_mapping, pd.DataFrame):
            blocs = {str(x) for x in cohesion_mapping.index.tolist()}
            slates = {str(x) for x in cohesion_mapping.columns.tolist()}
        else:
            blocs = set(cohesion_mapping.keys())
            slates = set()
            for bloc, slate_dict in cohesion_mapping.items():
                slate_set = set(slate_dict.keys())
                if slate_set != set(self.slates):
                    messages.append(
                        f"Cohesion mapping for bloc '{bloc}' has slates "
                        f"{sorted(list(slate_set))} but "
                        f"config has slates {sorted(self.slates)}"
                    )
                slates.update(slate_set)

        if set(blocs) != set(self.blocs):
            if self.blocs == []:
                messages.append(
                    "Cohesion mapping has voter blocs but no blocs are defined in "
                    "bloc_proportions."
                )
            else:
                messages.append(
                    f"Cohesion mapping expected exactly the blocs "
                    f"{sorted(self.blocs)}, got {sorted(list(blocs))}"
                )

        if set(slates) != set(self.slates):
            if self.slates == []:
                messages.append(
                    "Cohesion mapping has slates but no slates are defined in "
                    "slate_to_candidates."
                )
            else:
                messages.append(
                    f"Cohesion mapping expected exactly the slates "
                    f"'{sorted(self.slates)}', got {sorted(list(slates))}"
                )
        valid = len(messages) == 0

        if not self.silent:
            for msg in messages:
                warn(msg, ConfigurationWarning)

        return valid

    def __determine_errors(self) -> list[Exception]:
        """
        Determine if there are any errors in the current configuration that would
        cause an error when passed to a ballot generator.

        Returns:
            list[Exception]: A list of exceptions representing the errors found.
        """
        errors: list[Exception] = []

        if self.n_voters < len(self.bloc_proportions):
            errors.append(
                ValueError(
                    f"Number of voters ({self.n_voters}) must be >= number of blocs "
                    f"({len(self.bloc_proportions)})."
                )
            )

        if self.bloc_proportions == {}:
            errors.append(ValueError("At least one voter bloc must be defined."))
        else:
            # Check that bloc proportions sum to 1
            bloc_ser = self.bloc_proportions.to_series()
            if abs(bloc_ser.sum() - 1.0) > 1e-8:
                errors.append(
                    ValueError(
                        f"Bloc proportions currenlty sum to {bloc_ser.sum():.6f} "
                        "when they should sum to 1."
                    )
                )

        if any(self.bloc_proportions.to_series() <= 0):
            for block, prop in self.bloc_proportions.items():
                if prop <= 0:
                    errors.append(
                        ValueError(
                            f"Bloc '{block}' has non-positive proportion {prop:.6f}."
                        )
                    )

        if self.slate_to_candidates == {}:
            errors.append(
                ValueError("At least one slate and candidate list must be defined.")
            )

        if self.preference_df.empty:
            errors.append(ValueError("Preference mapping must be non-empty."))
        else:
            if set(self.preference_df.columns) != set(self.candidates):
                errors.append(
                    KeyError(
                        f"preference_df columns (candidates) must be exactly "
                        f"{list(self.candidates)}, as defined in the 'slate_to_candidates' "
                        f"parameter. Got {list(self.preference_df.columns)}"
                    )
                )
            else:
                for cand_list_proxy in self.slate_to_candidates.values():
                    cand_list = list(cand_list_proxy)
                    for row in self.preference_df[cand_list].iterrows():
                        if any(row[1] < 0):
                            errors.append(
                                ValueError(
                                    f"preference_df row for bloc '{row[0]}' has values that have "
                                    f"not been set (indicated with value of -1)."
                                )
                            )
                        elif abs(row[1].sum() - 1.0) > 1e-8:
                            errors.append(
                                ValueError(
                                    f"preference_df row for bloc '{row[0]}' and candidates "
                                    f"{cand_list} must sum to 1, got {row[1].sum():.6f}"
                                )
                            )

            if set(self.preference_df.index) != set(self.blocs):
                errors.append(
                    KeyError(
                        f"preference_df index (blocs) must be exactly {list(self.blocs)} "
                        f"as defined in the 'bloc_proportions' parameter. Got"
                        f"{list(self.preference_df.index)}"
                    )
                )

        if self.cohesion_df.empty:
            errors.append(ValueError("Cohesion mapping must be non-empty."))
        else:
            if set(self.cohesion_df.columns) != set(self.slates):
                errors.append(
                    KeyError(
                        f"cohesion_df columns (slates) must be exactly {list(self.slates)} "
                        f"as defined in the 'slate_to_candidates' parameter. Got "
                        f"{list(self.cohesion_df.columns)}"
                    )
                )

            if set(self.cohesion_df.index) != set(self.blocs):
                errors.append(
                    KeyError(
                        f"cohesion_df index (blocs) must be exactly {list(self.blocs)} "
                        f"as defined in the 'bloc_proportions' parameter. Got"
                        f"{list(self.cohesion_df.index)}"
                    )
                )

            for row in self.cohesion_df.iterrows():
                if any(row[1] < 0):
                    errors.append(
                        ValueError(
                            f"cohesion_df row for bloc '{row[0]}' has values that have not been "
                            f"set (indicated with value of -1)."
                        )
                    )
                elif abs(row[1].sum() - 1.0) > 1e-8:
                    errors.append(
                        ValueError(
                            f"cohesion_df row for bloc '{row[0]}' must sum to 1, "
                            f"got {row[1].sum():.6f}"
                        )
                    )

        return errors

    def is_valid(
        self, *, raise_errors: bool = False, raise_warnings: bool = True
    ) -> bool:
        """
        Check if the current configuration is valid and can be passed to a ballot generator.

        Args:
            raise_errors (bool): If True, raises the first error encountered instead of
                returning False. Defaults to False.
            raise_warnings (bool): If True, raises warnings for non-fatal issues
                (e.g., bloc proportions not summing to 1). Defaults to True.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        all_errors = self.__determine_errors()
        if len(all_errors) == 0:
            return True

        for err in all_errors:
            if raise_errors:
                raise err
            if raise_warnings and not self.silent:
                warn(str(err.args[0]), ConfigurationWarning)

        return False

    def __update_preference_df_on_candidate_change(self) -> None:
        """
        Update the preference DataFrame when candidates change in slate_to_candidates.
        """
        current_slate_cand_dict = self.slate_to_candidates.to_dict()
        curr_pref_df_slate_cands_dict = self._current_preference_df_slate_cand_mapping

        if current_slate_cand_dict == curr_pref_df_slate_cands_dict:
            return

        if self.preference_df.empty:
            # If the preference_df is empty, just create a new one with the right shape
            self.preference_df = pd.DataFrame(
                -1.0,
                index=self.blocs,
                columns=self.candidates,
                dtype=float,
            )
            return

        slate_cand_list_pairs: list[tuple[str, list[str]]]
        if curr_pref_df_slate_cands_dict is None:
            slate_cand_list_pairs = list()
        else:
            slate_cand_list_pairs = list(current_slate_cand_dict.items())

        drop_set = set()
        for slate, current_cand_list in slate_cand_list_pairs:
            if slate not in current_slate_cand_dict:  # pragma: no cover
                drop_set.update(set(current_cand_list))
            else:
                drop_set.update(
                    set(current_cand_list) - set(current_slate_cand_dict.get(slate, []))
                )

        # Remove candidates that are no longer in the config
        drop_list = list(drop_set)
        if len(drop_list) > 0:  # pragma: no cover
            self.preference_df.drop(columns=drop_list, inplace=True)

        pref_df_cands = set(self.preference_df.columns)
        current_cands = set(self.candidates)
        # Add new candidates with default -1.0 values
        for c in current_cands - pref_df_cands:
            self.preference_df[c] = -1.0

        self._current_preference_df_slate_cand_mapping = current_slate_cand_dict
        # Reorder columns to match config order
        self.preference_df = self.preference_df[self.candidates]

    def __update_cohesion_df_on_slate_change(self) -> None:
        """
        Update the cohesion DataFrame when slates change in slate_to_candidates.
        """
        current_slates = set(self.cohesion_df.columns)
        config_slates = set(self.slates)

        if current_slates == config_slates:
            return

        if self.cohesion_df.empty:
            # If the cohesion_df is empty, just create a new one with the right shape
            self.cohesion_df = pd.DataFrame(
                -1.0,
                index=self.blocs,
                columns=self.slates,
                dtype=float,
            )
            return

        # Remove slates that are no longer in the config
        self.cohesion_df.drop(
            columns=list(current_slates - config_slates), inplace=True
        )

        # Add new slates with default -1.0 values
        self.cohesion_df[list(config_slates - current_slates)] = -1.0

        # Reorder columns to match config order
        self.cohesion_df = self.cohesion_df[self.slates]

    def __update_preference_df_on_bloc_change(self) -> None:
        """
        Update the preference DataFrame when blocs change in bloc_proportions.
        """
        current_blocs = set(self.preference_df.index)
        config_blocs = set(self.blocs)

        if current_blocs == config_blocs:
            return

        if self.preference_df.empty:
            # If the preference_df is empty, just create a new one with the right shape
            self.preference_df = pd.DataFrame(
                -1.0,
                index=self.blocs,
                columns=self.candidates,
                dtype=float,
            )
            return

        # Remove blocs that are no longer in the config
        self.preference_df.drop(index=list(current_blocs - config_blocs), inplace=True)

        # Add new blocs with default -1.0 values
        self.preference_df.loc[list(config_blocs - current_blocs)] = -1.0

        # Reorder rows to match config order
        self.preference_df = self.preference_df.loc[self.blocs]

    def __update_cohesion_df_on_bloc_change(self) -> None:
        """
        Update the cohesion DataFrame when blocs change in bloc_proportions.
        """
        current_blocs = set(self.cohesion_df.index)
        config_blocs = set(self.blocs)

        if current_blocs == config_blocs:
            return

        if self.cohesion_df.empty:
            # If the cohesion_df is empty, just create a new one with the right shape
            self.cohesion_df = pd.DataFrame(
                -1.0,
                index=self.blocs,
                columns=self.slates,
                dtype=float,
            )
            return

        # Remove blocs that are no longer in the config
        self.cohesion_df.drop(index=list(current_blocs - config_blocs), inplace=True)

        # Add new blocs with default -1.0 values
        self.cohesion_df.loc[list(config_blocs - current_blocs)] = -1.0

        # Reorder rows to match config order
        self.cohesion_df = self.cohesion_df.loc[self.blocs]

    def _update_preference_and_cohesion_slates(self) -> None:
        """
        Update preference and cohesion DataFrames when slates or candidates change
        """
        self.__update_preference_df_on_candidate_change()
        self.__update_cohesion_df_on_slate_change()

    def _update_preference_and_cohesion_blocs(self) -> None:
        """
        Update preference and cohesion DataFrames when blocs change
        """
        self.__update_preference_df_on_bloc_change()
        self.__update_cohesion_df_on_bloc_change()

    def __setattr__(self, name: str, value: Any) -> None:
        match name:
            case "n_voters":
                self.__validate_voters(value)
                value = int(value)
                object.__setattr__(self, name, value)

            case "slate_to_candidates":
                if isinstance(value, SlateCandMap):
                    object.__setattr__(self, name, value)
                else:
                    value = SlateCandMap(
                        self, value
                    )  # This class handles the validation
                    object.__setattr__(self, name, value)

                if self.bloc_proportions != {}:
                    self._update_preference_and_cohesion_slates()

            case "bloc_proportions":
                if isinstance(value, BlocProportions):
                    object.__setattr__(self, name, value)
                else:
                    value = BlocProportions(self, value)
                    object.__setattr__(self, name, value)

                if self.slate_to_candidates != {}:
                    self._update_preference_and_cohesion_blocs()

            case "preference_df":
                self.__validate_pref_df_mapping_keys_ok_in_config(value)
                value = convert_preference_map_to_preference_df(value)
                object.__setattr__(self, name, value)

                if self.__clear_alpha_bool:
                    self.__alphas = None
                if not value.empty:
                    self.__update_preference_df_on_candidate_change()
                    self.__update_preference_df_on_bloc_change()

                # must come after update
                self.__clear_alpha_bool = True

            case "cohesion_df":
                self.__validate_cohesion_df_mapping_keys_ok_in_config(value)
                value = convert_cohesion_map_to_cohesion_df(value)
                object.__setattr__(self, name, value)

                if not value.empty:
                    self.__update_cohesion_df_on_slate_change()
                    self.__update_cohesion_df_on_bloc_change()

            case "_BlocSlateConfig__alphas":
                if value is not None:
                    self.__keycheck_dirichlet_alphas(value)
                object.__setattr__(self, name, value)

            case "_BlocSlateConfig__clear_alpha_bool":
                object.__setattr__(self, name, value)

            case "_current_preference_df_slate_cand_mapping":
                object.__setattr__(self, name, value)

            case "silent":  # pragma: no cover
                if not isinstance(cast(object, value), bool):
                    raise TypeError("silent must be a bool.")
                object.__setattr__(self, name, value)

            case "candidates":
                raise AttributeError("'candidates' is a read-only property.")

            case "slates":
                raise AttributeError("'slates' is a read-only property.")

            case "blocs":
                raise AttributeError("'blocs' is a read-only property.")

            case _:  # pragma: no cover
                raise AttributeError(
                    f"'BlocSlateConfig' object has no attribute '{name}'"
                )

    # ============
    #   MAIN API
    # ============

    def get_preference_interval_for_bloc_and_slate(
        self, bloc_name: str, slate_name: str
    ) -> PreferenceInterval:
        """
        Get the preference interval for a given bloc and slate.

        Args:
            bloc (str): The name of the voter bloc.
            slate (str): The name of the slate.

        Returns:
            PreferenceInterval: The preference interval for the given bloc and slate.
        """
        # Check to make sure the slate preference intervals are set
        for bloc in self.blocs:
            if slate_name not in self.slate_to_candidates:
                raise KeyError(
                    f"Slate '{slate_name}' not found in slate_to_candidates. "
                    f"Available slates: {self.slates}"
                )
            cand_list = list(self.slate_to_candidates[slate_name])
            if bloc not in self.preference_df.index:
                raise KeyError(
                    f"Bloc '{bloc}' not found in preference_df index. "
                    f"Available blocs: {list(self.preference_df.index)}"
                )
            if any(self.preference_df[cand_list].loc[bloc] < 0):
                raise ValueError(
                    f"Preference interval for bloc '{bloc}' and slate '{slate_name}' has "
                    f"candidates {cand_list} that have not been set (indicated with "
                    f"value of -1)."
                )
            if abs(self.preference_df[cand_list].loc[bloc].sum() - 1.0) > 1e-8:
                raise ValueError(
                    f"Preference interval for bloc '{bloc}' and slate '{slate_name}' must "
                    f"sum to 1, got {self.preference_df[cand_list].loc[bloc].sum():.6f}"
                )

        return PreferenceInterval(
            {
                c: float(self.preference_df[c].loc[bloc_name])
                for c in self.slate_to_candidates[slate_name]
            }
        )

    def get_preference_intervals_for_bloc(
        self, block_name
    ) -> dict[str, PreferenceInterval]:
        """
        Get the preference intervals for each bloc and slate.

        Returns:
            dict[str, dict[str, PreferenceInterval]]: A nested mapping of bloc names to
                slate names to their preference intervals.
        """
        return {
            slate: self.get_preference_interval_for_bloc_and_slate(
                bloc_name=block_name, slate_name=slate
            )
            for slate in self.slates
        }

    def get_combined_preference_intervals_by_bloc(
        self,
    ) -> dict[str, PreferenceInterval]:
        """
        Get the combined preference intervals for each bloc across all slates.

        The combined preference interval for a bloc is computed by combining the
        preference intervals for each slate, weighted by the bloc's cohesion
        parameters for each slate.

        Returns:
            dict[str, PreferenceInterval]: A mapping of bloc names to their combined
                preference intervals.
        """
        return {
            bloc: combine_preference_intervals(
                [
                    self.get_preference_interval_for_bloc_and_slate(
                        bloc_name=bloc, slate_name=slate
                    )
                    for slate in self.slates
                ],
                [self.cohesion_df[slate].loc[bloc] for slate in self.slates],
            )
            for bloc in self.blocs
        }

    def normalize_preference_intervals(self) -> None:
        """
        Normalize each bloc's preference interval so that it sums to 1.

        Note: Will set all uninitialized candidates (value -1.0) to 0.0 before
        normalizing.
        """
        mask = self.preference_df == -1.0
        self.preference_df = self.preference_df.mask(mask, 0.0)
        for cand_lst_proxy in self.slate_to_candidates.values():
            cand_list = list(cand_lst_proxy)
            self.preference_df[cand_list] = self.preference_df[cand_list].div(
                self.preference_df[cand_list].sum(axis=1), axis=0
            )

    def normalize_cohesion_df(self) -> None:
        """
        Normalize each bloc's cohesion parameters so that they sum to 1.

        Note: Will set all uninitialized slates (value -1.0) to 0.0 before
        normalizing.
        """
        mask = self.cohesion_df == -1.0
        self.cohesion_df = self.cohesion_df.mask(mask, 0.0)
        self.cohesion_df = self.cohesion_df.div(self.cohesion_df.sum(axis=1), axis=0)

    def unset_candidate_preferences(
        self, candidates: Union[str, Sequence[str]]
    ) -> None:
        """
        Unset the preferences for the given candidates by setting their values to -1.0.

        Args:
            candidates (Union[str, Sequence[str]]): Candidate name or list of candidate names
                to unset.
        """
        if isinstance(cast(object, candidates), str):  # pragma: no cover
            candidate = str(candidates)
            candidates = [candidate]

        cand_set = set(candidates)

        if set(self.candidates).intersection(cand_set) == set():
            return

        self.preference_df[list(cand_set)] = -1.0

    def add_slate(self, slate: str, slate_candidate_list: Sequence[str]) -> None:
        """
        Add a new slate with the given candidates to the configuration.

        Note: Also modifies the preference_df and cohesion_df to add dummy
        values for the new slate and candidates.

        Args:
            slate (str): Name of the new slate to add.
            slate_candidate_list (Sequence[str]): List of candidate names for the new slate.

        Raises:
            ValueError: If the slate already exists
            ValueError: If any candidate in the slate_candidate_list already exists in the
                configuration.
            ValueError: If the slate_candidate_list is empty.
            TypeError: If slate is not a str or any candidate in the slate_candidate_list is
                       not a str.
        """
        if slate in self.slates:
            raise ValueError(f"Slate '{slate}' already present in configuration.")

        if not isinstance(cast(object, slate_candidate_list), Sequence):
            raise TypeError("slate_candidate_list must be a sequence of str.")

        if set(slate_candidate_list).intersection(set(self.candidates)) != set():
            raise ValueError(
                "Some candidates in the slate are already present in configuration."
            )

        if slate_candidate_list == []:
            raise ValueError("Slate candidate list cannot be empty.")

        if len(slate_candidate_list) != len(set(slate_candidate_list)):
            raise ValueError(
                "slate_candidate_list cannot contain duplicate candidates."
            )

        new_candidate_list = []
        for cand in slate_candidate_list:
            if not isinstance(cast(object, cand), str):
                raise TypeError("Slate candidates must be a 'str'")
            new_candidate_list.append(cand)

        self.slate_to_candidates[slate] = new_candidate_list

        self.__update_preference_df_on_candidate_change()
        self.__update_cohesion_df_on_slate_change()

    def remove_slate(self, slate: str) -> None:
        """
        Remove a slate and its candidates from the configuration.

        Note: Also modifies the preference_df and cohesion_df to remove the slate
        and its candidates.

        Args:
            slate (str): Name of the slate to remove.
        """
        if slate not in self.slates:
            raise KeyError(f"Slate '{slate}' not found in configuration.")

        del self.slate_to_candidates[slate]

        self.__update_preference_df_on_candidate_change()
        self.__update_cohesion_df_on_slate_change()

    def remove_candidates(self, candidates: Union[str, Sequence[str]]) -> None:
        """
        Remove candidates from the configuration.

        Note: Also modifies the preference_df and cohesion_df to remove any slates
        that contained the removed candidates if the slate becomes empty.

        Args:
            candidates (Union[str, Sequence[str]]): Candidate name or list of candidate names
                to remove.
        """
        if isinstance(cast(object, candidates), str):
            candidate = str(candidates)
            candidates = [candidate]

        cand_set = set(candidates)
        if set(self.candidates).intersection(cand_set) == set():
            return

        slates_to_remove_set = set()
        for slate, clist in self.slate_to_candidates.items():
            if any(candidate in clist for candidate in candidates):
                slates_to_remove_set.add(slate)

        if slates_to_remove_set != set():
            for slate in slates_to_remove_set:
                clist = list(self.slate_to_candidates[slate])
                intersection = set(clist).intersection(cand_set)
                new_clist = [c for c in clist if c not in intersection]
                if len(new_clist) == 0:
                    del self.slate_to_candidates[slate]
                else:
                    self.slate_to_candidates[slate] = new_clist

        self.__update_preference_df_on_candidate_change()
        self.__update_cohesion_df_on_slate_change()

    def rename_candidates(self, candidate_mapping: Mapping[str, str]) -> None:
        """
        Rename candidates in the configuration in place.

        Args:
            candidate_mapping (Mapping[str, str]): A mapping of old candidate names to new
                candidate names.

        Raises:
            ValueError: If any old candidate name does not exist in the configuration.
            TypeError: If any key or value in candidate_mapping is not a str.
        """
        if not isinstance(cast(object, candidate_mapping), Mapping):
            raise TypeError("candidate_mapping must be a mapping of str to str.")

        for key, value in candidate_mapping.items():
            if not isinstance(cast(object, key), str):
                raise TypeError(
                    f"Candidate mapping keys must be a 'str', got '{key!r}' of type "
                    f"'{type(key).__name__}'"
                )
            if not isinstance(cast(object, value), str):
                raise TypeError(
                    f"Candidate mapping values must be a 'str', got '{value!r}' of type "
                    f"'{type(value).__name__}' for key '{key}'"
                )

        current_cands = set(self.candidates)
        for old_cand in candidate_mapping.keys():
            if old_cand not in current_cands:
                raise ValueError(
                    f"Candidate mapping key '{old_cand}' not present in configuration."
                )

        new_slate_to_candidates: dict[str, list[str]] = {}
        full_new_candidate_list: list[str] = []
        for slate, clist in self.slate_to_candidates.items():
            new_clist = [
                candidate_mapping[c] if c in candidate_mapping else c for c in clist
            ]
            new_slate_to_candidates[slate] = new_clist
            full_new_candidate_list.extend(new_clist)

        if len(full_new_candidate_list) != len(set(full_new_candidate_list)):
            raise ValueError("Candidate mapping results in duplicate candidate names.")

        self.preference_df.rename(columns=candidate_mapping, inplace=True)
        self.slate_to_candidates = SlateCandMap(self, new_slate_to_candidates)

    # ===================
    #   DIRICHLET STUFF
    # ===================

    def __keycheck_dirichlet_alphas(
        self, alphas: Union[pd.DataFrame, Mapping[str, Mapping[str, Union[float, int]]]]
    ) -> None:
        """
        Check that the given Dirichlet alphas have the correct keys and values.

        Args:
            alphas (Union[pd.DataFrame, Mapping[str, Mapping[str, Union[float, int]]]]):
                The Dirichlet alphas to check.

        Raises:
            TypeError: If alphas is not a pd.DataFrame or a mapping of mappings with the proper
                types at every level.
            ValueError: If alphas does not have the correct blocs and slates as keys, or if any
                alpha value is not a positive finite real.
        """
        all_blocs = set(self.blocs)
        all_slates = set(self.slates)

        if isinstance(alphas, pd.DataFrame):
            df = alphas
            if not all(isinstance(i, str) for i in df.index):
                raise TypeError("Dirichlet alphas index (blocs) must be a 'str'.")
            if not all(isinstance(c, str) for c in df.columns):
                raise TypeError("Dirichlet alphas columns (slates) must be a 'str'.")
            if not all(pd.api.types.is_float_dtype(dt) for dt in df.dtypes):
                raise TypeError(
                    "Dirichlet alphas must have float dtypes in every column."
                )
            if not np.isfinite(df.to_numpy()).all():
                raise ValueError("Dirichlet alphas contains non-finite values.")
            if not (df.to_numpy() > 0).all():
                raise ValueError("Dirichlet alphas must be positive finite reals.")
            if set(df.index) != all_blocs:
                raise ValueError(
                    f"Dirichlet alphas must have exactly the blocs {all_blocs}, "
                    f"got {set(df.index)}"
                )
            if set(df.columns) != all_slates:
                raise ValueError(
                    f"Dirichlet alphas must have exactly the slates {all_slates}, "
                    f"got {set(df.columns)}"
                )
            return

        if set(alphas.keys()) != all_blocs:
            for k in alphas.keys():
                if not isinstance(cast(object, k), str):
                    raise TypeError(
                        f"Dirichlet alphas bloc keys must be a 'str', got '{k!r}' of type "
                        f"'{type(k).__name__}'"
                    )
            raise ValueError(
                f"Dirichlet alphas must have exactly the blocs {all_blocs}, got "
                f"{set(alphas.keys())}"
            )

        for bloc, slate_map in alphas.items():
            if set(slate_map.keys()) != all_slates:
                if not isinstance(cast(object, slate_map), Mapping):
                    raise TypeError(
                        f"In bloc '{bloc}': Dirichlet alphas must be a mapping, got "
                        f"'{type(slate_map).__name__}'"
                    )
                raise ValueError(
                    f"In bloc '{bloc}': Dirichlet alphas must have exactly the slates "
                    f"{all_slates}, got {set(slate_map.keys())}"
                )
            for slate, v in slate_map.items():
                if not isinstance(cast(object, slate), str):
                    raise TypeError(
                        f"In bloc '{bloc}': Dirichlet alphas slate keys must be a 'str', got "
                        f"'{slate!r}' of type '{type(slate).__name__}'"
                    )
                if not isinstance(cast(object, v), Real):
                    raise TypeError(
                        f"In bloc '{bloc}', slate '{slate}': Dirichlet alpha must be a finite "
                        f"real (int|float), got '{v!r}' of type '{type(v).__name__}'"
                    )
                if not _is_finite_real(v) or v <= 0:
                    raise ValueError(
                        f"In bloc '{bloc}', slate '{slate}': Dirichlet alpha must be a positive "
                        f"finite real, got '{v!r}'"
                    )

    def set_dirichlet_alphas(
        self, alphas: Union[Mapping[str, Mapping[str, Union[float, int]]], pd.DataFrame]
    ) -> None:
        """
        Set the Dirichlet alphas for the configuration and resample the preference intervals.

        Args:
            alphas (Union[Mapping[str, Mapping[str, Union[float, int]]], pd.DataFrame]): A mapping
                of bloc names to mappings of slate names to their Dirichlet alpha values. Each bloc
                must have a mapping for every slate defined in slate_to_candidates. All alpha
                values must be positive finite reals.

        Raises:
            ConfigurationWarning: If preference intervals have already been set without
                setting the Dirichlet alphas. Setting the Dirichlet alphas will overwrite
                the existing preference intervals
        """
        if self.__alphas is None and not all(
            self.preference_df.values.flatten() == -1.0
        ):
            warning_msg = (
                "Preference intervals have already been set without setting the Dirichlet "
                "alphas. Setting the Dirichlet alphas will overwrite the existing preference "
                "intervals."
            )
            if not self.silent:
                warn(warning_msg, ConfigurationWarning)

        self.__keycheck_dirichlet_alphas(alphas)
        if isinstance(alphas, Mapping):
            self.__alphas = pd.DataFrame(alphas).astype(float).T
        else:
            self.__alphas = alphas.copy().astype(float)
        self.__clear_alpha_bool = False
        self.resample_preference_intervals_from_dirichlet_alphas()

    def clear_dirichlet_alphas(self) -> None:
        """
        Remove the Dirichlet alphas from the configuration.
        """
        self.__alphas = None

    def read_dirichlet_alphas(self) -> Optional[pd.DataFrame]:
        """
        Get a copy of the current Dirichlet alphas.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with blocs as the index and slates as columns,
            containing the Dirichlet alpha values for each slate in each bloc, or None if
            Dirichlet alphas have not been set.
        """
        return self.__alphas.copy() if self.__alphas is not None else None

    def resample_preference_intervals_from_dirichlet_alphas(
        self,
    ) -> None:
        """
        Resample the preference intervals for each bloc from the current Dirichlet alphas.

        Note: This will overwrite any existing preference intervals in the configuration.
        """
        if self.__alphas is None:
            raise ValueError("Dirichlet alphas have not been set.")

        self.__clear_alpha_bool = False
        preference_dict = {}
        for bloc in self.blocs:
            slate_intervals = {}
            for slate in self.slates:
                slate_intervals[slate] = PreferenceInterval.from_dirichlet(
                    candidates=list(self.slate_to_candidates[slate]),
                    alpha=float(self.__alphas[slate].loc[bloc]),
                )
            preference_dict[bloc] = slate_intervals
        preference_df = convert_preference_map_to_preference_df(preference_dict)
        self.preference_df = preference_df
        self.__clear_alpha_bool = True

    def copy(self) -> "BlocSlateConfig":
        """
        Create a deep copy of the current configuration.

        Returns:
            BlocSlateConfig: A deep copy of the current configuration.
        """
        new_config = BlocSlateConfig(
            n_voters=self.n_voters,
            slate_to_candidates=self.slate_to_candidates.to_dict(),
            bloc_proportions=self.bloc_proportions.copy(),
            cohesion_mapping=self.cohesion_df.copy(),
            silent=self.silent,
        )

        # Cannot put the preference_df in the constructor because there could
        # be -1 value in it which will be overwritten by the constructor logic
        new_config.preference_df = self.preference_df.copy()

        new_config._BlocSlateConfig__alphas = (
            self.__alphas.copy() if self.__alphas is not None else None
        )
        new_config._BlocSlateConfig__clear_alpha_bool = self.__clear_alpha_bool
        return new_config

    def __repr__(self) -> str:  # pragma: no cover
        def _block(title: str, body: str) -> str:
            # indent non-empty lines
            return f"{title}:\n{indent(body, '    ', lambda message: message.strip() != '')}"

        header = "\n".join(
            [
                "Ballot Generator Bloc Slate Config",
                "----------------------------------",
                f"Number of Voters: {self.n_voters}",
            ]
        )

        parts = [
            header,
            _block(
                "Slates to Candidates",
                pformat(dict(self.slate_to_candidates), sort_dicts=False),
            ),
            _block("Bloc Proportions", pformat(dict(self.bloc_proportions))),
            _block("Preference mapping", self.preference_df.to_string()),
            _block("Cohesion mapping", self.cohesion_df.to_string()),
            _block("Dirichlet Alphas", pformat(self.__alphas)),
            _block(
                "Valid",
                pformat(self.is_valid(raise_errors=False, raise_warnings=False)),
            ),
        ]
        return "\n\n".join(parts)  # blank line between sections, optional

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlocSlateConfig):  # pragma: no cover
            return False

        return (
            self.n_voters == other.n_voters
            and self.slate_to_candidates == other.slate_to_candidates
            and self.bloc_proportions == other.bloc_proportions
            and self.preference_df.equals(other.preference_df)
            and self.cohesion_df.equals(other.cohesion_df)
            and (
                (self.__alphas is None and other.__alphas is None)
                or (
                    self.__alphas is not None
                    and other.__alphas is not None
                    and self.__alphas.equals(other.__alphas)
                )
            )
        )
