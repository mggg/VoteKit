"""Mutable collection wrappers used by BlocSlateConfig."""

import operator
import weakref
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from copy import deepcopy
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Self,
    SupportsIndex,
    Union,
    cast,
    overload,
)
from warnings import warn

import pandas as pd

from votekit.ballot_generator.bloc_slate_generator.config.validation import (
    FLOAT_TOL,
    BlocProportionMapping,
    ConfigurationWarning,
    _sum_differs_from_one,
    convert_bloc_proportion_map_to_series,
    typecheck_bloc_proportion_mapping,
)

if TYPE_CHECKING:
    from votekit.ballot_generator.bloc_slate_generator.config.core import (
        BlocSlateConfig,
    )


class _CandListProxy(MutableSequence[str]):
    """
    A proxy for a list of candidates in a slate.

    This proxy routes all changes through the owning SlateCandMap to ensure validation.

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
    def __getitem__(self, index: SupportsIndex) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[str]: ...

    def __getitem__(self, index: Union[SupportsIndex, slice]) -> Union[str, MutableSequence[str]]:
        data = self.__owner._data[self.__key]
        if isinstance(index, slice):
            return data[index]
        return data[operator.index(index)]

    @overload
    def __setitem__(self, index: SupportsIndex, value: str) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[str]) -> None: ...

    def __setitem__(
        self,
        index: Union[SupportsIndex, slice],
        value: Union[str, Iterable[str]],
    ) -> None:
        new = list(self.__owner._data[self.__key])
        if isinstance(index, slice):  # pragma: no cover
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise TypeError("Slice assignment requires an iterable of str")
            new[index] = [str(x) for x in value]
        else:
            new[operator.index(index)] = str(value)
        self.__owner[self.__key] = new

    @overload
    def __delitem__(self, index: SupportsIndex) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: Union[SupportsIndex, slice]) -> None:
        new = list(self.__owner._data[self.__key])
        if isinstance(index, slice):
            del new[index]
        else:
            del new[operator.index(index)]
        self.__owner[self.__key] = new

    def _current(self) -> list[str]:
        return list(self.__owner._data[self.__key])

    def insert(self, index: SupportsIndex, value: str) -> None:
        """
        Inserts candidate value at index if not already present.

        Args:
            index (SupportsIndex): The index at which to insert the candidate.
            value (str): The candidate name to insert.
        """
        if not isinstance(cast(object, value), str):
            raise TypeError("Slate candidates must be a 'str'")
        try:
            int_index = operator.index(index)
        except TypeError:
            raise TypeError("Index must be an 'int'")
        new = list(self.__owner._data[self.__key])
        if value not in new:
            new.insert(int_index, str(value))
        self.__owner[self.__key] = new

    def extend(self, values: Iterable[str]) -> None:
        """
        Extend the candidate list by appending elements from the iterable.

        Args:
            values (Iterable[str]): An iterable of candidate names to append.
        """
        rollback = self._current().copy()
        try:
            for value in values:
                self.insert(len(self), value)

        except Exception as e:
            self.__owner[self.__key] = rollback
            raise e

    def __iadd__(self, values: Iterable[str]) -> Self:
        self.extend(values)
        return self

    def append(self, value: str) -> None:
        """
        Append candidate value to the end of the list if not already present.

        Args:
            value (str): The candidate name to append.
        """
        self.insert(len(self), value)

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
            rollback_slate_dict = self.__parent._current_preference_df_slate_cand_mapping
            self._data[key] = val_list
            try:
                self.__parent._update_preference_and_cohesion_slates()
            except KeyError as e:
                if rollback is None:
                    del self._data[key]
                else:
                    self._data[key] = rollback

                self.__parent._current_preference_df_slate_cand_mapping = rollback_slate_dict
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
        update_hook = getattr(self.__parent, "_update_preference_and_cohesion_slates", None)
        if callable(update_hook):
            update_hook()

    def __iter__(self) -> Iterator[str]:  # noqa
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def update(
        self,
        other: Any = (),
        /,
        **kw: Sequence[str],
    ) -> None:
        """
        Update the slate to candidates mapping with the key-value pairs from other.

        Args:
            other (Mapping[str, Sequence[str]] or Iterable[tuple[str, Sequence[str]]]):
                Another mapping or iterable of key-value pairs to update the slate to candidates
                mapping with.
            **kw: Additional key-value pairs to update the slate to candidates mapping with.
        """
        if isinstance(other, Mapping):
            item_pairs = list(other.items())
        else:
            item_pairs = list(dict(other).items())
        item_pairs.extend(list(kw.items()))

        for k, v in item_pairs:
            if not isinstance(k, str):
                raise TypeError("Slate keys must be str in update().")
            if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
                raise TypeError("Slate values must be sequences of candidate names.")
            self[k] = cast(Sequence[str], v)  # route through __setitem__

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
    Mapping[str, float] that enforces bloc proportion rules.

    Bloc proportion rules are as follows:
    - Each bloc must have a non-negative proportion
    - The proportions must sum to 1

    Args:
        parent (BlocSlateConfig): The owning BlocSlateConfig.
        init (Optional[BlocProportionMapping]): Initial mapping of bloc names to their
            proportions in the electorate. If None, defaults to an empty mapping.
    """

    __slots__ = ("__parent", "__data")

    def __init__(
        self,
        parent: "BlocSlateConfig",
        init: Optional[BlocProportionMapping] = None,
    ) -> None:
        self.__parent = weakref.proxy(parent)
        self.__data: dict[str, float] = {}

        if init is not None:
            ser = convert_bloc_proportion_map_to_series(init)
            self.__data.update({str(bloc): float(value) for bloc, value in ser.items()})

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
        if _sum_differs_from_one(total):
            if not self.__parent.silent:
                warn(
                    "Bloc proportions currently sum to "
                    f"{total:.6f} when they should sum to 1 within tolerance "
                    f"{FLOAT_TOL:g}.",
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
            update_hook = getattr(self.__parent, "_update_preference_and_cohesion_blocs", None)
            if callable(update_hook):
                update_hook()
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
