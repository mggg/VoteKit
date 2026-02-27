"""BlocSlateConfig core class implementation."""

from collections.abc import Mapping, Sequence
from pprint import pformat
from textwrap import indent
from typing import Any, Optional, Union, cast
from warnings import warn
from numbers import Real

import numpy as np
import pandas as pd

from votekit.pref_interval import PreferenceInterval, combine_preference_intervals

from votekit.ballot_generator.bloc_slate_generator.config.collections import (
    BlocProportions,
    SlateCandMap,
)
from votekit.ballot_generator.bloc_slate_generator.config.validation import (
    BlocProportionMapping,
    CohesionMapping,
    ConfigurationWarning,
    PreferenceMapping,
    UNSET_VALUE,
    _first_probability_row_error,
    _is_finite_real,
    _invalid_negative_values,
    _sum_differs_from_one,
    _to_finite_float_array,
    _unset_mask,
    convert_bloc_proportion_map_to_series,
    convert_cohesion_map_to_cohesion_df,
    convert_preference_map_to_preference_df,
)


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
        bloc_proportions (Optional[BlocProportionMapping]): A mapping of voter bloc names to
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
        bloc_proportions: Optional[BlocProportionMapping] = None,
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
            preference_df = preference_df.reindex(
                columns=self.candidates, fill_value=UNSET_VALUE
            )  # ensure column order and preserve unset candidates
        object.__setattr__(self, "preference_df", preference_df)

        object.__setattr__(self, "_BlocSlateConfig__alphas", None)

    @property
    def candidates(self) -> list[str]:
        """
        Computed property: A flat list of all candidates in all slate.

        Derived from the values of slate_to_candidates.
        """
        return [c for clist in self.slate_to_candidates.values() for c in clist]

    @property
    def slates(self) -> list[str]:
        """Computed property: A list of all slates. Derived from the keys of slate_to_candidates."""
        return list(self.slate_to_candidates.keys())

    @property
    def blocs(self) -> list[str]:
        """
        Computed property: A list of all voter blocs.

        Derived from the keys of bloc_proportions.
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
        Validate that the keys in the preference mapping are compatible with the current config.

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
        Validate that the keys in the cohesion mapping are compatible with the current config.

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
                    "Cohesion mapping has voter blocs but no blocs are defined in bloc_proportions."
                )
            else:
                messages.append(
                    f"Cohesion mapping expected exactly the blocs "
                    f"{sorted(self.blocs)}, got {sorted(list(blocs))}"
                )

        if set(slates) != set(self.slates):
            if self.slates == []:
                messages.append(
                    "Cohesion mapping has slates but no slates are defined in slate_to_candidates."
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
        Determine if there are any errors in the current configuration.

        Errors are those settings which will produce invalid states when passed to a ballot
        generator function.

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
            if _sum_differs_from_one(bloc_ser.sum()):
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
                        bloc_name = str(row[0])
                        row_vals = row[1]

                        def _pref_invalid_negative_error(
                            invalid_negatives: list[float],
                        ) -> Exception:
                            return ValueError(
                                f"preference_df row for bloc '{bloc_name}' has invalid "
                                f"negative values {invalid_negatives}. Use -1 to mark "
                                "unset values."
                            )

                        def _pref_unset_error() -> Exception:
                            return ValueError(
                                f"preference_df row for bloc '{bloc_name}' has values "
                                f"that have "
                                f"not been set (indicated with value of {UNSET_VALUE:g})."
                            )

                        def _pref_sum_error(total: float) -> Exception:
                            return ValueError(
                                f"preference_df row for bloc '{bloc_name}' and candidates "
                                f"{cand_list} must sum to 1, got {total:.6f}"
                            )

                        row_error = _first_probability_row_error(
                            row_vals,
                            context=f"preference_df row for bloc '{bloc_name}'",
                            invalid_negative_error=_pref_invalid_negative_error,
                            unset_error=_pref_unset_error,
                            sum_error=_pref_sum_error,
                        )
                        if row_error is not None:
                            errors.append(row_error)

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
                bloc_name = str(row[0])
                row_vals = row[1]

                def _cohesion_invalid_negative_error(
                    invalid_negatives: list[float],
                ) -> Exception:
                    return ValueError(
                        f"cohesion_df row for bloc '{bloc_name}' has invalid "
                        f"negative values {invalid_negatives}. Use -1 to mark "
                        "unset values."
                    )

                def _cohesion_unset_error() -> Exception:
                    return ValueError(
                        f"cohesion_df row for bloc '{bloc_name}' has values that have not been "
                        f"set (indicated with value of {UNSET_VALUE:g})."
                    )

                def _cohesion_sum_error(total: float) -> Exception:
                    return ValueError(
                        f"cohesion_df row for bloc '{bloc_name}' must sum to 1, got {total:.6f}"
                    )

                row_error = _first_probability_row_error(
                    row_vals,
                    context=f"cohesion_df row for bloc '{bloc_name}'",
                    invalid_negative_error=_cohesion_invalid_negative_error,
                    unset_error=_cohesion_unset_error,
                    sum_error=_cohesion_sum_error,
                )
                if row_error is not None:
                    errors.append(row_error)
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

    def __make_unset_df(
        self, *, index: Sequence[str], columns: Sequence[str]
    ) -> pd.DataFrame:
        """Build a float DataFrame filled with UNSET_VALUE for the requested shape."""
        return pd.DataFrame(
            UNSET_VALUE,
            index=pd.Index(index),
            columns=pd.Index(columns),
            dtype=float,
        )

    def __sync_df_columns(
        self, df: pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        """Drop unknown columns, add missing columns as UNSET_VALUE, and reorder."""
        expected_columns = list(columns)
        expected_set = set(expected_columns)

        drop_columns = [col for col in df.columns if col not in expected_set]
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True)

        for col in expected_columns:
            if col not in df.columns:
                df[col] = UNSET_VALUE

        return df[expected_columns]

    def __sync_df_index(self, df: pd.DataFrame, index: Sequence[str]) -> pd.DataFrame:
        """Drop unknown rows, add missing rows as UNSET_VALUE, and reorder."""
        expected_index = list(index)
        expected_set = set(expected_index)

        drop_index = [idx for idx in df.index if idx not in expected_set]
        if drop_index:
            df.drop(index=drop_index, inplace=True)

        missing_index = [idx for idx in expected_index if idx not in df.index]
        if missing_index:
            df.loc[missing_index] = UNSET_VALUE

        return df.loc[expected_index]

    def __update_preference_df_on_candidate_change(self) -> None:
        """Update the preference DataFrame when candidates change in slate_to_candidates."""
        current_slate_cand_dict = self.slate_to_candidates.to_dict()
        curr_pref_df_slate_cands_dict = self._current_preference_df_slate_cand_mapping

        if current_slate_cand_dict == curr_pref_df_slate_cands_dict:
            return

        if self.preference_df.empty:
            # If the preference_df is empty, just create a new one with the right shape
            self.preference_df = self.__make_unset_df(
                index=self.blocs,
                columns=self.candidates,
            )
            return

        self._current_preference_df_slate_cand_mapping = current_slate_cand_dict
        self.preference_df = self.__sync_df_columns(self.preference_df, self.candidates)

    def __update_cohesion_df_on_slate_change(self) -> None:
        """Update the cohesion DataFrame when slates change in slate_to_candidates."""
        config_slates = self.slates

        if list(self.cohesion_df.columns) == list(config_slates):
            return

        if self.cohesion_df.empty:
            # If the cohesion_df is empty, just create a new one with the right shape
            self.cohesion_df = self.__make_unset_df(
                index=self.blocs,
                columns=self.slates,
            )
            return

        self.cohesion_df = self.__sync_df_columns(self.cohesion_df, config_slates)

    def __update_preference_df_on_bloc_change(self) -> None:
        """Update the preference DataFrame when blocs change in bloc_proportions."""
        config_blocs = self.blocs

        if list(self.preference_df.index) == list(config_blocs):
            return

        if self.preference_df.empty:
            # If the preference_df is empty, just create a new one with the right shape
            self.preference_df = self.__make_unset_df(
                index=self.blocs,
                columns=self.candidates,
            )
            return

        self.preference_df = self.__sync_df_index(self.preference_df, config_blocs)

    def __update_cohesion_df_on_bloc_change(self) -> None:
        """Update the cohesion DataFrame when blocs change in bloc_proportions."""
        config_blocs = self.blocs

        if list(self.cohesion_df.index) == list(config_blocs):
            return

        if self.cohesion_df.empty:
            # If the cohesion_df is empty, just create a new one with the right shape
            self.cohesion_df = self.__make_unset_df(
                index=self.blocs,
                columns=self.slates,
            )
            return

        self.cohesion_df = self.__sync_df_index(self.cohesion_df, config_blocs)

    def _update_preference_and_cohesion_slates(self) -> None:
        """Update preference and cohesion DataFrames when slates or candidates change"""
        self.__update_preference_df_on_candidate_change()
        self.__update_cohesion_df_on_slate_change()

    def _update_preference_and_cohesion_blocs(self) -> None:
        """Update preference and cohesion DataFrames when blocs change"""
        self.__update_preference_df_on_bloc_change()
        self.__update_cohesion_df_on_bloc_change()

    def __set_n_voters_attr(self, value: Any) -> None:
        self.__validate_voters(value)
        object.__setattr__(self, "n_voters", int(value))

    def __set_slate_to_candidates_attr(self, value: Any) -> None:
        slate_map = (
            value if isinstance(value, SlateCandMap) else SlateCandMap(self, value)
        )
        object.__setattr__(self, "slate_to_candidates", slate_map)

        if self.bloc_proportions != {}:
            self._update_preference_and_cohesion_slates()

    def __set_bloc_proportions_attr(self, value: Any) -> None:
        bloc_props = (
            value
            if isinstance(value, BlocProportions)
            else BlocProportions(self, value)
        )
        object.__setattr__(self, "bloc_proportions", bloc_props)

        if self.slate_to_candidates != {}:
            self._update_preference_and_cohesion_blocs()

    def __set_preference_df_attr(self, value: Any) -> None:
        self.__validate_pref_df_mapping_keys_ok_in_config(value)
        pref_df = convert_preference_map_to_preference_df(value)
        object.__setattr__(self, "preference_df", pref_df)

        if self.__clear_alpha_bool:
            self.__alphas = None
        if not pref_df.empty:
            self.__update_preference_df_on_candidate_change()
            self.__update_preference_df_on_bloc_change()

        # must come after update
        self.__clear_alpha_bool = True

    def __set_cohesion_df_attr(self, value: Any) -> None:
        self.__validate_cohesion_df_mapping_keys_ok_in_config(value)
        cohesion_df = convert_cohesion_map_to_cohesion_df(value)
        object.__setattr__(self, "cohesion_df", cohesion_df)

        if not cohesion_df.empty:
            self.__update_cohesion_df_on_slate_change()
            self.__update_cohesion_df_on_bloc_change()

    def __set_alphas_attr(self, value: Any) -> None:
        if value is not None:
            self.__keycheck_dirichlet_alphas(value)
        object.__setattr__(self, "_BlocSlateConfig__alphas", value)

    def __set_silent_attr(self, value: Any) -> None:
        if not isinstance(cast(object, value), bool):
            raise TypeError("silent must be a bool.")
        object.__setattr__(self, "silent", value)

    def __setattr__(self, name: str, value: Any) -> None:
        match name:
            case "n_voters":
                self.__set_n_voters_attr(value)

            case "slate_to_candidates":
                self.__set_slate_to_candidates_attr(value)

            case "bloc_proportions":
                self.__set_bloc_proportions_attr(value)

            case "preference_df":
                self.__set_preference_df_attr(value)

            case "cohesion_df":
                self.__set_cohesion_df_attr(value)

            case "_BlocSlateConfig__alphas":
                self.__set_alphas_attr(value)

            case "_BlocSlateConfig__clear_alpha_bool":
                object.__setattr__(self, "_BlocSlateConfig__clear_alpha_bool", value)

            case "_current_preference_df_slate_cand_mapping":
                object.__setattr__(
                    self, "_current_preference_df_slate_cand_mapping", value
                )

            case "silent":  # pragma: no cover
                self.__set_silent_attr(value)

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
            row_vals = self.preference_df[cand_list].loc[bloc]

            def _interval_invalid_negative_error(
                invalid_negatives: list[float],
            ) -> Exception:
                return ValueError(
                    f"Preference interval for bloc '{bloc}' and slate '{slate_name}' has "
                    f"invalid negative values {invalid_negatives}. Use -1 to mark unset "
                    "values."
                )

            def _interval_unset_error() -> Exception:
                return ValueError(
                    f"Preference interval for bloc '{bloc}' and slate '{slate_name}' has "
                    f"candidates {cand_list} that have not been set (indicated with "
                    f"value of {UNSET_VALUE:g})."
                )

            def _interval_sum_error(total: float) -> Exception:
                return ValueError(
                    f"Preference interval for bloc '{bloc}' and slate '{slate_name}' must "
                    f"sum to 1, got {total:.6f}"
                )

            row_error = _first_probability_row_error(
                row_vals,
                context=f"Preference interval for bloc '{bloc}' and slate '{slate_name}'",
                invalid_negative_error=_interval_invalid_negative_error,
                unset_error=_interval_unset_error,
                sum_error=_interval_sum_error,
            )
            if row_error is not None:
                raise row_error

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
        try:
            pref_values = _to_finite_float_array(
                self.preference_df, context="preference_df"
            )
        except TypeError:
            raise TypeError("preference_df must contain numeric values to normalize.")
        except ValueError:
            raise ValueError(
                "preference_df contains non-finite values and cannot be normalized."
            )

        invalid_negatives = _invalid_negative_values(pref_values)
        if invalid_negatives:
            raise ValueError(
                f"preference_df contains invalid negative values {invalid_negatives}. "
                "Use -1 to mark unset values before normalization."
            )

        mask = _unset_mask(pref_values)
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
        try:
            cohesion_values = _to_finite_float_array(
                self.cohesion_df, context="cohesion_df"
            )
        except TypeError:
            raise TypeError("cohesion_df must contain numeric values to normalize.")
        except ValueError:
            raise ValueError(
                "cohesion_df contains non-finite values and cannot be normalized."
            )

        invalid_negatives = _invalid_negative_values(cohesion_values)
        if invalid_negatives:
            raise ValueError(
                f"cohesion_df contains invalid negative values {invalid_negatives}. "
                "Use -1 to mark unset values before normalization."
            )

        mask = _unset_mask(cohesion_values)
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

        self.preference_df[list(cand_set)] = UNSET_VALUE

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
                    f"Dirichlet alphas must have exactly the blocs {all_blocs}, got {set(df.index)}"
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
            self.preference_df.values.flatten() == UNSET_VALUE
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
        """Remove the Dirichlet alphas from the configuration."""
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
