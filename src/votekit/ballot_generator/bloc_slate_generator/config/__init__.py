"""Configuration model and utilities for bloc-slate ballot generators."""

from votekit.ballot_generator.bloc_slate_generator.config.collections import (
    BlocProportions,
    SlateCandMap,
)
from votekit.ballot_generator.bloc_slate_generator.config.core import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.config.validation import (
    FLOAT_TOL,
    UNSET_VALUE,
    BlocProportionMapping,
    BlocPropotionMapping,
    CohesionMapping,
    ConfigurationWarning,
    PreferenceIntervalLike,
    PreferenceMapping,
    convert_bloc_proportion_map_to_series,
    convert_cohesion_map_to_cohesion_df,
    convert_preference_map_to_preference_df,
    typecheck_bloc_proportion_mapping,
    typecheck_cohesion_mapping,
    typecheck_preference,
)

__all__ = [
    "UNSET_VALUE",
    "FLOAT_TOL",
    "ConfigurationWarning",
    "BlocProportionMapping",
    "BlocPropotionMapping",
    "CohesionMapping",
    "PreferenceIntervalLike",
    "PreferenceMapping",
    "typecheck_bloc_proportion_mapping",
    "convert_bloc_proportion_map_to_series",
    "typecheck_cohesion_mapping",
    "convert_cohesion_map_to_cohesion_df",
    "typecheck_preference",
    "convert_preference_map_to_preference_df",
    "SlateCandMap",
    "BlocProportions",
    "BlocSlateConfig",
]
