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

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
