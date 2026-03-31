from votekit.ballot_generator.bloc_slate_generator.cambridge import (
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator.config import (
    BlocSlateConfig,
    convert_bloc_proportion_map_to_series,
    convert_cohesion_map_to_cohesion_df,
    convert_preference_map_to_preference_df,
)
from votekit.ballot_generator.bloc_slate_generator.cumulative import (
    name_cumulative_ballot_generator_by_bloc,
    name_cumulative_profile_generator,
)
from votekit.ballot_generator.bloc_slate_generator.name_bradley_terry import (
    name_bt_profile_generator,
    name_bt_profile_generator_using_mcmc,
    name_bt_profiles_by_bloc_generator,
    name_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.name_plackett_luce import (
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator.slate_bradley_terry import (
    slate_bt_profile_generator,
    slate_bt_profile_generator_using_mcmc,
    slate_bt_profiles_by_bloc_generator,
    slate_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)

__all__ = [
    "slate_bt_profile_generator",
    "slate_bt_profiles_by_bloc_generator",
    "slate_bt_profile_generator_using_mcmc",
    "slate_bt_profiles_by_bloc_generator_using_mcmc",
    "name_bt_profile_generator",
    "name_bt_profiles_by_bloc_generator",
    "name_bt_profile_generator_using_mcmc",
    "name_bt_profiles_by_bloc_generator_using_mcmc",
    "name_pl_profile_generator",
    "name_pl_profiles_by_bloc_generator",
    "name_cumulative_profile_generator",
    "name_cumulative_ballot_generator_by_bloc",
    "cambridge_profile_generator",
    "cambridge_profiles_by_bloc_generator",
    "slate_pl_profile_generator",
    "slate_pl_profiles_by_bloc_generator",
    "slate_bt_profile_generator",
    "slate_bt_profiles_by_bloc_generator",
    "convert_cohesion_map_to_cohesion_df",
    "convert_preference_map_to_preference_df",
    "convert_bloc_proportion_map_to_series",
    "BlocSlateConfig",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
