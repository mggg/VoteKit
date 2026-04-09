from .multi_profile_bar_plot import (
    multi_profile_ballot_lengths_plot,
    multi_profile_bar_plot,
    multi_profile_borda_plot,
    multi_profile_fpv_plot,
    multi_profile_mentions_plot,
)
from .profile_bar_plot import (
    profile_ballot_lengths_plot,
    profile_bar_plot,
    profile_borda_plot,
    profile_fpv_plot,
    profile_mentions_plot,
)

__all__ = [
    "profile_bar_plot",
    "profile_borda_plot",
    "profile_mentions_plot",
    "profile_fpv_plot",
    "profile_ballot_lengths_plot",
    "multi_profile_bar_plot",
    "multi_profile_ballot_lengths_plot",
    "multi_profile_borda_plot",
    "multi_profile_fpv_plot",
    "multi_profile_mentions_plot",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
