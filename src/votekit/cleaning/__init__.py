from .rank_ballots_cleaning import (
    condense_rank_ballot,
    remove_cand_rank_ballot,
    remove_repeat_cands_rank_ballot,
)
from .rank_profiles_cleaning import (
    clean_rank_profile,
    condense_rank_profile,
    remove_and_condense_rank_profile,
    remove_cand_rank_profile,
    remove_repeat_cands_rank_profile,
)
from .score_ballots_cleaning import remove_cand_score_ballot
from .score_profiles_cleaning import clean_score_profile, remove_cand_score_profile

__all__ = [
    "clean_rank_profile",
    "remove_repeat_cands_rank_profile",
    "remove_cand_rank_profile",
    "condense_rank_profile",
    "remove_and_condense_rank_profile",
    "remove_cand_rank_ballot",
    "condense_rank_ballot",
    "remove_repeat_cands_rank_ballot",
    "remove_cand_score_ballot",
    "clean_score_profile",
    "remove_cand_score_profile",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
