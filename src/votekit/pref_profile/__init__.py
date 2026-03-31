from votekit.pref_profile.cleaned_pref_profile import CleanedRankProfile, CleanedScoreProfile
from votekit.pref_profile.pref_profile import (
    PreferenceProfile,
    ProfileError,
    RankProfile,
    ScoreProfile,
)
from votekit.pref_profile.utils import (
    convert_rank_profile_to_score_profile_via_score_vector,
    convert_row_to_rank_ballot,
    profile_df_head,
    profile_df_tail,
    rank_profile_to_ballot_dict,
    rank_profile_to_ranking_dict,
    score_profile_to_ballot_dict,
    score_profile_to_scores_dict,
)

__all__ = [
    "PreferenceProfile",
    "RankProfile",
    "ScoreProfile",
    "ProfileError",
    "rank_profile_to_ballot_dict",
    "score_profile_to_ballot_dict",
    "rank_profile_to_ranking_dict",
    "score_profile_to_scores_dict",
    "profile_df_head",
    "profile_df_tail",
    "CleanedRankProfile",
    "CleanedScoreProfile",
    "convert_row_to_rank_ballot",
    "convert_rank_profile_to_score_profile_via_score_vector",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
