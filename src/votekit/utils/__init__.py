from votekit.utils.common_utils import (
    COLOR_LIST,
    _first_place_votes_from_df_no_ties,
    add_missing_cands,
    ballot_lengths,
    ballots_by_first_cand,
    borda_scores,
    build_df_from_ballot_samples,
    elect_cands_from_set_ranking,
    expand_tied_ballot,
    first_place_votes,
    fixed_zero_index_lex_block_size,
    index_to_lexicographic_ballot,
    mentions,
    resolve_profile_ties,
    score_dict_from_score_vector,
    score_dict_to_ranking,
    score_profile_from_ballot_scores,
    sort_candidates_pseudo_lex,
    sort_candidates_pseudo_lexicographically,
    tiebreak_set,
    tiebroken_ranking,
    validate_score_vector,
)
from votekit.utils.search import search_profile_for_rank_pattern

__all__ = [
    "ballots_by_first_cand",
    "add_missing_cands",
    "validate_score_vector",
    "score_dict_from_score_vector",
    "_first_place_votes_from_df_no_ties",
    "first_place_votes",
    "mentions",
    "borda_scores",
    "tiebreak_set",
    "tiebroken_ranking",
    "score_dict_to_ranking",
    "elect_cands_from_set_ranking",
    "expand_tied_ballot",
    "resolve_profile_ties",
    "score_profile_from_ballot_scores",
    "ballot_lengths",
    "fixed_zero_index_lex_block_size",
    "index_to_lexicographic_ballot",
    "build_df_from_ballot_samples",
    "sort_candidates_pseudo_lexicographically",
    "sort_candidates_pseudo_lex",
    "COLOR_LIST",
    "search_profile_for_rank_pattern",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
