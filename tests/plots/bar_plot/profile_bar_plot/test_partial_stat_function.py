from votekit.plots.profiles.profile_bar_plot import _partial_stat_function
from votekit.utils import first_place_votes, mentions, borda_scores, ballot_lengths


def profile_func(profile, kwd=5):
    return {"1": kwd}


def test_partial_stat_function_no_kwds():
    assert _partial_stat_function(profile_func, None) == profile_func

    assert _partial_stat_function("first place votes", None) == first_place_votes
    assert _partial_stat_function("mentions", None) == mentions
    assert _partial_stat_function("borda", None) == borda_scores
    assert _partial_stat_function("ballot lengths", None) == ballot_lengths
