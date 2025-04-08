from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile

filepath = "tests/data/pickle"


def test_pkl_bijection_rankings():
    profile_rankings = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
        )
        * 5,
        max_ranking_length=3,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_rankings.to_pickle(f"{filepath}/test_pkl_pp_rankings.pkl")
    read_profile = PreferenceProfile.from_pickle(f"{filepath}/test_pkl_pp_rankings.pkl")
    assert profile_rankings == read_profile


def test_pkl_bijection_scores():
    profile_scores = PreferenceProfile(
        ballots=(
            Ballot(
                scores={"A": 2, "B": 4, "D": 1},
            ),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                scores={"A": 5, "B": 4, "C": 1},
            ),
        )
        * 5,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_scores.to_pickle(f"{filepath}/test_pkl_pp_scores.pkl")
    read_profile = PreferenceProfile.from_pickle(f"{filepath}/test_pkl_pp_scores.pkl")
    assert profile_scores == read_profile


def test_pkl_bijection_mixed():
    profile_mixed = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 5, "B": 4, "C": 1},
            ),
        )
        * 5,
        max_ranking_length=3,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_mixed.to_pickle(f"{filepath}/test_pkl_pp_mixed.pkl")
    read_profile = PreferenceProfile.from_pickle(f"{filepath}/test_pkl_pp_mixed.pkl")
    assert profile_mixed == read_profile
