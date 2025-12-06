from votekit.elections import Schulze, ElectionState
from votekit.pref_profile import (
    RankProfile,
    ScoreProfile,
    ProfileError,
)
from votekit.ballot import RankBallot, ScoreBallot
import pytest
import pandas as pd
import numpy as np
from time import time


# Wikipedia example for Schulze method
# https://en.wikipedia.org/wiki/Schulze_method
wikipedia_profile = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}, {"E"}, {"D"}])),
            weight=5,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"A"}, {"D"}, {"E"}, {"C"}, {"B"}])),
            weight=5,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"B"}, {"E"}, {"D"}, {"A"}, {"C"}])),
            weight=8,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"C"}, {"A"}, {"B"}, {"E"}, {"D"}])),
            weight=3,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"C"}, {"A"}, {"E"}, {"B"}, {"D"}])),
            weight=7,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}, {"D"}, {"E"}])),
            weight=2,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"D"}, {"C"}, {"E"}, {"B"}, {"A"}])),
            weight=7,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"E"}, {"B"}, {"A"}, {"D"}, {"C"}])),
            weight=8,
        ),
    ),
    max_ranking_length=5,
)

electowiki_profile = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(
                map(
                    frozenset,
                    [{"Memphis"}, {"Nashville"}, {"Chattanoga"}, {"Knoxville"}],
                )
            ),
            weight=42,
        ),
        RankBallot(
            ranking=tuple(
                map(
                    frozenset,
                    [{"Nashville"}, {"Chattanoga"}, {"Knoxville"}, {"Memphis"}],
                )
            ),
            weight=26,
        ),
        RankBallot(
            ranking=tuple(
                map(
                    frozenset,
                    [{"Chattanoga"}, {"Knoxville"}, {"Nashville"}, {"Memphis"}],
                )
            ),
            weight=15,
        ),
        RankBallot(
            ranking=tuple(
                map(
                    frozenset,
                    [{"Knoxville"}, {"Chattanoga"}, {"Nashville"}, {"Memphis"}],
                )
            ),
            weight=15,
        ),
    ),
    max_ranking_length=4,
)

profile_with_skips = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(map(frozenset, [{"C"}, {"D"}, {"A"}, {}])),
            weight=42,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"D"}, {"A"}, {}, {"C"}])),
            weight=26,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{"A"}, {}, {"D"}, {"C"}])),
            weight=15,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, [{}, {"A"}, {"D"}, {"C"}])),
            weight=15,
        ),
    ),
    max_ranking_length=4,
)

test_profile_limit_case = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(map(frozenset, ["A", "B", "C"])),
            weight=48,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, ["B", "C", "A"])),
            weight=3,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, ["C", "A", "B"])),
            weight=49,
        ),
    ),
    max_ranking_length=3,
)

borda_ambiguous_profile = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(map(frozenset, ["A", "B", "C"])),
            weight=48,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, ["B", "C", "A"])),
            weight=24,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, ["C", "A"])),
            weight=28,
        ),
    ),
    max_ranking_length=3,
)

dominating_ambiguous_profile = RankProfile(
    ballots=(
        RankBallot(
            ranking=tuple(map(frozenset, ["A", "B", "C", "D"])),
            weight=1,
        ),
        RankBallot(
            ranking=tuple(map(frozenset, ["A", "C", "B", "D"])),
            weight=1,
        ),
    ),
)

profile_tied_set = RankProfile(
    ballots=(
        RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}]))),
        RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}]))),
        RankBallot(ranking=tuple(map(frozenset, [{"B"}, {"A"}, {"C"}])), weight=2),
    ),
    max_ranking_length=3,
)


profile_cycle = RankProfile(
    ballots=(
        RankBallot(ranking=tuple(map(frozenset, ({"A"}, {"B"}, {"C"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"A"}, {"C"}, {"B"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"B"}, {"A"}, {"C"})))),
    ),
    max_ranking_length=3,
)

profile_tied_borda = RankProfile(
    ballots=(
        RankBallot(ranking=tuple(map(frozenset, ({"A"}, {"B"}, {"C"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"A"}, {"C"}, {"B"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"B"}, {"A"}, {"C"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"B"}, {"C"}, {"A"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"C"}, {"A"}, {"B"})))),
        RankBallot(ranking=tuple(map(frozenset, ({"C"}, {"B"}, {"A"})))),
    ),
    max_ranking_length=3,
)


def convert_to_fs_tuple(lst):
    """
    Convert a list of strings to a tuple of frozensets.
    """
    return tuple(map(frozenset, lst))


def test_wikipedia_example():
    """Test the example from Wikipedia's Schulze method page.

    Number of voters | Order of preference
    5                | A C B E D
    5                | A D E C B
    8                | B E D A C
    3                | C A B E D
    7                | C A E B D
    2                | C B A D E
    7                | D C E B A
    8                | E B A D C

    Expected winner is E according to the Wikipedia article.
    """
    e = Schulze(wikipedia_profile, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["E"])


def test_init():
    e = Schulze(electowiki_profile, m=1)
    assert e.get_elected() == (frozenset({"Nashville"}),)


def test_init_with_empty():
    e = Schulze(profile_with_skips, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["D", "A", "C"])


def test_init_with_empty_errors_out_when_too_many_elected():
    with pytest.raises(ValueError):
        Schulze(profile_with_skips, m=4)


def test_limit_case():
    e = Schulze(test_profile_limit_case, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["C", "A"])


def test_borda_ambiguous_profile_returns_lexicographic_order():
    e = Schulze(borda_ambiguous_profile, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["A"])
    e = Schulze(borda_ambiguous_profile, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B"])
    e = Schulze(borda_ambiguous_profile, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B", "C"])


def test_dominating_ambigous_profile_returns_lexicographic_order():
    e = Schulze(dominating_ambiguous_profile, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["A"])
    e = Schulze(dominating_ambiguous_profile, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B"])
    e = Schulze(dominating_ambiguous_profile, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B", "C"])


def test_tied_set():
    e = Schulze(profile_tied_set, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["A"])
    e = Schulze(profile_tied_set, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B"])
    e = Schulze(profile_tied_set, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B", "C"])


def test_profile_cycle():
    e = Schulze(profile_cycle, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["A"])
    e = Schulze(profile_cycle, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B"])
    e = Schulze(profile_cycle, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B", "C"])


def test_tied_borda():
    e = Schulze(profile_tied_borda, m=1)
    assert e.get_elected() == convert_to_fs_tuple(["A"])
    e = Schulze(profile_tied_borda, m=2)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B"])
    e = Schulze(profile_tied_borda, m=3)
    assert e.get_elected() == convert_to_fs_tuple(["A", "B", "C"])


def test_errors():
    with pytest.raises(ValueError, match="m must be strictly positive"):
        Schulze(profile_tied_set, m=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Schulze(profile_tied_set, m=4)

    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        Schulze(ScoreProfile(ballots=(ScoreBallot(scores={"A": 4}),)))  # type: ignore


@pytest.mark.slow
def test_large_set_timing():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ballots_tup = tuple(
        [
            RankBallot(
                ranking=tuple(
                    map(
                        lambda x: frozenset({x}),
                        np.random.permutation(list(alphabet))[
                            : np.random.randint(1, len(alphabet) + 1)
                        ],
                    )
                ),
                weight=np.random.randint(1, 100),
            )
            for _ in range(100_000)
        ]
    )

    prof = RankProfile(
        ballots=ballots_tup,
        max_ranking_length=26,
    )

    start = time()
    for _ in range(10):
        Schulze(
            prof,
            m=5,
        )
    end = time()

    assert (
        end - start < 120
    ), f"Schulze runtime took too long. Expected < 120 seconds, got {end - start} seconds."


states = [
    ElectionState(
        round_number=0,
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        elected=(frozenset(),),
        eliminated=(frozenset(),),
        tiebreaks={},
        scores={"C": 0, "B": 1, "A": 2},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"B"}), frozenset({"C"})),
        elected=(frozenset({"A"}),),
        eliminated=(frozenset(),),
        tiebreaks={frozenset({"A", "B"}): (frozenset({"A"}), frozenset({"B"}))},
        scores={},
    ),
]


def test_state_list():
    e = Schulze(profile_tied_set)
    assert e.election_states == states


def test_get_profile():
    e = Schulze(profile_tied_set)
    assert e.get_profile(0) == profile_tied_set


def test_get_step():
    e = Schulze(profile_tied_set)
    profile, state = e.get_step(1)
    assert profile.group_ballots(), state == (profile_tied_set, states[1])


def test_get_step_does_not_extend_election_states():
    e = Schulze(profile_tied_set)
    assert len(e.election_states) == 2

    profile, state = e.get_step(1)
    assert profile.group_ballots(), state == (profile_tied_set, states[1])
    assert len(e.election_states) == 2

    profile, state = e.get_step(1)
    assert len(e.election_states) == 2


def test_get_elected():
    e = Schulze(profile_tied_set)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}),)


def test_get_eliminated():
    e = Schulze(profile_tied_set)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Schulze(profile_tied_set)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (
        frozenset({"B"}),
        frozenset({"C"}),
    )


def test_get_ranking():
    e = Schulze(profile_tied_set)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = Schulze(profile_tied_set)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Remaining", "Remaining"], "Round": [1] * 3},
        index=["A", "B", "C"],
    )

    assert e.get_status_df(0).sort_index().equals(df_0)
    assert e.get_status_df(1).sort_index().equals(df_1)
