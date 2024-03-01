from fractions import Fraction
import pandas as pd
import pytest
from unittest.mock import MagicMock

from votekit.ballot import Ballot  # type: ignore
from votekit.election_state import ElectionState  # type: ignore
from votekit.pref_profile import PreferenceProfile  # type: ignore


# TODO: use Scottish 3-cand ward_03 data,

b1 = Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(250, 1))
b2 = Ballot(ranking=[{"B"}, {"A"}, {"C"}], weight=Fraction(200, 1))
b3 = Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(100, 1))

ballots_2 = [b1, b2, b3]
pref_0 = PreferenceProfile(ballots=ballots_2)
pref_1 = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(250, 1)),
        Ballot(ranking=[{"B"}, {"A"}], weight=Fraction(300, 1)),
    ]
)
pref_2 = PreferenceProfile(ballots=[Ballot(ranking=[{"A"}], weight=Fraction(274, 1))])
round_0 = ElectionState(
    curr_round=0,
    elected=[],
    eliminated_cands=[],
    remaining=[{"A", "B", "C"}],
    profile=pref_0,
    previous=None,
)
round_1 = ElectionState(
    curr_round=1,
    elected=[],
    eliminated_cands=[{"C"}],
    remaining=[{"B", "A"}],
    profile=pref_1,
    previous=round_0,
)
round_2 = ElectionState(
    curr_round=2,
    elected=[{"B"}],
    eliminated_cands=[],
    remaining=[{"A"}],
    profile=pref_2,
    previous=round_1,
)

rounds = [round_0, round_1, round_2]
rds = [0, 1, 2]
elects = [[], [], [{"B"}]]
elims = [[], [{"C"}], []]
remains = [[{"A", "B", "C"}], [{"B", "A"}], [{"A"}]]
wins = [[], [], [{"B"}]]
los = [[], [{"C"}], [{"C"}]]
ranks = [[{"A", "B", "C"}], [{"B", "A"}, {"C"}], [{"B"}, {"A"}, {"C"}]]


correct_status = pd.DataFrame(
    {
        "Candidate": ["A", "B", "C"],
        "Status": ["Remaining", "Elected", "Eliminated"],
        "Round": [2, 2, 1],
    }
)


def test_get_attributes():
    for i in range(3):
        assert rounds[i].curr_round == rds[i]
        assert rounds[i].elected == elects[i]
        assert rounds[i].eliminated_cands == elims[i]
        assert rounds[i].remaining == remains[i]


def test_lists():
    for i in range(3):
        assert rounds[i].winners() == wins[i]
        assert rounds[i].eliminated() == los[i]
        assert rounds[i].rankings() == ranks[i]


def test_changed_rankings():
    assert rounds[1].changed_rankings() == {"C": (0, 2)}


def test_get_all_winners():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    first_winners = first.winners()
    assert first_winners == [{"A"}, {"B"}]

    second_winners = second.winners()
    assert second_winners == [{"A"}, {"B"}, {"D"}]


def test_round_previous():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    results = second.round_outcome(round=1)
    assert results == {
        "Elected": [{"A"}, {"B"}],
        "Eliminated": [{"C"}],
        "Remaining": [],
    }


def test_round_outcome_error():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )

    with pytest.raises(ValueError):
        first.round_outcome(round=4)


def test_elimination_order():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )
    third = ElectionState(
        curr_round=3,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"F"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=second,
    )
    fourth = ElectionState(
        curr_round=4,
        elected=[{"D"}],
        eliminated_cands=[{"G"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=third,
    )

    elims = fourth.eliminated()
    assert elims == [{"G"}, {"F"}, {"E"}, {"C"}]


def test_ranking_no_remaining():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    rank = second.rankings()
    assert rank == [{"A"}, {"B"}, {"D"}, {"E"}, {"C"}]


def test_ranking_w_remaing():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        remaining=[{"F"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}, {"F"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    rank = second.rankings()
    assert rank == [{"A"}, {"B"}, {"D"}, {"F"}, {"E"}, {"C"}]


def test_status_df_post_election():
    df = round_2.status()
    assert (
        df.sort_values(by="Candidate", ascending=True)
        .reset_index(drop=True)
        .equals(correct_status)
    )


def test_status_df_one_round():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        remaining=[{"F"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    df = first.status()
    assert (
        df.sort_values(by="Candidate", ascending=True)
        .reset_index(drop=True)
        .equals(
            pd.DataFrame(
                {
                    "Candidate": ["A", "B", "C", "F"],
                    "Status": ["Elected", "Elected", "Eliminated", "Remaining"],
                    "Round": [1, 1, 1, 1],
                }
            )
        )
    )


def test_status_no_remaining():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        remaining=[{"F"}],
        eliminated_cands=[{"C"}],
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}, {"F"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    correct = pd.DataFrame(
        {
            "Candidate": ["A", "B", "C", "D", "E", "F"],
            "Status": [
                "Elected",
                "Elected",
                "Eliminated",
                "Elected",
                "Eliminated",
                "Elected",
            ],
            "Round": [1, 1, 1, 2, 2, 2],
        }
    )

    df = second.status().sort_values(by="Candidate", ascending=True)
    assert df.reset_index(drop=True).equals(correct)


def test_status_missing_fields():
    rd = ElectionState(
        curr_round=1,
        elected=[{"D"}, {"F"}],
        eliminated_cands=[{"E"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=None,
    )
    df = rd.status().sort_values(by="Candidate", ascending=True)
    assert df.reset_index(drop=True).equals(
        pd.DataFrame(
            {
                "Candidate": ["D", "E", "F"],
                "Status": ["Elected", "Eliminated", "Elected"],
                "Round": [1, 1, 1],
            }
        )
    )


def test_to_dict():
    rd = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        remaining=[{"D"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=None,
    )

    expected = {
        "elected": ["A", "B"],
        "eliminated": ["C"],
        "remaining": ["D"],
        "ranking": ["A", "B", "D", "C"],
    }

    assert rd.to_dict() == expected


def test_to_dict_keep():
    rd = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        eliminated_cands=[{"C"}],
        remaining=[{"D"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=None,
    )
    assert rd.to_dict(keep=["elected"]) == {"elected": ["A", "B"]}


def test_to_dict_maintain_ties():
    rd = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B", "E"}],
        eliminated_cands=[{"C"}],
        remaining=[{"D"}],
        profile=MagicMock(spec=PreferenceProfile),
        previous=None,
    )

    results_dict = rd.to_dict(keep=["elected"])

    # tuples ('E', 'B') and ('B', 'E') represent same tied ranking
    assert results_dict == {"elected": ["A", ("B", "E")]} or results_dict == {
        "elected": ["A", ("E", "B")]
    }


def test_get_scores():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        remaining=[{"F"}],
        eliminated_cands=[{"C"}],
        scores={"A": 4, "B": 6, "F": 3, "C": 9},
        profile=MagicMock(spec=PreferenceProfile),
    )
    second = ElectionState(
        curr_round=2,
        elected=[{"D"}, {"F"}],
        eliminated_cands=[{"E"}],
        scores={"D": 6, "F": 3, "E": 9},
        profile=MagicMock(spec=PreferenceProfile),
        previous=first,
    )

    assert second.get_scores(1) == {"A": 4, "B": 6, "F": 3, "C": 9}


def test_score_error():
    first = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"B"}],
        remaining=[{"F"}],
        eliminated_cands=[{"C"}],
        scores={"A": 4, "B": 6, "F": 3, "C": 9},
        profile=MagicMock(spec=PreferenceProfile),
    )
    with pytest.raises(ValueError):
        first.get_scores(4)
