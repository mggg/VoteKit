import pandas as pd
from votekit.ballot import RankBallot
from votekit.pref_profile.utils import convert_row_to_rank_ballot
import pytest


def test_convert_row_to_rank_ballot():
    b = convert_row_to_rank_ballot(
        pd.Series(
            {
                "Weight": 2,
                "Voter Set": {"Chris"},
                "Ranking_1": frozenset({"A"}),
                "Ranking_2": frozenset({"B"}),
                "Ranking_3": frozenset({"CD"}),
            }
        ),
        max_ranking_length=3,
    )

    assert isinstance(b, RankBallot)
    assert b.weight == 2
    assert b.ranking == (frozenset({"A"}), frozenset({"B"}), frozenset({"CD"}))
    assert b.voter_set == {"Chris"}

    b = convert_row_to_rank_ballot(
        pd.Series(
            {
                "Weight": 2,
                "Voter Set": {"Chris"},
            }
        ),
        max_ranking_length=0,
    )

    assert isinstance(b, RankBallot)
    assert b.weight == 2
    assert b.ranking is None
    assert b.voter_set == {"Chris"}


def test_convert_row_to_rank_ballot_errors():
    with pytest.raises(ValueError, match="Row has improper ranking columns:"):
        convert_row_to_rank_ballot(
            pd.Series(
                {
                    "Weight": 2,
                    "Voter Set": {"Chris"},
                    "Ranking_1": frozenset("A"),
                    "Ranking_3": frozenset("C"),
                }
            ),
            max_ranking_length=3,
        )

    with pytest.raises(
        ValueError,
        match="has '~' between valid ranking positions. "
        "'~' values can only trail on a ranking.",
    ):
        convert_row_to_rank_ballot(
            pd.Series(
                {
                    "Weight": 2,
                    "Voter Set": {"Chris"},
                    "Ranking_1": frozenset({"AD"}),
                    "Ranking_2": frozenset("~"),
                    "Ranking_3": frozenset("C"),
                }
            ),
            max_ranking_length=3,
        )
