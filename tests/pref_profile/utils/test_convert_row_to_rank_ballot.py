import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.pref_profile.utils import _df_to_rank_ballot_tuple, convert_row_to_rank_ballot


@pytest.mark.parametrize(
    "ballots",
    [
        (),
        (
            RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"Chris"}),
            RankBallot(weight=0),
        ),
    ],
)
def test_dataframe_to_ballots_preserves_order_weights_and_empty_rankings(ballots):
    profile = RankProfile(ballots=ballots, candidates=("A", "B"))

    result = _df_to_rank_ballot_tuple(
        profile.to_df(), profile.candidates, profile.max_ranking_length
    )

    assert result == ballots


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
        match="has '~' between valid ranking positions. '~' values can only trail on a ranking.",
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
