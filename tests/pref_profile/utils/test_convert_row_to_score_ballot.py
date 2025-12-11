import pandas as pd
from votekit.ballot import ScoreBallot
from votekit.pref_profile.utils import convert_row_to_score_ballot


def test_convert_row_to_score_ballot():
    b = convert_row_to_score_ballot(
        pd.Series(
            {
                "Weight": 2,
                "Voter Set": {"Chris"},
                "A": 1,
                "B": 2.1,
                "CD": pd.NA,
            }
        ),
        candidates=["A", "B", "CD"],
    )

    assert isinstance(b, ScoreBallot)
    assert b.weight == 2
    assert b.scores == {"A": 1, "B": 2.1}
    assert b.voter_set == {"Chris"}

    b = convert_row_to_score_ballot(
        pd.Series(
            {
                "Weight": 2,
                "Voter Set": {"Chris"},
                "A": pd.NA,
                "B": pd.NA,
                "CD": pd.NA,
            }
        ),
        candidates=["A", "B", "CD"],
    )

    assert isinstance(b, ScoreBallot)
    assert b.weight == 2
    assert b.scores is None
    assert b.voter_set == {"Chris"}
