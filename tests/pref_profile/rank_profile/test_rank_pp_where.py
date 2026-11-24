import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.constants import END_OF_RANKING_SET
from votekit.pref_profile import RankProfile


@pytest.fixture
def labeled_profile() -> RankProfile:
    """Ballots A > B and B > A whose Ballot Index is 10 and 20, not their row positions 0 and 1."""
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Ranking_2": [frozenset({"B"}), frozenset({"A"})],
            "Voter Set": [frozenset(), frozenset()],
            "Weight": [2.0, 1.0],
        },
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    return RankProfile(df=df, candidates=["A", "B"], max_ranking_length=2)


def test_where_returns_the_ballot_index_label(labeled_profile):
    b_first = labeled_profile.to_df().iloc[1]

    assert labeled_profile.where(b_first).tolist() == [20]


def test_where_finds_every_ballot_with_a_ranking(labeled_profile):
    profile = labeled_profile + labeled_profile
    a_first = labeled_profile.to_df().iloc[0]

    assert profile.where(a_first).tolist() == [0, 2]


def test_where_finds_no_ballot_for_an_absent_ranking(labeled_profile):
    row = {"Ranking_1": frozenset({"A"}), "Ranking_2": END_OF_RANKING_SET}

    assert labeled_profile.where(pd.Series(row)).tolist() == []


def test_where_ignores_weight_and_voter_set(labeled_profile):
    row = labeled_profile.to_df().iloc[0].copy()
    row["Weight"] = 99.0
    row["Voter Set"] = frozenset({"v1"})

    assert labeled_profile.where(row).tolist() == [10]


def test_where_rejects_a_row_missing_a_ranking_column(labeled_profile):
    with pytest.raises(KeyError):
        labeled_profile.where(pd.Series({"Ranking_1": frozenset({"A"})}))


def test_where_rejects_a_ranking_entry_that_is_not_a_set():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}])])

    with pytest.raises(TypeError, match="sets or frozensets"):
        profile.where(pd.Series({"Ranking_1": "A"}))
