from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import ProfileError, RankProfile


def test_pp_group_ballots_ranking():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1, voter_set={"Chris"}),
            RankBallot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                voter_set={"Moon", "Peter"},
            ),
        ),
        candidates=("A", "B", "C", "D"),
    )

    pp = profile.group_ballots()
    assert pp.ballots == (
        RankBallot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=5,
            voter_set={"Chris", "Moon", "Peter"},
        ),
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fraction_weights", [False, True])
def test_matrix_grouping_preserves_first_appearance_weights_and_voters(seed, fraction_weights):
    rng = np.random.default_rng(seed)
    candidate_sets = [
        frozenset(),
        frozenset({"~"}),
        frozenset({1}),
        frozenset({"A"}),
        frozenset({1, "A"}),
    ]
    rows = rng.integers(0, len(candidate_sets), size=(40, 3))
    rankings = np.empty(rows.shape, dtype=object)
    for position, row in enumerate(rows):
        rankings[position] = [candidate_sets[index] for index in row]
    source = pd.DataFrame(rankings, columns=["Ranking_1", "Ranking_2", "Ranking_3"])
    source["Weight"] = pd.Series(
        [
            Fraction(int(weight), 3) if fraction_weights else float(weight)
            for weight in rng.integers(0, 4, len(source))
        ]
    )
    source["Voter Set"] = pd.Series(
        [frozenset({f"voter_{i % 5}"}) if i % 3 else frozenset() for i in range(len(source))]
    )
    source.index = pd.Index(np.arange(len(source)) // 2 + 10, name="Ballot Index")
    profile = RankProfile(df=source, candidates=("unused", "A", 1), max_ranking_length=3)

    ranking_columns = list(profile._matrix.ranking_columns)
    expected_weights = {}
    expected_voters = {}
    for ranking, weight, voters in zip(
        map(tuple, rankings), source["Weight"], source["Voter Set"], strict=True
    ):
        expected_weights[ranking] = expected_weights.get(ranking, 0) + weight
        expected_voters.setdefault(ranking, set()).update(voters)

    expected = pd.DataFrame(list(expected_weights), columns=ranking_columns)
    expected["Weight"] = list(expected_weights.values())
    expected["Voter Set"] = pd.Series(
        [frozenset(voters) for voters in expected_voters.values()], dtype=object
    )
    expected.index.name = "Ballot Index"
    expected_profile = RankProfile(df=expected, candidates=profile.candidates, max_ranking_length=3)

    grouped = profile.group_ballots()

    pd.testing.assert_frame_equal(grouped.to_df(), expected_profile.to_df())
    assert grouped.candidates == expected_profile.candidates
    assert grouped.candidates_cast == expected_profile.candidates_cast
    assert grouped.total_ballot_wt == expected_profile.total_ballot_wt
    assert grouped.max_ranking_length == 3
    assert grouped.max_candidates_ranked == expected_profile.max_candidates_ranked
    pd.testing.assert_frame_equal(
        profile.to_df(), source[["Ranking_1", "Ranking_2", "Ranking_3", "Voter Set", "Weight"]]
    )


@pytest.mark.parametrize("num_ballots", [0, 2])
def test_grouping_needs_no_dataframe_export_or_public_profile_constructor(monkeypatch, num_ballots):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=Fraction(1, 3))] * num_ballots,
        candidates=("A", "unused"),
        max_ranking_length=3,
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Grouping should use matrix storage directly")

    with monkeypatch.context() as patch:
        patch.setattr(RankProfile, "to_df", unexpected_call)
        patch.setattr(RankProfile, "__init__", unexpected_call)
        grouped = profile.group_ballots()

    expected_ballots = (RankBallot(ranking=[{"A"}], weight=Fraction(2, 3)),) if num_ballots else ()
    assert grouped.ballots == expected_ballots
    assert grouped.candidates == profile.candidates
    assert grouped.max_ranking_length == 3
    assert grouped._matrix.voter_weight_data.voter_sets is None


@pytest.mark.parametrize("num_ballots", [0, 3])
def test_grouping_without_ranking_columns(num_ballots):
    profile = RankProfile(ballots=[RankBallot(weight=2, voter_set={"voter"})] * num_ballots)

    grouped = profile.group_ballots()

    assert grouped.num_ballots == (1 if num_ballots else 0)
    assert grouped.total_ballot_wt == 2 * num_ballots
    assert grouped.candidates_cast == ()
    assert grouped.max_ranking_length == grouped.max_candidates_ranked == 0
    if num_ballots:
        assert grouped.ballots == (RankBallot(weight=6, voter_set={"voter"}),)


def test_grouping_rejects_weight_overflow():
    with np.errstate(over="ignore"):
        profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=1e308)] * 2)

    with pytest.raises(ProfileError, match="infinite"):
        profile.group_ballots()


def test_grouping_keeps_zero_weight_candidates_out_of_candidates_cast():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}]), RankBallot(ranking=[{"B"}], weight=0)],
    )

    grouped = profile.group_ballots()

    assert grouped.candidates == grouped.candidates_cast == ("A",)
    assert grouped.num_ballots == 2
    assert any(ballot.ranking == (frozenset({"B"}),) for ballot in grouped.ballots)


@pytest.mark.parametrize("fraction_weights", [False, True])
def test_grouping_retains_unique_voters_and_unions_repeated_rankings(fraction_weights):
    weights = [Fraction(n, 3) if fraction_weights else n / 3 for n in (1, 2, 3)]
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"B"}], weight=weights[0], voter_set={"Alice"}),
            RankBallot(ranking=[{"A"}], weight=weights[1], voter_set={"Bob"}),
            RankBallot(ranking=[{"B"}], weight=weights[2], voter_set={"Alice", "Carol"}),
        ],
    )
    source = profile.to_df()

    grouped = profile.group_ballots()

    assert grouped.ballots == (
        RankBallot(ranking=[{"B"}], weight=weights[0] + weights[2], voter_set={"Alice", "Carol"}),
        RankBallot(ranking=[{"A"}], weight=weights[1], voter_set={"Bob"}),
    )
    pd.testing.assert_frame_equal(profile.to_df(), source)
