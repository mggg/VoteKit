from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ProfileError, ScoreProfile


def test_pp_group_ballots_scores():
    profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2, voter_set={"Chris"}),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2, voter_set={"Moon", "Peter"}),
            ScoreBallot(
                scores={"D": 2, "E": 2},
                weight=2,
            ),
            ScoreBallot(),
        )
    )

    pp = profile.group_ballots()
    assert set(pp.ballots) == set(
        (
            ScoreBallot(
                scores={"D": 2, "E": 2},
                weight=6,
                voter_set={"Chris", "Moon", "Peter"},
            ),
            ScoreBallot(),
        )
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("fraction_weights", [False, True])
def test_score_grouping_preserves_first_appearance_weights_voters_and_metadata(
    seed, fraction_weights
):
    rng = np.random.default_rng(seed)
    scores = rng.choice([np.nan, 0.0, 1.0, 2.0], size=(40, 2))
    weights = [
        Fraction(int(n), 3) if fraction_weights else int(n) / 4
        for n in rng.integers(0, 4, len(scores))
    ]
    voters = [frozenset({f"voter_{i % 5}"}) if i % 3 else frozenset() for i in range(len(scores))]
    source = pd.DataFrame(scores, columns=["B", "A"])
    source["unused"] = np.nan
    source["Weight"] = pd.Series(weights)
    source["Voter Set"] = pd.Series(voters, dtype=object)
    source.index = pd.Index(np.arange(len(source)) // 2 + 10, name="Ballot Index")
    profile = ScoreProfile(df=source, candidates=("unused", "B", "A"))
    snapshot = profile.to_df()

    expected_weights = {}
    expected_voters = {}
    expected_scores = {}
    for row, weight, voter_set in zip(scores, weights, voters, strict=True):
        key = tuple(None if np.isnan(score) else score for score in row)
        expected_weights[key] = expected_weights.get(key, 0) + weight
        expected_voters.setdefault(key, set()).update(voter_set)
        expected_scores.setdefault(key, row)

    expected = pd.DataFrame(list(expected_scores.values()), columns=["B", "A"])
    expected["unused"] = np.nan
    expected["Voter Set"] = pd.Series(
        [frozenset(voter_set) for voter_set in expected_voters.values()], dtype=object
    )
    expected["Weight"] = list(expected_weights.values())
    expected.index.name = "Ballot Index"

    grouped = profile.group_ballots()

    pd.testing.assert_frame_equal(grouped.to_df(), expected)
    pd.testing.assert_frame_equal(profile.to_df(), snapshot)
    assert grouped.candidates == profile.candidates
    assert grouped.candidates_cast == profile.candidates_cast
    assert grouped.total_ballot_wt == profile.total_ballot_wt
    assert grouped.num_ballots == len(expected)
    assert grouped._matrix.score_columns == profile._matrix.score_columns
    assert grouped._matrix.voter_weight_data.ballot_index is None
    assert not grouped._matrix.scores.flags.writeable
    assert not grouped._matrix.voter_weight_data.weights.flags.writeable


@pytest.mark.parametrize("num_ballots", [0, 2])
def test_score_grouping_needs_no_export_or_public_constructor(monkeypatch, num_ballots):
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"B": 2}, weight=Fraction(1, 3))] * num_ballots,
        candidates=("unused", "B", "A"),
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Grouping should use matrix storage directly")

    with monkeypatch.context() as patch:
        patch.setattr(ScoreProfile, "to_df", unexpected_call)
        patch.setattr(type(profile._matrix), "to_df", unexpected_call)
        patch.setattr(ScoreProfile, "__init__", unexpected_call)
        grouped = profile.group_ballots()

    assert type(grouped) is ScoreProfile
    assert grouped.candidates == profile.candidates
    assert grouped.candidates_cast == (("B",) if num_ballots else ())
    assert grouped.total_ballot_wt == Fraction(num_ballots, 3)
    assert grouped.num_ballots == (1 if num_ballots else 0)
    assert grouped._matrix.voter_weight_data.voter_sets is None
    with pytest.raises(AttributeError, match="frozen"):
        grouped.candidates = ()


@pytest.mark.parametrize("num_ballots", [0, 3])
def test_score_grouping_without_score_columns(num_ballots):
    profile = ScoreProfile(ballots=[ScoreBallot(weight=2, voter_set={"voter"})] * num_ballots)

    grouped = profile.group_ballots()

    assert grouped.num_ballots == (1 if num_ballots else 0)
    assert grouped.total_ballot_wt == 2 * num_ballots
    assert grouped.candidates_cast == ()
    assert grouped._matrix.scores.shape == ((1 if num_ballots else 0), 0)
    if num_ballots:
        assert grouped.ballots == (ScoreBallot(weight=6, voter_set={"voter"}),)


def test_score_grouping_rejects_weight_overflow():
    with np.errstate(over="ignore"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1}, weight=1e308)] * 2)

    with pytest.raises(ProfileError, match="infinite"):
        profile.group_ballots()


def test_score_grouping_candidates_cast_requires_positive_weight_and_score():
    source = pd.DataFrame(
        {"B": [1.0, np.nan], "A": [0.0, np.nan], "C": [np.nan, 2.0], "Weight": [1.0, 0.0]}
    )
    source["unused"] = np.nan
    source["Voter Set"] = pd.Series([frozenset(), frozenset()], dtype=object)
    source.index.name = "Ballot Index"
    profile = ScoreProfile(df=source, candidates=("B", "A", "unused"))

    grouped = profile.group_ballots()

    assert grouped.candidates_cast == ("B",)
    assert grouped.candidates == profile.candidates
    assert grouped.num_ballots == 2
    pd.testing.assert_frame_equal(grouped.to_df(), profile.to_df())
