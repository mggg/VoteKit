from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest
from pydantic_core import ValidationError


def test_pp_candidate_list():
    with pytest.raises(ValidationError) as e:
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset(), frozenset())),),
            candidates=["Peter", "Peter"],
        )

    assert "All candidates must be unique." in str(e.value)

    with pytest.raises(ValidationError) as e:
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"Chris"}), frozenset())),),
            candidates=["Peter"],
        )

    assert "Candidate Chris found in ballot " in str(e.value)

    with pytest.raises(ValidationError) as e:
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset({"Ranking_0"}), frozenset())),),
            candidates=["Peter", "Ranking_0"],
        )

    assert (
        "Candidate Ranking_0 must not share name with" " ranking columns: Ranking_i."
    ) in str(e.value)


def test_pp_excede_ranking_length():
    with pytest.raises(
        ValidationError,
    ) as e:
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset(), frozenset())),), max_ranking_length=1
        )
    assert "Max ballot length 1 given but " in str(e.value)


def test_pp_contains_ranking():
    with pytest.raises(ValidationError) as e:
        PreferenceProfile(
            ballots=(Ballot(ranking=(frozenset(),)),), contains_rankings=False
        )

    assert "but contains_rankings is set to False." in str(e.value)

    with pytest.raises(
        ValidationError,
    ) as e:
        PreferenceProfile(ballots=(Ballot(),), contains_rankings=True)

    assert "contains_rankings is True but we found no ballots with rankings." in str(
        e.value
    )


def test_pp_contains_scores():
    with pytest.raises(
        ValidationError,
    ) as e:
        PreferenceProfile(ballots=(Ballot(scores={"A": 2}),), contains_scores=False)

    assert "but contains_scores is set to False." in str(e.value)

    with pytest.raises(
        ValidationError,
    ) as e:
        PreferenceProfile(ballots=(Ballot(),), contains_scores=True)

    assert "contains_scores is True but we found no ballots with scores." in str(
        e.value
    )
