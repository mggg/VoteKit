from votekit.representation_scores import r_representation_score
from votekit import PreferenceProfile, Ballot
import pytest

profile = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"Moon"}, {"Chris"}, {"Peter"})),
        Ballot(ranking=({"Peter"},)),
        Ballot(ranking=({"Moon"},)),
    ),
    candidates=["Moon", "Peter", "Chris", "Mala"],
)


def test_r_rep_score_one_cand():
    assert r_representation_score(profile, 1, ["Chris"]) == 0
    assert r_representation_score(profile, 2, ["Chris"]) == 1 / 3
    assert r_representation_score(profile, 3, ["Chris"]) == 1 / 3

    assert r_representation_score(profile, 3, ["Mala"]) == 0


def test_r_rep_score_multi_cand():
    assert (
        r_representation_score(
            profile,
            1,
            ["Chris", "Peter"],
        )
        == 1 / 3
    )
    assert r_representation_score(profile, 2, ["Peter", "Chris"]) == 2 / 3
    assert r_representation_score(profile, 3, ["Chris", "Peter", "Mala"]) == 2 / 3


def test_r_rep_score_error():
    with pytest.raises(ValueError, match="r \(0\) must be at least 1."):
        r_representation_score(PreferenceProfile(), 0, [])
