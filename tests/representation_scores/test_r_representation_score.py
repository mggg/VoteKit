from votekit.representation_scores import r_representation_score
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cvr_loaders import load_ranking_csv
from votekit.cleaning import remove_cand_from_rank_profile
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data/csv/"

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
    with pytest.raises(ValueError, match="must be at least 1."):
        r_representation_score(PreferenceProfile(), 0, [])


def test_r_rep_score_warning():
    with pytest.warns(
        UserWarning, match="are not found in the profile's candidate list:"
    ):
        r_representation_score(profile, 3, ["David", "Chris", "Jeanne", "Mala"])


@pytest.mark.slow
def test_r_rep_score_portland():
    profile = load_ranking_csv(
        CSV_DIR / "Portland_D1_Condensed.csv",
        rank_cols=[1, 2, 3, 4, 5, 6],
        header_row=0,
    )
    clean_profile = remove_cand_from_rank_profile("skipped", profile)

    assert (
        round(
            r_representation_score(
                clean_profile, 3, ["Candace Avalos", "Loretta Smith", "Jamie Dunphy"]
            ),
            3,
        )
        == 0.709
    )
