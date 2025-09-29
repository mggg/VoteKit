from votekit.representation_scores import winner_sets_r_representation_scores
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cvr_loaders import load_ranking_csv
from votekit.cleaning import remove_cand_rank_profile
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


def test_winner_sets_r_rep_score():
    assert winner_sets_r_representation_scores(profile, 2, 1) == {
        frozenset({"Moon", "Peter"}): 1,
        frozenset({"Moon", "Chris"}): 2 / 3,
        frozenset({"Chris", "Peter"}): 1 / 3,
    }

    assert winner_sets_r_representation_scores(profile, 2, 2) == {
        frozenset({"Moon", "Peter"}): 1,
        frozenset({"Moon", "Chris"}): 2 / 3,
        frozenset({"Chris", "Peter"}): 2 / 3,
    }

    assert winner_sets_r_representation_scores(profile, 2, 3) == {
        frozenset({"Moon", "Peter"}): 1,
        frozenset({"Moon", "Chris"}): 2 / 3,
        frozenset({"Chris", "Peter"}): 2 / 3,
    }


def test_winner_sets_r_rep_score_subset_cands():
    assert winner_sets_r_representation_scores(profile, 1, 1, ["Chris", "Peter"]) == {
        frozenset({"Peter"}): 1 / 3,
        frozenset({"Chris"}): 0,
    }

    assert winner_sets_r_representation_scores(profile, 1, 2, ["Chris", "Peter"]) == {
        frozenset({"Peter"}): 1 / 3,
        frozenset({"Chris"}): 1 / 3,
    }

    assert winner_sets_r_representation_scores(profile, 1, 3, ["Chris", "Peter"]) == {
        frozenset({"Peter"}): 2 / 3,
        frozenset({"Chris"}): 1 / 3,
    }


def test_winner_sets_r_rep_score_error_r():
    with pytest.raises(ValueError, match="r \\(0\\) must be at least 1."):
        winner_sets_r_representation_scores(profile, 1, 0)


def test_winner_sets_r_rep_score_error_m():
    with pytest.raises(
        ValueError, match="Number of seats m \\(0\\) must be at least 1."
    ):
        winner_sets_r_representation_scores(PreferenceProfile(), 0, 1)

    with pytest.raises(
        ValueError,
        match="Number of seats m \\(2\\) must be less than number of candidates \\(1\\).",
    ):
        winner_sets_r_representation_scores(PreferenceProfile(), 2, 1, ["Chris"])


@pytest.mark.slow
def test_winner_sets_r_rep_score_portland():
    profile = load_ranking_csv(
        CSV_DIR / "Portland_D1_Condensed.csv",
        rank_cols=[1, 2, 3, 4, 5, 6],
        header_row=0,
    )
    clean_profile = remove_cand_rank_profile("skipped", profile)

    score_dict = winner_sets_r_representation_scores(
        clean_profile,
        3,
        3,
        ["Candace Avalos", "Loretta Smith", "Jamie Dunphy", "Steph Routh"],
    )
    assert (
        round(
            score_dict[frozenset(["Candace Avalos", "Loretta Smith", "Jamie Dunphy"])],
            3,
        )
        == 0.709
    )
    assert (
        round(
            score_dict[frozenset(["Candace Avalos", "Steph Routh", "Jamie Dunphy"])], 3
        )
        == 0.636
    )
