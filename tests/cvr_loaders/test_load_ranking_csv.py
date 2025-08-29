from pandas.errors import EmptyDataError
from pathlib import Path
import pytest
import re
from votekit.cvr_loaders import load_csv, load_ranking_csv
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data/csv/load_ranking_csv"


def test_empty_csv():
    with pytest.raises(EmptyDataError):
        load_ranking_csv(
            CSV_DIR / "empty.csv",
            rank_cols=[0, 1, 2],
        )


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_ranking_csv(
            "fake_path.csv",
            rank_cols=[0, 1, 2],
        )


def test_candidate_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was provided in candidates ['Chris', 'Peter'] but "
            "not found in the csv."
        ),
    ):
        load_ranking_csv(
            CSV_DIR / "missing_candidate_error.csv",
            rank_cols=[0],
            candidates=["Chris", "Peter"],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was found in the csv but not provided in candidates ['Peter']."
        ),
    ):
        load_ranking_csv(
            CSV_DIR / "extra_candidate_error.csv",
            rank_cols=[0, 1],
            candidates=["Peter"],
        )


def test_id_and_weight_error():
    with pytest.raises(
        ValueError, match="Only one of weight_col and id_col can be provided"
    ):
        load_ranking_csv(
            CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=3, id_col=4
        )


def test_distinct_col_error():
    with pytest.raises(
        ValueError,
        match=re.escape(f"ID column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=1)

    with pytest.raises(
        ValueError,
        match=re.escape(f"Weight column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=1)


def test_columns_out_of_range_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[-1])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=-1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=-1)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[5])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=5)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=5)


def test_numerical_weights_error():
    with pytest.raises(
        ValueError, match=f"Weight {'a'} in row {0} must be able to be cast to float."
    ):
        load_ranking_csv(
            CSV_DIR / "non_numeric_weight.csv", rank_cols=[0, 1, 2], weight_col=3
        )

    with pytest.raises(ValueError, match=f"No weight provided in row {1}."):
        load_ranking_csv(
            CSV_DIR / "no_weight_provided.csv", rank_cols=[0, 1, 2], weight_col=3
        )


def test_header_error():
    with pytest.raises(ValueError, match="Header row -1 must be non-negative."):
        load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], header_row=-1)


def test_load_ranking_csv():
    profile = load_ranking_csv(
        CSV_DIR / "valid_cvr.csv",
        rank_cols=[0, 1, 2],
    )

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Peter"}, {"Chris"})),
            Ballot(ranking=({"Chris"}, {"Moon"})),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"})),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"})),
            Ballot(
                ranking=(
                    frozenset(),
                    frozenset(),
                    {"Moon"},
                )
            ),
        )
    )

    assert profile == true_profile
    # checks that converting from df to ballot list works
    assert profile.ballots == true_profile.ballots
    assert set(profile.candidates) == set(true_profile.candidates)


def test_load_ranking_csv_w_weight():
    profile = load_ranking_csv(
        CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=3
    )

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"Peter"}, {"Chris"}),
                weight=1,
            ),
            Ballot(
                ranking=({"Chris"}, {"Moon"}),
                weight=1,
            ),
            Ballot(
                ranking=({"Jeanne"}, {"David"}, {"Mala"}),
                weight=1.5,
            ),
            Ballot(
                ranking=({"Jeanne"}, {"David"}, {"Mala"}),
                weight=0.5,
            ),
            Ballot(
                ranking=(
                    frozenset(),
                    frozenset(),
                    {"Moon"},
                ),
                weight=2,
            ),
        )
    )

    assert profile == true_profile
    # checks that converting from df to ballot list works
    assert profile.ballots == true_profile.ballots


def test_load_ranking_csv_w_id():
    profile = load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=4)

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Peter"}, {"Chris"}), weight=1, voter_set={"X40"}),
            Ballot(ranking=({"Chris"}, {"Moon"}), weight=1, voter_set={"X31"}),
            Ballot(
                ranking=({"Jeanne"}, {"David"}, {"Mala"}), weight=1, voter_set={"X29"}
            ),
            Ballot(
                ranking=({"Jeanne"}, {"David"}, {"Mala"}), weight=1, voter_set={"X400"}
            ),
            Ballot(
                ranking=(
                    frozenset(),
                    frozenset(),
                    {"Moon"},
                ),
                weight=1,
                voter_set={"X31"},
            ),
        )
    )

    assert profile == true_profile
    # checks that converting from df to ballot list works
    assert profile.ballots == true_profile.ballots


def test_load_ranking_csv_header():
    profile = load_ranking_csv(
        CSV_DIR / "valid_cvr_w_header.csv", rank_cols=[0, 1, 2], id_col=4, header_row=0
    )

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Peter"}, {"Chris"}), voter_set={"X40"}),
            Ballot(ranking=({"Chris"}, {"Moon"}), voter_set={"X31"}),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"}), voter_set={"X29"}),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"}), voter_set={"X400"}),
            Ballot(
                ranking=(
                    frozenset(),
                    frozenset(),
                    {"Moon"},
                ),
                voter_set={"X31"},
            ),
        )
    )

    assert profile == true_profile
    # checks that converting from df to ballot list works
    assert profile.ballots == true_profile.ballots

    profile = load_ranking_csv(
        CSV_DIR / "valid_cvr_w_header.csv",
        rank_cols=[0, 1, 2],
        weight_col=3,
        header_row=0,
    )

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Peter"}, {"Chris"}), weight=1),
            Ballot(ranking=({"Chris"}, {"Moon"}), weight=1),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"}), weight=1.5),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"}), weight=0.5),
            Ballot(
                ranking=(
                    frozenset(),
                    frozenset(),
                    {"Moon"},
                ),
                weight=2,
            ),
        )
    )

    assert profile == true_profile
    # checks that converting from df to ballot list works
    assert profile.ballots == true_profile.ballots


def test_deprecation():
    with pytest.warns(
        DeprecationWarning,
        match="This function is being deprecated in March "
        "2026. The correct function call is now load_ranking_csv.",
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2])


def test_print(capsys):
    load_ranking_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], print_profile=True)

    captured = capsys.readouterr()

    assert "Profile contains rankings:" in captured.out

    load_ranking_csv(
        CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], print_profile=False
    )

    captured = capsys.readouterr()

    assert "Profile contains rankings:" not in captured.out
