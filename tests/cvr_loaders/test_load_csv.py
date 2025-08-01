from pandas.errors import EmptyDataError

# from urllib.error import URLError
from pathlib import Path
import pytest
import re
from votekit.cvr_loaders import load_csv
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data/csv/load_csv"


def test_empty_csv():
    with pytest.raises(EmptyDataError):
        load_csv(CSV_DIR / "empty.csv")


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_csv("fake_path.csv")


# occasionally stalls the test suite
# def test_invalid_url():
#     with pytest.raises(URLError):
#         load_csv("http://example.com/bad.csv")


# TODO bad encoding
# TODO parser error
# TODO HTTPError


def test_one_column_error():
    with pytest.raises(
        ValueError,
        match="CSV has only one column but one of weight_col or id_col is provided.",
    ):
        load_csv(CSV_DIR / "one_column_error.csv", weight_col=0)
    with pytest.raises(
        ValueError,
        match="CSV has only one column but one of weight_col or id_col is provided.",
    ):
        load_csv(CSV_DIR / "one_column_error.csv", id_col=0)


def test_candidate_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was provided in candidates ['Chris', 'Peter'] but "
            "not found in the csv."
        ),
    ):
        load_csv(CSV_DIR / "missing_candidate_error.csv", candidates=["Chris", "Peter"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was found in the csv but not provided in candidates ['Peter']."
        ),
    ):
        load_csv(CSV_DIR / "extra_candidate_error.csv", candidates=["Peter"])


def test_id_and_weight_error():
    with pytest.raises(
        ValueError, match="Only one of weight_col and id_col can be provided"
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=3, id_col=4)


def test_distinct_col_error():
    with pytest.raises(
        ValueError,
        match=re.escape(f"ID column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=1)

    with pytest.raises(
        ValueError,
        match=re.escape(f"Weight column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=1)


def test_columns_out_of_range_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[-1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=-1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", id_col=-1)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[5])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", id_col=5)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=5)


def test_numerical_weights_error():
    with pytest.raises(
        ValueError, match=f"Weight {'a'} in row {0} must be able to be cast to float."
    ):
        load_csv(CSV_DIR / "non_numeric_weight.csv", rank_cols=[0, 1, 2], weight_col=3)

    with pytest.raises(ValueError, match=f"No weight provided in row {1}."):
        load_csv(CSV_DIR / "no_weight_provided.csv", rank_cols=[0, 1, 2], weight_col=3)


def test_header_error():
    with pytest.raises(ValueError, match="Header -1 must be non-negative."):
        load_csv(CSV_DIR / "valid_cvr.csv", header=-1)

    # yeah unexpected behavior, if the CSV has weight and ID but you don't tell it to incorporate,
    # then you get them as rankings

    # what do I do about tildes and skips?!?!


def test_load_csv():
    profile = load_csv(CSV_DIR / "cvr_no_weight_no_id.csv")

    true_profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Peter"}, {"Chris"})),
            Ballot(ranking=({"Chris"}, {"Moon"})),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"})),
            Ballot(ranking=({"Jeanne"}, {"David"}, {"Mala"})),
            Ballot(ranking=({"Moon"},)),
        )
    )

    print(profile.df.to_string())
    print()
    print(true_profile.df.to_string())

    assert profile == true_profile


# check that skips load correctly
# check that ID col and weights load
# check that header is used
# check for automatic rank col selection vs manual selection
# check for auto cand selection
# check that accessing the ballot list after loading skips works
