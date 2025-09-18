from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
import pytest

filepath = "tests/pref_profile/data/score_profile"


def test_csv_bijection_scores():
    profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"Alex": 2, "Allen": 4, "D": 1}, voter_set={"Chris"}),
            ScoreBallot(
                scores={"Alex": 2, "Allen": 4, "D": 1}, voter_set={"Peter", "Moon"}
            ),
            ScoreBallot(
                scores={"Alex": 2, "Allen": 4, "C": 1},
            ),
            ScoreBallot(
                scores={"Alex": 2, "Allen": 4, "C": 1},
            ),
            ScoreBallot(
                scores={"Alex": 5, "Allen": 4, "C": 1},
            ),
        )
        * 5,
        candidates=["Alex", "Allen", "C", "D", "E"],
    )
    profile.to_csv(f"{filepath}/test_csv_pp_scores.csv", include_voter_set=True)
    read_profile = ScoreProfile.from_csv(f"{filepath}/test_csv_pp_scores.csv")
    assert profile == read_profile


def test_csv_filepath_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        ScoreProfile().to_csv("")


def test_csv_misformatted_header_rows_error():
    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 0 should be "
            "'VoteKit ScoreProfile'."
        ),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_0.csv")

    with pytest.raises(
        ValueError,
        match=("csv file is improperly formatted. Row 1 should be " "'Candidates'."),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 3 should be " "'Includes Voter Set'."
        ),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_3.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 5 should be "
            "'=,=,=,=,=,=,=,=,=,='."
        ),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_5.csv")


def test_csv_misformatted_header_values_error():
    with pytest.raises(
        ValueError, match="Row 2 should contain tuples mapping candidates"
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_2.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be " "'True' or 'False'."
        ),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_4_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be " "'True' or 'False'."
        ),
    ):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_4_2.csv")


def test_csv_misformatted_ballot_header_values_error():
    with pytest.raises(
        ValueError, match="Row 6 should include all " "candidates before the first &."
    ):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_header_candidates.csv"
        )

    with pytest.raises(ValueError, match="Row 6 should include 'Weight' column."):
        ScoreProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_weight_missing.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        ScoreProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_voter_set_false.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        ScoreProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_voter_set_true.csv")
        )


def test_csv_misformatted_ballot_score_error():
    with pytest.raises(ValueError, match=("Ballot in row 7 is missing some scores")):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_score_missing.csv"
        )

    with pytest.raises(
        ValueError, match=(f"Ballot in row 7 has non-float score value: {'a'}. ")
    ):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_score_non_float.csv"
        )


def test_csv_misformatted_ballot_weight_error():
    with pytest.raises(
        ValueError,
        match=("Ballot in row 14 has a weight" " entry that is too long or short. "),
    ):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_long.csv"
        )

    with pytest.raises(
        match="Ballot in row 14 has a "
        f"weight entry that can't be converted to float {'a'}. "
    ):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_non_float.csv"
        )


def test_csv_misformatted_voter_set_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 16 has a voter set but it should not.")
    ):
        ScoreProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_no_voter_set.csv"
        )


def test_csv_voter_set_whitespace():
    assert ScoreProfile.from_csv(
        f"{filepath}/test_csv_pp_voter_set_whitespace.csv"
    ) == ScoreProfile.from_csv(f"{filepath}/test_csv_pp_scores.csv")


def test_csv_seven_errors():
    with pytest.raises(ValueError, match="There are errors on 7 lines"):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_7_errors.csv")


def test_csv_11_errors():
    with pytest.raises(ValueError, match="There are errors on at least ten lines"):
        ScoreProfile.from_csv(f"{filepath}/test_csv_pp_misformat_11_errors.csv")
