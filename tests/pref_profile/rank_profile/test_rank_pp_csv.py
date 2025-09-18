from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
import pytest

filepath = "tests/pref_profile/data/rank_profile"


def test_csv_bijection_rankings():
    profile_rankings = RankProfile(
        ballots=(
            RankBallot(
                ranking=({"Aleine", "Alex"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=1.5,
            ),
            RankBallot(
                ranking=({"Aleine", "Alex"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=0.5,
            ),
            RankBallot(
                ranking=(
                    {"Aleine"},
                    {"Alex"},
                ),
            ),
            RankBallot(
                ranking=(
                    {"Aleine"},
                    {"Alex"},
                ),
            ),
            RankBallot(
                ranking=(
                    {"Aleine"},
                    {"Alex"},
                ),
            ),
        )
        * 5,
        max_ranking_length=3,
        candidates=["Aleine", "Alex", "C", "D", "E"],
    )

    profile_rankings.to_csv(
        f"{filepath}/test_csv_pp_rankings.csv", include_voter_set=True
    )
    read_profile = RankProfile.from_csv(f"{filepath}/test_csv_pp_rankings.csv")
    assert profile_rankings == read_profile


def test_csv_filepath_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        RankProfile().to_csv("")


def test_csv_misformatted_header_rows_error():
    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 0 should be "
            "'VoteKit RankProfile'."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_0.csv")

    with pytest.raises(
        ValueError,
        match=("csv file is improperly formatted. Row 1 should be " "'Candidates'."),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 3 should be " "'Max Ranking Length'."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_3.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 5 should be " "'Includes Voter Set'."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_5.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 7 should be "
            "'=,=,=,=,=,=,=,=,=,='."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_7.csv")


def test_csv_misformatted_header_values_error():
    with pytest.raises(
        ValueError, match="Row 2 should contain tuples mapping candidates"
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_2.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            "a single non-negative integer denoting the max ranking length, not -1"
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_4_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            r"a single non-negative integer denoting the max ranking length, not \['3', '3'\]."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_4_2.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            "a single non-negative integer denoting the max ranking length, not a"
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_4_3.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 6 should be " "'True' or 'False'."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_6_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 6 should be " "'True' or 'False'."
        ),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_value_6_2.csv")


def test_csv_misformatted_ballot_header_values_error():
    with pytest.raises(ValueError, match="Row 8 should include 'Ranking_i'"):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_ballot_header_1.csv")

    with pytest.raises(ValueError, match="Row 8 should not include 'Ranking_'"):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_ballot_header_2.csv")

    with pytest.raises(ValueError, match="Row 8 should include 'Weight' column."):
        RankProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "weight_missing.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        RankProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "voter_set_false.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        RankProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "voter_set_true.csv")
        )


def test_csv_misformatted_ballot_ranking_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 11 has improperly formatted ranking ")
    ):
        RankProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_ranking_brace.csv"
        )

    with pytest.raises(
        ValueError, match=("Ballot in row 11 has undefined candidate prefix")
    ):
        RankProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_ranking_prefix.csv"
        )


def test_csv_misformatted_ballot_weight_error():
    with pytest.raises(
        ValueError,
        match=("Ballot in row 16 has a weight" " entry that is too long or short. "),
    ):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_ballot_weight_long.csv")

    with pytest.raises(
        match="Ballot in row 16 has a "
        f"weight entry that can't be converted to float {'a'}. "
    ):
        RankProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_non_float.csv"
        )


def test_csv_misformatted_voter_set_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 16 has a voter set but it should not.")
    ):
        RankProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_no_voter_set.csv"
        )


def test_csv_voter_set_whitespace():
    assert RankProfile.from_csv(
        f"{filepath}/test_csv_pp_voter_set_whitespace.csv"
    ) == RankProfile.from_csv(f"{filepath}/test_csv_pp_rankings.csv")


def test_csv_nine_errors():
    with pytest.raises(ValueError, match="There are errors on 7 lines"):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_7_errors.csv")


def test_csv_11_errors():
    with pytest.raises(ValueError, match="There are errors on at least ten lines"):
        RankProfile.from_csv(f"{filepath}/test_csv_pp_misformat_11_errors.csv")
