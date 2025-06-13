from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pytest

filepath = "tests/data/csv"


def test_csv_bijection_rankings():
    profile_rankings = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"Aleine", "Alex"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            Ballot(
                ranking=({"Aleine", "Alex"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            Ballot(
                ranking=(
                    {"Aleine"},
                    {"Alex"},
                ),
            ),
            Ballot(
                ranking=(
                    {"Aleine"},
                    {"Alex"},
                ),
            ),
            Ballot(
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
    read_profile = PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_rankings.csv")
    assert profile_rankings == read_profile


def test_csv_bijection_scores():
    profile_scores = PreferenceProfile(
        ballots=(
            Ballot(
                scores={"Alex": 2, "Allen": 4, "D": 1},
            ),
            Ballot(scores={"Alex": 2, "Allen": 4, "D": 1}),
            Ballot(
                scores={"Alex": 2, "Allen": 4, "C": 1},
            ),
            Ballot(
                scores={"Alex": 2, "Allen": 4, "C": 1},
            ),
            Ballot(
                scores={"Alex": 5, "Allen": 4, "C": 1},
            ),
        )
        * 5,
        candidates=["Alex", "Allen", "C", "D", "E"],
    )

    profile_scores.to_csv(f"{filepath}/test_csv_pp_scores.csv", include_voter_set=True)
    read_profile = PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_scores.csv")
    assert profile_scores == read_profile


def test_csv_bijection_mixed():
    profile_mixed = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 5, "B": 4, "C": 1},
            ),
        )
        * 2,
        max_ranking_length=3,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_mixed.to_csv(f"{filepath}/test_csv_pp_mixed.csv", include_voter_set=True)
    read_profile = PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_mixed.csv")
    assert profile_mixed == read_profile


def test_csv_bijection_mixed_no_voter_set():
    profile_mixed = PreferenceProfile(
        ballots=(
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            Ballot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(scores={"A": 2, "B": 4, "D": 1}),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 2, "B": 4, "C": 1},
            ),
            Ballot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
                scores={"A": 5, "B": 4, "C": 1},
            ),
        )
        * 2,
        max_ranking_length=3,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_mixed.to_csv(
        f"{filepath}/test_csv_pp_mixed_no_voter_set.csv", include_voter_set=False
    )
    read_profile = PreferenceProfile.from_csv(
        f"{filepath}/test_csv_pp_mixed_no_voter_set.csv"
    )
    assert profile_mixed.df.drop(columns=["Voter Set"]).equals(
        read_profile.df.drop(columns=["Voter Set"])
    )
    assert (read_profile.df["Voter Set"] == set()).all()


def test_csv_filepath_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        PreferenceProfile().to_csv("")


def test_csv_misformatted_header_rows_error():
    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 0 should be "
            "'VoteKit PreferenceProfile'."
        ),
    ):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_0.csv")

    with pytest.raises(
        ValueError,
        match=("csv file is improperly formatted. Row 1 should be " "'Candidates'."),
    ):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_1.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 3 should be " "'Max Ranking Length'."
        ),
    ):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_3.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 5 should be " "'Includes Voter Set'."
        ),
    ):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_5.csv")

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 7 should be "
            "'=,=,=,=,=,=,=,=,=,='."
        ),
    ):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_header_7.csv")


def test_csv_misformatted_header_values_error():
    with pytest.raises(
        ValueError, match="Row 2 should contain tuples mapping candidates"
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_2.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            "a single non-negative integer."
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_4_1.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            "a single non-negative integer."
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_4_2.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 4 should be "
            "a single non-negative integer."
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_4_3.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 6 should be " "'True' or 'False'."
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_6_1.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "csv file is improperly formatted. Row 6 should be " "'True' or 'False'."
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_header_value_6_2.csv"
        )


def test_csv_misformatted_ballot_header_values_error():
    with pytest.raises(ValueError, match="Row 8 should include 'Ranking_i'"):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_header_ranking_1.csv"
        )

    with pytest.raises(ValueError, match="Row 8 should not include 'Ranking_'"):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_header_ranking_2.csv"
        )

    with pytest.raises(
        ValueError, match="Row 8 should include all candidates before the first &"
    ):
        PreferenceProfile.from_csv(
            (
                f"{filepath}/test_csv_pp_misformat_ballot_header_"
                "candidates_missing.csv"
            )
        )

    with pytest.raises(ValueError, match="Row 8 should include 'Weight' column."):
        PreferenceProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "weight_missing.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        PreferenceProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "voter_set_false.csv")
        )

    with pytest.raises(
        ValueError, match="Includes Voter Set is not set to the correct value"
    ):
        PreferenceProfile.from_csv(
            (f"{filepath}/test_csv_pp_misformat_ballot_header_" "voter_set_true.csv")
        )


def test_csv_misformatted_ballot_score_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 16 has scores but it should not. ")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_no_scores.csv"
        )

    with pytest.raises(ValueError, match=("Ballot in row 16 is missing some scores. ")):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_score_len.csv"
        )

    with pytest.raises(
        ValueError, match=("Ballot in row 16 has non-float score value.")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_score_non_float.csv"
        )


def test_csv_misformatted_ballot_ranking_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 16 has rankings but it should not. ")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_no_rankings.csv"
        )

    with pytest.raises(
        ValueError, match=("Ballot in row 11 has improperly formatted ranking ")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_ranking_brace.csv"
        )

    with pytest.raises(
        ValueError, match=("Ballot in row 11 has undefined candidate prefix")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_ranking_prefix.csv"
        )


def test_csv_misformatted_ballot_weight_error():
    with pytest.raises(
        ValueError,
        match=("Ballot in row 16 has a weight" " entry that is too long or short. "),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_long.csv"
        )

    with pytest.raises(
        ValueError,
        match=("Ballot in row 16 has a weight" " entry that is not a fraction. "),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_not_frac.csv"
        )

    with pytest.raises(
        ValueError,
        match=(
            "Ballot in row 16 has a weight"
            " entry with non-integer numerator or denominator. "
        ),
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_weight_not_int_num.csv"
        )


def test_csv_misformatted_voter_set_error():
    with pytest.raises(
        ValueError, match=("Ballot in row 16 has a voter set but it should not.")
    ):
        PreferenceProfile.from_csv(
            f"{filepath}/test_csv_pp_misformat_ballot_no_voter_set.csv"
        )


def test_csv_voter_set_whitespace():
    assert PreferenceProfile.from_csv(
        f"{filepath}/test_csv_pp_voter_set_whitespace.csv"
    ) == PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_mixed.csv")


def test_csv_nine_errors():
    with pytest.raises(ValueError, match="There are errors on 9 lines"):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_9_errors.csv")


def test_csv_11_errors():
    with pytest.raises(ValueError, match="There are errors on at least ten lines"):
        PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_misformat_11_errors.csv")
