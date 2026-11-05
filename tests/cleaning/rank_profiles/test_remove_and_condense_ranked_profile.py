from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import (
    clean_rank_profile,
    condense_rank_profile,
    remove_and_condense_rank_profile,
    remove_cand_rank_profile,
)
from votekit.cleaning.rank_profiles_cleaning import (
    condense_ranking_row,
    remove_cand_from_ranking_row,
)
from votekit.pref_profile import CleanedRankProfile, RankProfile

profile_no_ties = RankProfile(
    ballots=[
        RankBallot(
            ranking=[
                {"A"},
            ],
            weight=1,
        ),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
    ]
)

profile_with_ties = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A", "B"}], weight=1),
        RankBallot(ranking=[{"A", "B", "C"}], weight=1 / 2),
        RankBallot(ranking=[{"A"}, {"C"}, {"B"}], weight=3),
    ]
)


def test_remove_and_condense():
    cleaned_profile = remove_and_condense_rank_profile("A", profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"B"}, {"C"}], weight=1 / 2),
        RankBallot(ranking=[{"C"}, {"B"}], weight=3),
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_then_condense_equivalence():
    cleaned_profile_1 = remove_and_condense_rank_profile("A", profile_no_ties)
    cleaned_profile_2 = condense_rank_profile(remove_cand_rank_profile("A", profile_no_ties))

    assert cleaned_profile_1 == cleaned_profile_2


def test_remove_mult_cands():
    cleaned_profile = remove_and_condense_rank_profile(["A", "B"], profile_no_ties)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_no_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_no_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


def test_remove_and_condense_with_ties():
    cleaned_profile = remove_and_condense_rank_profile(["A", "B"], profile_with_ties)
    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile_with_ties

    with pytest.warns(UserWarning, match="Grouping the ballots of a CleanedRankProfile"):
        grouped = cleaned_profile.group_ballots()
    assert set(grouped.ballots) == set(
        [
            RankBallot(ranking=[{"C"}], weight=7 / 2),
        ]
    )
    assert cleaned_profile != profile_with_ties
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {0}
    assert cleaned_profile.nonempty_altr_idxs == {1, 2}
    assert cleaned_profile.unaltr_idxs == set()


@pytest.mark.parametrize("retain_candidates", [False, True])
@pytest.mark.parametrize("remove_null", [False, True])
@pytest.mark.parametrize("remove_zero", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_remove_and_condense_constructs_once_with_equivalent_provenance(
    retain_candidates, remove_null, remove_zero, exact, monkeypatch
):
    weight = Fraction(1, 3) if exact else 0.5
    source = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}], weight=weight),
            RankBallot(ranking=[{"A", 1}, set(), {"B"}], weight=weight, voter_set={"voter"}),
            RankBallot(ranking=[{"B"}], weight=0),
            RankBallot(ranking=[{1}], weight=weight),
        ],
        candidates=("A", "B", 1, "unused"),
        max_ranking_length=4,
    ).to_df()
    source.index = pd.Index([8, 8, 20, 31], name="Ballot Index")
    parent = RankProfile(df=source, candidates=("A", "B", 1, "unused"), max_ranking_length=4)

    # Exercise a provenance chain as well as duplicate, non-default labels.
    profile = clean_rank_profile(parent, lambda row: row, remove_zero_weight_ballots=False)
    original_df = profile.to_df()

    intermediate = clean_rank_profile(
        profile,
        lambda row: condense_ranking_row(remove_cand_from_ranking_row(["A"], row)),
        remove_null,
        remove_zero,
        retain_original_candidate_list=True,
    )
    candidates = profile.candidates if retain_candidates else tuple(set(profile.candidates) - {"A"})
    expected = CleanedRankProfile(
        df=intermediate.to_df(),
        candidates=candidates,
        max_ranking_length=intermediate.max_ranking_length,
        parent_profile=intermediate.parent_profile,
        df_index_column=intermediate.df_index_column,
        no_wt_altr_idxs=intermediate.no_wt_altr_idxs,
        no_rank_altr_idxs=intermediate.no_rank_altr_idxs,
        nonempty_altr_idxs=intermediate.nonempty_altr_idxs,
        unaltr_idxs=intermediate.unaltr_idxs,
    )

    constructions = 0
    original_init = RankProfile.__init__

    def count_init(self, **kwargs):
        nonlocal constructions
        constructions += 1

        original_init(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(RankProfile, "__init__", count_init)

        result = remove_and_condense_rank_profile(
            "A", profile, remove_null, remove_zero, retain_candidates
        )

    assert constructions == 1
    assert type(result) is CleanedRankProfile
    assert result.parent_profile is profile
    assert isinstance(result.parent_profile, CleanedRankProfile)
    assert result.parent_profile.parent_profile is parent
    assert result.candidates == expected.candidates
    assert result.candidates_cast == expected.candidates_cast
    assert result.total_ballot_wt == expected.total_ballot_wt
    assert result.max_ranking_length == expected.max_ranking_length
    assert result.df_index_column == expected.df_index_column
    assert result.no_wt_altr_idxs == expected.no_wt_altr_idxs
    assert result.no_rank_altr_idxs == expected.no_rank_altr_idxs
    assert result.nonempty_altr_idxs == expected.nonempty_altr_idxs
    assert result.unaltr_idxs == expected.unaltr_idxs
    pd.testing.assert_frame_equal(result.to_df(), expected.to_df())
    pd.testing.assert_frame_equal(profile.to_df(), original_df)

    result.unaltr_idxs.clear()

    assert profile.unaltr_idxs == {8, 20, 31}


@pytest.mark.parametrize("retain_candidates", [False, True])
def test_remove_and_condense_all_candidates(retain_candidates):
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}])])

    result = remove_and_condense_rank_profile(
        "A", profile, retain_original_candidate_list=retain_candidates
    )

    assert result.num_ballots == 0
    assert result.candidates == (("A",) if retain_candidates else ())
    assert result.no_rank_altr_idxs == {0}
    assert result.parent_profile is profile
