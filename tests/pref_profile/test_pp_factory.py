from typing import assert_type

import pandas as pd
import pytest

from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.pref_profile import (
    CleanedRankProfile,
    CleanedScoreProfile,
    PreferenceProfile,
    ProfileError,
    RankProfile,
    ScoreProfile,
)
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix


def test_pp_factory_to_rank_profile():
    pp = PreferenceProfile(ballots=[RankBallot(ranking=[{"Chris"}])])

    assert isinstance(pp, RankProfile)


def test_pp_factory_to_score_profile():
    pp = PreferenceProfile(ballots=[ScoreBallot(scores={"Chris": 1})])

    assert isinstance(pp, ScoreProfile)


def test_pp_factory_errors():
    with pytest.raises(ProfileError, match="Cannot pass a dataframe and a ballot list"):
        PreferenceProfile(df=pd.DataFrame({"Col": [1]}), ballots=[RankBallot()])

    with pytest.raises(
        ProfileError,
        match="Profile cannot contain RankBallots and ScoreBallots. "
        "There are 1 ScoreBallots and 1 RankBallots.",
    ):
        PreferenceProfile(ballots=[RankBallot(), ScoreBallot()])


class DerivedRankProfile(RankProfile):
    pass


class DerivedScoreProfile(ScoreProfile):
    pass


@pytest.mark.parametrize(
    "profile_type, ballot",
    [
        (RankProfile, RankBallot(ranking=[{"Chris"}], weight=2)),
        (ScoreProfile, ScoreBallot(scores={"Chris": 1}, weight=2)),
        (DerivedRankProfile, RankBallot(ranking=[{"Chris"}], weight=2)),
        (DerivedScoreProfile, ScoreBallot(scores={"Chris": 1}, weight=2)),
    ],
)
@pytest.mark.parametrize("from_dataframe", [False, True])
def test_explicit_profile_preserves_type_and_initializes(profile_type, ballot, from_dataframe):
    kwargs = (
        {"df": PreferenceProfile(ballots=[ballot]).to_df()}
        if from_dataframe
        else {"ballots": [ballot]}
    )
    profile = profile_type(**kwargs, candidates=("Chris",), max_ranking_length=1)

    assert type(profile) is profile_type
    assert profile.num_ballots == 1
    assert profile.total_ballot_wt == 2
    assert profile.candidates == ("Chris",)
    assert profile.ballots == (ballot,)


@pytest.mark.parametrize(
    "profile_type, valid_ballot, invalid_ballot",
    [
        (RankProfile, RankBallot(), ScoreBallot()),
        (ScoreProfile, ScoreBallot(), RankBallot()),
        (DerivedRankProfile, RankBallot(), ScoreBallot()),
        (DerivedScoreProfile, ScoreBallot(), RankBallot()),
        (CleanedRankProfile, RankBallot(), ScoreBallot()),
        (CleanedScoreProfile, ScoreBallot(), RankBallot()),
    ],
)
@pytest.mark.parametrize("valid_prefix", [False, True])
def test_explicit_profile_rejects_incompatible_ballots(
    profile_type, valid_ballot, invalid_ballot, valid_prefix
):
    ballots = [valid_ballot, invalid_ballot] if valid_prefix else [invalid_ballot]

    with pytest.raises(ProfileError, match=f"ballot at index {int(valid_prefix)} is"):
        profile_type(ballots=ballots)


@pytest.mark.parametrize(
    "profile_type",
    [
        PreferenceProfile,
        RankProfile,
        ScoreProfile,
        DerivedRankProfile,
        DerivedScoreProfile,
        CleanedRankProfile,
        CleanedScoreProfile,
    ],
)
def test_all_profile_constructors_reject_dataframe_and_ballots(profile_type):
    with pytest.raises(ProfileError, match="Cannot pass a dataframe and a ballot list"):
        profile_type(df=pd.DataFrame({"Col": [1]}), ballots=[RankBallot()])


@pytest.mark.parametrize(
    "profile_type",
    [PreferenceProfile, RankProfile, ScoreProfile, DerivedRankProfile, DerivedScoreProfile],
)
def test_empty_profile_preserves_requested_type(profile_type):
    profile = profile_type()

    assert type(profile) is profile_type
    assert profile.num_ballots == 0
    assert profile.total_ballot_wt == 0


@pytest.mark.parametrize(
    "profile_type, ballot",
    [
        (RankProfile, RankBallot(ranking=[{"Chris"}])),
        (ScoreProfile, ScoreBallot(scores={"Chris": 1})),
    ],
)
def test_factory_infers_profile_type_from_dataframe(profile_type, ballot):
    profile = PreferenceProfile(
        df=profile_type(ballots=[ballot]).to_df(), candidates=("Chris",), max_ranking_length=1
    )

    assert type(profile) is profile_type
    assert profile.ballots == (ballot,)


def test_explicit_score_profile_does_not_infer_type_from_column_prefix():
    ballot = ScoreBallot(scores={"Ranking_label": 1})
    profile = ScoreProfile(df=ScoreProfile(ballots=[ballot]).to_df(), candidates=("Ranking_label",))

    assert type(profile) is ScoreProfile
    assert profile.ballots == (ballot,)


@pytest.mark.parametrize(
    "cleaned_type, profile_type, ballot",
    [
        (CleanedRankProfile, RankProfile, RankBallot(ranking=[{"Chris"}])),
        (CleanedScoreProfile, ScoreProfile, ScoreBallot(scores={"Chris": 1})),
    ],
)
def test_cleaned_profile_accepts_positional_parent(cleaned_type, profile_type, ballot):
    parent = profile_type(ballots=[ballot])
    profile = cleaned_type(parent, df_index_column=[0], unaltr_idxs={0}, ballots=[ballot])

    assert type(profile) is cleaned_type
    assert profile.parent_profile is parent
    assert profile.ballots == (ballot,)


def test_explicit_profile_constructor_return_types():
    rank_profile = assert_type(RankProfile(), RankProfile)
    score_profile = assert_type(ScoreProfile(), ScoreProfile)

    assert_type(RankProfile(df=rank_profile.to_df()), RankProfile)
    assert_type(ScoreProfile(df=score_profile.to_df()), ScoreProfile)
    assert_type(DerivedRankProfile(), DerivedRankProfile)
    assert_type(DerivedScoreProfile(), DerivedScoreProfile)
    assert_type(DerivedRankProfile(ballots=[RankBallot()]), DerivedRankProfile)
    assert_type(DerivedScoreProfile(ballots=[ScoreBallot()]), DerivedScoreProfile)


def test_generic_profile_matrix_and_ballot_types():
    rank_profile: PreferenceProfile[RankMatrix, RankBallot] = RankProfile(ballots=[RankBallot()])
    score_profile: PreferenceProfile[ScoreMatrix, ScoreBallot] = ScoreProfile(
        ballots=[ScoreBallot()]
    )

    assert_type(rank_profile._matrix, RankMatrix)
    assert_type(score_profile._matrix, ScoreMatrix)
    assert_type(rank_profile.ballots, tuple[RankBallot, ...])
    assert_type(score_profile.ballots, tuple[ScoreBallot, ...])
    assert_type(rank_profile.get_ballot(0), RankBallot)
    assert_type(score_profile.get_ballot(0), ScoreBallot)
    assert isinstance(rank_profile._matrix, RankMatrix)
    assert isinstance(score_profile._matrix, ScoreMatrix)

    with pytest.raises(AttributeError, match="Cannot modify frozen instance"):
        rank_profile._matrix = rank_profile._matrix


def test_factory_return_types_include_empty_profiles():
    empty = assert_type(PreferenceProfile(), PreferenceProfile[None, Ballot])
    assert_type(empty._matrix, None)
    assert empty._matrix is None
    assert empty.num_ballots == empty.total_ballot_wt == 0

    rank_ballots: list[RankBallot] = []
    score_ballots: list[ScoreBallot] = []
    empty_rank = assert_type(
        PreferenceProfile(ballots=rank_ballots), RankProfile | PreferenceProfile[None, Ballot]
    )
    empty_score = assert_type(
        PreferenceProfile(ballots=score_ballots), ScoreProfile | PreferenceProfile[None, Ballot]
    )
    assert empty_rank._matrix is empty_score._matrix is None

    from_df = assert_type(
        PreferenceProfile(
            df=ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})]).to_df(), candidates=("A",)
        ),
        RankProfile | ScoreProfile | PreferenceProfile[None, Ballot],
    )
    assert isinstance(from_df, ScoreProfile)


def test_zero_ranking_length_infers_length_for_factory_and_explicit_profile():
    ballots = [RankBallot(ranking=["A", "B"])]
    explicit = RankProfile(ballots=ballots, max_ranking_length=0)
    inferred = PreferenceProfile(ballots=ballots, max_ranking_length=0)

    assert explicit.max_ranking_length == inferred.max_ranking_length == 2
    assert explicit.ballots == inferred.ballots == tuple(ballots)


@pytest.mark.parametrize(
    "profile_type, ballot",
    [(RankProfile, RankBallot(ranking=[{"A"}])), (ScoreProfile, ScoreBallot(scores={"A": 1}))],
)
def test_ballot_construction_normalizes_declared_candidates_before_membership(profile_type, ballot):
    profile = profile_type(ballots=[ballot], candidates=[" A ", " unused "])

    assert profile.candidates == profile._matrix.candidates == ("A", "unused")
    assert profile.candidates_cast == ("A",)
    assert profile.ballots == (ballot,)


@pytest.mark.parametrize("profile_type", [RankProfile, ScoreProfile])
@pytest.mark.parametrize("candidate", [True, 1.5, ["A"]])
def test_candidate_types_are_validated_before_duplicate_detection(profile_type, candidate):
    with pytest.raises(TypeError, match="candidate"):
        profile_type(candidates=[candidate])


@pytest.mark.parametrize("profile_type", [PreferenceProfile, RankProfile, ScoreProfile])
def test_profile_rejects_bare_string_candidates(profile_type):
    with pytest.raises(TypeError, match="Candidates cannot be a string"):
        profile_type(candidates="Alice")


@pytest.mark.parametrize("profile_type", [PreferenceProfile, RankProfile, ScoreProfile])
@pytest.mark.parametrize("name", ["Ranking_0", "Ranking_1", "Ranking_100"])
@pytest.mark.parametrize("other_candidates", [[], ["A", "B"]])
def test_reserved_ranking_names_do_not_depend_on_candidate_count(
    profile_type, name, other_candidates
):
    with pytest.raises(ProfileError, match="name reserved for ranking columns"):
        profile_type(candidates=[name, *other_candidates])


@pytest.mark.parametrize(
    "cleaned_type, profile_type, ballot",
    [
        (CleanedRankProfile, RankProfile, RankBallot(ranking=[{"A"}])),
        (CleanedScoreProfile, ScoreProfile, ScoreBallot(scores={"A": 1})),
    ],
)
def test_cleaned_profile_retains_parent_indices_in_matrix(cleaned_type, profile_type, ballot):
    parent = profile_type(ballots=[type(ballot)(weight=0), ballot])
    cleaned = cleaned_type(
        parent,
        df_index_column=[1],
        unaltr_idxs={1},
        no_wt_altr_idxs={0},
        ballots=[ballot],
    )

    assert cleaned.to_df().index.tolist() == [1]
    assert cleaned._matrix.to_df().index.tolist() == [1]
    assert cleaned.copy().to_df().index.tolist() == [1]
