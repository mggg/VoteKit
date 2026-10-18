from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
from votekit.pref_profile.profile_matrix import ScoreMatrix


@pytest.mark.parametrize("weight", [1.5, Fraction(2, 3)])
def test_to_matrix_exports_independent_read_only_arrays_without_dataframe_conversion(
    weight, monkeypatch
):
    source = ScoreProfile(
        ballots=[ScoreBallot(scores={"B": 2, "A": 1}, weight=weight, voter_set={"voter"})],
        candidates=("B", "A", "unused"),
    ).to_df()
    source.index = pd.Index([37], name="Ballot Index")
    profile = ScoreProfile(df=source, candidates=("B", "A", "unused"))
    expected = profile.to_df()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Matrix export should copy matrix storage directly")

    with monkeypatch.context() as patch:
        patch.setattr(ScoreProfile, "to_df", unexpected_call)
        patch.setattr(ScoreProfile, "__init__", unexpected_call)
        exported = profile.to_matrix()

    assert type(exported) is ScoreMatrix
    assert exported is not profile._matrix
    assert exported == profile._matrix
    assert exported.score_columns == profile._matrix.score_columns
    assert exported.candidates == profile.candidates
    assert exported.voter_weight_data.weight_kind == profile._matrix.voter_weight_data.weight_kind
    pd.testing.assert_frame_equal(exported.to_df(), expected)

    source_data = profile._matrix.voter_weight_data
    exported_data = exported.voter_weight_data
    for copied, original in (
        (exported.scores, profile._matrix.scores),
        (exported_data.weights, source_data.weights),
        (exported_data.voter_sets, source_data.voter_sets),
        (exported_data.ballot_index, source_data.ballot_index),
    ):
        assert copied is not None and original is not None
        assert not np.shares_memory(copied, original)
        assert not copied.flags.writeable

    exported.scores.setflags(write=True)
    exported.scores[0, 0] = 99.0
    pd.testing.assert_frame_equal(profile.to_df(), expected)
