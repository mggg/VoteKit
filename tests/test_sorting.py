import pytest

from votekit.sorting import sort_candidates_lexicographically


def test_sort_mixed_cands_lexicographically():
    cands = ["1", "01", 1, 2, "2.0", "3"]
    expected_sorted_cands = [1, "01", "1", 2, "3", "2.0"]
    assert expected_sorted_cands == sort_candidates_lexicographically(cands)


def test_sort_non_valid_type_cand_raises_error():
    cands = ["1", 1, 1.0]
    with pytest.raises(TypeError, match="Candidates can only be strings or integers."):
        sort_candidates_lexicographically(cands)  # type: ignore[arg-type]
