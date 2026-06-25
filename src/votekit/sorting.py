from typing import Iterable

from votekit.types import Candidate


def sort_candidates_lexicographically(candidates: Iterable[Candidate]) -> list[Candidate]:
    """
    Sort candidates in lexicographical/alphabetical order.

    If candidates are of mixed type (i.e. strings and integers), integer candidates will be ordered
    before string candidates with the exception of string candidates that can be cast to an integer.
    String candidates that can be cast to a corresponding integer candidate will follow directly
    after that integer in lexicographical order. String candidates that can be cast to an integer
    but have no corresponding integer cadidate will follow all integer candidates, ordered by their
    integer value.

    Example:
        If we have candidates = ["1", "01", 1, "1.0", 2, "20", "3"],
        the sorted candidates will be [1, "01", "1", 2, "3", "20", "1.0"]
        "01" and "1" string candidates are equivalent to the 1 integer candidate.
        "1.0" cannot be converted into an integer and is treated as a non-integer string candidate.
        "3" and "20" are string candidates that can be cast to integers but do not have a
        corresponding integer candidate.

    Args:
        candidates (Sequence[Candidate]): list of candidates to sort

    Returns:
        tuple[Candidate,...]: sorted candidates

    """
    candidates = list(candidates)
    try:
        return sorted(candidates)
    except TypeError:
        int_candidates = [cand for cand in candidates if isinstance(cand, int)]

        def sort_mixed_cands(cand):
            if isinstance(cand, int):
                return (0, cand, "")
            elif isinstance(cand, str) and cand.isdigit():
                str_as_int_cand = int(cand)
                if str_as_int_cand in int_candidates:
                    return (0, str_as_int_cand, cand)
                return (1, 0, str_as_int_cand)
            return (2, 0, cand)

        return sorted(candidates, key=sort_mixed_cands)
