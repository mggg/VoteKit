import random

import numpy as np

from votekit.ballot import RankBallot
from votekit.elections import ElectionState
from votekit.elections.election_types.ranking.stv.meek import (
    MeekSTV,
    _permutation_matrix_constructor,
    _vectorized_perm_updater,
)
from votekit.pref_profile import RankProfile
from votekit.utils import index_to_lexicographic_ballot

# explanations for these tests available here:
# https://epfheitzmann.com/projects/meek/

basic_profile_1 = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A", "B"])), weight=101),
            RankBallot(ranking=tuple(map(frozenset, ["B"])), weight=80),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["D", "A", "C"])), weight=29),
        ]
    ),
    max_ranking_length=3,
)

basic_profile_2 = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A", "C"])), weight=120),
            RankBallot(ranking=tuple(map(frozenset, ["B"])), weight=100),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=79),
            RankBallot(ranking=tuple(map(frozenset, ["D"])), weight=60),
            RankBallot(ranking=tuple(map(frozenset, ["E"])), weight=50),
            RankBallot(ranking=tuple(map(frozenset, ["F"])), weight=41),
        ]
    ),
    max_ranking_length=2,
)

advanced_profile_1 = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A", "B"])), weight=60),
            RankBallot(ranking=tuple(map(frozenset, ["A"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["B"])), weight=63),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=87),
        ]
    ),
    max_ranking_length=2,
)

advanced_profile_2 = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A", "B", "C"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["A", "C"])), weight=30),
            RankBallot(ranking=tuple(map(frozenset, ["B", "A", "C"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["B", "C"])), weight=30),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=61),
            RankBallot(ranking=tuple(map(frozenset, ["D"])), weight=99),
        ]
    ),
    max_ranking_length=3,
)

basic_profile_1_states = [
    ElectionState(
        round_number=0,
        remaining=tuple(map(frozenset, ["A", "C", "B", "D"])),
        scores={
            "A": 101,
            "B": 80,
            "C": 90,
            "D": 29,
        },
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(map(frozenset, ["C", "B", "D"])),
        elected=(frozenset({"A"}),),
        scores={
            "A": 100.000001,
            "B": 80.999999,
            "C": 90,
            "D": 29,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=tuple(map(frozenset, ["B", "C"])),
        eliminated=(frozenset({"D"}),),
        scores={
            "A": 100.000001,
            "B": 103.307692,
            "C": 96.692307,
        },
    ),
    ElectionState(
        round_number=3,
        remaining=tuple(map(frozenset, ["C"])),
        elected=(frozenset({"B"}),),
        scores={
            "A": 97.295601,
            "B": 97.295601,
            "C": 97.295596,
        },
    ),
]

basic_profile_2_states = [
    ElectionState(
        round_number=0,
        remaining=tuple(map(frozenset, ["A", "B", "C", "D", "E", "F"])),
        scores={"A": 120.0, "C": 79.0, "B": 100.0, "D": 60.0, "E": 50.0, "F": 41.0},
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(map(frozenset, ["A", "B", "C", "D", "E"])),
        eliminated=(frozenset({"F"}),),
        scores={"A": 120.0, "C": 79.0, "B": 100.0, "D": 60.0, "E": 50.0},
    ),
    ElectionState(
        round_number=2,
        remaining=tuple(map(frozenset, ["A", "B", "C", "D"])),
        eliminated=(frozenset({"E"}),),
        scores={"A": 120.0, "C": 79.0, "B": 100.0, "D": 60.0},
    ),
    ElectionState(
        round_number=3,
        remaining=tuple(map(frozenset, ["B", "C", "D"])),
        elected=(frozenset({"A"}),),
        scores={"A": 119.666668, "C": 79.333332, "B": 100.0, "D": 60.0},
    ),
    ElectionState(
        round_number=4,
        remaining=tuple(map(frozenset, ["B", "C"])),
        eliminated=(frozenset({"D"}),),
        scores={"A": 99.666668, "C": 99.333332, "B": 100.0},
    ),
    ElectionState(
        round_number=5,
        remaining=tuple(map(frozenset, ["C"])),
        elected=(frozenset({"B"}),),
        scores={"A": 99.500002, "C": 99.499998, "B": 99.500002},
    ),
]

advanced_profile_1_states = [
    ElectionState(
        round_number=0,
        remaining=tuple(map(frozenset, ["A", "C", "B"])),
        scores={"A": 150.0, "B": 63.0, "C": 87.0},
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(map(frozenset, ["B", "C"])),
        elected=(frozenset({"A"}),),
        scores={"A": 87.500002, "B": 87.999999, "C": 87.0},
    ),
    ElectionState(
        round_number=2,
        remaining=tuple(map(frozenset, ["C"])),
        elected=(frozenset({"B"}),),
        scores={"A": 87.000005, "B": 87.000005, "C": 87.0},
    ),
]

advanced_profile_2_states = [
    ElectionState(
        round_number=0,
        remaining=(frozenset({"A", "B"}), frozenset({"D"}), frozenset({"C"})),
        scores={"A": 120.0, "B": 120.0, "C": 61.0, "D": 99.0},
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(map(frozenset, ["C", "D"])),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"A": 100.000001, "B": 100.000001, "C": 100.999997, "D": 99.0},
    ),
    ElectionState(
        round_number=2,
        remaining=tuple(map(frozenset, ["D"])),
        elected=(frozenset({"C"}),),
        scores={"A": 99.000006, "B": 99.000006, "C": 99.000007, "D": 99.0},
    ),
]


def looser_equality_for_election_states(state1, state2, precision):
    if state1.remaining != state2.remaining:
        return False
    if state1.eliminated != state2.eliminated:
        return False
    if state1.elected != state2.elected:
        return False
    for cand, score in state1.scores.items():
        if cand not in state2.scores:
            return False
        if abs(score - state2.scores[cand]) > precision:
            return False
    return True


def test_basic_profile_1():
    elec = MeekSTV(basic_profile_1, n_seats=2)
    assert all(
        looser_equality_for_election_states(
            elec.election_states[i], basic_profile_1_states[i], precision=1e-6
        )
        for i in [0, 1, 2, 3]
    )


def test_basic_profile_2():
    elec = MeekSTV(basic_profile_2, n_seats=2)
    assert all(
        looser_equality_for_election_states(
            elec.election_states[i], basic_profile_2_states[i], precision=1e-6
        )
        for i in [0, 1, 2, 3, 4, 5]
    )


def test_advanced_profile_1():
    elec = MeekSTV(advanced_profile_1, n_seats=2)
    assert all(
        looser_equality_for_election_states(
            elec.election_states[i], advanced_profile_1_states[i], precision=1e-6
        )
        for i in [0, 1, 2]
    )


def test_advanced_profile_2():
    elec = MeekSTV(advanced_profile_2, n_seats=3)
    assert all(
        looser_equality_for_election_states(
            elec.election_states[i], advanced_profile_2_states[i], precision=1e-6
        )
        for i in [0, 1, 2]
    )


def test_vectorized_perm_updater():
    """
    What this test does:
    - generate a full permutation matrix (and pad it with a negative column)
    - pick some row indices to sample from it
    - iteratively apply vectorized_perm_updater using the winner indices from the matrix
    - for each row, this means that we will have called the updater once per non-negative entry
        in that row
    - if the updater is working correctly, the final winner_comb_vec should recover the sampled
        row indices
    E.g. if the sampled row is [3,0,1,-1,-1]: we call the updater first with 3, then 0, then 1.
    We check that the final comb_idx after these three updates is the same as the index of the row
        in the permuation matrix.
    """
    seed = random.randint(0, 10_000_000)
    rng = np.random.default_rng(seed)
    for m in range(1, 9):
        for L in range(1, m + 1):
            dense_matrix = _permutation_matrix_constructor(m, L)
            P = dense_matrix.shape[0]
            sample = rng.choice(P, size=min(100, P), replace=False)
            sampled_matrix = dense_matrix[
                sample, :
            ]  # also append a column of -1s to the end of the sampled matrix
            sampled_matrix = np.hstack(
                [
                    sampled_matrix,
                    -1 * np.ones((sampled_matrix.shape[0], 1), dtype=sampled_matrix.dtype),
                ]
            )
            pos_vec = np.zeros_like(sample)
            fpv_vec = sampled_matrix[:, 0]
            needs_update = fpv_vec > -1
            winner_comb_vec = np.zeros_like(sample)
            winner_bitstring_vec = np.zeros_like(sample)
            while np.any(needs_update):
                winner_comb_vec[needs_update], winner_bitstring_vec[needs_update] = (
                    _vectorized_perm_updater(
                        winner_comb_vec[needs_update],
                        m,
                        L,
                        winner_bitstring_vec[needs_update],
                        fpv_vec[needs_update],
                    )
                )
                pos_vec[needs_update] += 1
                fpv_vec[needs_update] = sampled_matrix[
                    np.arange(len(sample))[needs_update], pos_vec[needs_update]
                ]
                needs_update = fpv_vec > -1
            assert np.array_equal(sample, winner_comb_vec), (
                f"vectorized perm updater mismatch (seed={seed}, m={m}, L={L})"
            )


def test_permutation_matrix_constructor():
    """
    What this test does:
    - generate the full permutation matrix for given m and L
    - sample some (usually 100) row indices from it
    - give those row indices to the index_to_lexicographic_ballot utils function,
        and see if the permutation it generates is the same as the corresponding row of the
        permutation matrix
    N.b. that the permutation matrix has a leading row of -1s representing the empty permutation,
        whereas the utils function does not support empty permutations --
        hence there is some funny shifting to make sure these two permutation constructors are
        speaking the same language.
    """
    seed = random.randint(0, 10_000_000)
    rng = np.random.default_rng(seed)
    for m in range(1, 9):
        for L in range(1, m + 1):
            A = _permutation_matrix_constructor(m, L)
            P = A.shape[0] - 1
            sample_indices = rng.choice(P, size=min(100, P), replace=False) + 1
            for idx in sample_indices:
                try:
                    perm_from_utils = index_to_lexicographic_ballot(idx - 1, m, L)
                except Exception as e:
                    print(f"Error for idx={idx}, m={m}, L={L}, seed={seed}: {e}")
                    raise
                perm_from_utils = np.pad(
                    perm_from_utils,
                    (0, L - len(perm_from_utils)),
                    mode="constant",
                    constant_values=-1,
                )
                assert np.array_equal(perm_from_utils, A[idx, :]), (
                    f"permutation mismatch (seed={seed}, idx={idx}, m={m}, L={L})"
                )
