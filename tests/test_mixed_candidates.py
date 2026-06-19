from functools import partial

import pytest

from votekit import utils
from votekit.ballot_generator import ic_profile_generator
from votekit.elections import STV, Borda, Plurality

MIXED_CANDS = ["A", "B", "1", 1, 2, 3]
N_SEATS = 2


@pytest.fixture(params=[10, 1000, 10000])
def ic_mixed_profile(request):
    """
    IC profile over mixed str/int candidates.
    """
    return ic_profile_generator(candidates=MIXED_CANDS, number_of_ballots=request.param)


@pytest.mark.parametrize(
    "make_election",
    [
        pytest.param(
            lambda profile: Borda(profile, n_seats=N_SEATS, tiebreak="random"), id="borda"
        ),
        pytest.param(
            lambda profile: Plurality(profile, n_seats=N_SEATS, tiebreak="random"), id="plurality"
        ),
        pytest.param(lambda profile: STV(profile, n_seats=N_SEATS, tiebreak="random"), id="stv"),
    ],
)
def test_election_runs_with_mixed_candidates(ic_mixed_profile, make_election):
    """
    Election Fuzz Test: Run Borda, STV, and Plurality elections on profiles
    with mixed candidate types. IC ballot generator will generate
    ballots with mix of string and integer candidates
    with varying number of ballots (10, 1000, 10000).
    """
    election = make_election(ic_mixed_profile)
    elected = election.get_elected()
    n_elected = sum(len(seat) for seat in elected)
    assert n_elected == N_SEATS


SCORE_VECTOR = [5, 4, 3, 2, 1, 0]

UTILS_FXNS = [
    pytest.param(utils.first_place_votes, id="first_place_votes"),
    pytest.param(utils.mentions, id="mentions"),
    pytest.param(utils.borda_scores, id="borda_scores"),
    pytest.param(utils.ballots_by_first_cand, id="ballots_by_first_cand"),
    pytest.param(utils.ballot_lengths, id="ballot_lengths"),
    pytest.param(utils.add_missing_cands, id="add_missing_cands"),
    pytest.param(
        partial(utils.score_dict_from_score_vector, score_vector=SCORE_VECTOR),
        id="score_dict_from_score_vector",
    ),
]


@pytest.mark.parametrize("utils_fxns", UTILS_FXNS)
def test_utils_fxns_accept_mixed_candidates(ic_mixed_profile, utils_fxns):
    """
    utils.py Functions Fuzz Test: Run all utils.py functions with
    profiles of mixed candidate types.
    The same profiles from the Election Fuzz Test.
    """
    result = utils_fxns(ic_mixed_profile)
    assert result is not None
