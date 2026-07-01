from functools import partial

import pytest

from votekit import utils
from votekit.ballot_generator import ic_profile_generator
from votekit.elections import (
    IRV,
    SNTV,
    STV,
    Alaska,
    BoostedRandomDictator,
    Borda,
    CondoBorda,
    FastSTV,
    Plurality,
    PluralityVeto,
    RandomDictator,
    RankedPairs,
    Schulze,
    SequentialRCV,
    SerialVeto,
    SimultaneousVeto,
)

MIXED_CANDS = ["A", "B", "1", 1, 2, 3]
N_SEATS = 1

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*appear as both str and int.*within a ballot.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*appear as both str and int.*within a profile.*:UserWarning"
    ),
]


@pytest.fixture(params=[10, 1000, 10000])
def ic_mixed_profile(request):
    """
    IC RankProfile over mixed str/int candidates.
    """
    return ic_profile_generator(candidates=MIXED_CANDS, number_of_ballots=request.param)


@pytest.mark.parametrize(
    "make_election",
    [
        pytest.param(
            lambda profile: Borda(profile, n_seats=N_SEATS, tiebreak="random"), id="borda"
        ),
        pytest.param(lambda profile: CondoBorda(profile, n_seats=N_SEATS), id="condo_borda"),
        pytest.param(
            lambda profile: SequentialRCV(profile, n_seats=N_SEATS),
            id="sequential_rcv",
        ),
        pytest.param(lambda profile: Alaska(profile, m_1=N_SEATS + 2, m_2=N_SEATS), id="alaska"),
        pytest.param(
            lambda profile: Plurality(profile, n_seats=N_SEATS, tiebreak="random"), id="plurality"
        ),
        pytest.param(lambda profile: IRV(profile, tiebreak="random"), id="irv"),
        pytest.param(lambda profile: STV(profile, n_seats=N_SEATS, tiebreak="random"), id="stv"),
        pytest.param(lambda profile: SNTV(profile, n_seats=N_SEATS, tiebreak="random"), id="sntv"),
        pytest.param(
            lambda profile: FastSTV(profile, n_seats=N_SEATS, tiebreak="random"), id="fast_stv"
        ),
        pytest.param(
            lambda profile: RankedPairs(profile, n_seats=N_SEATS, tiebreak="lexicographic"),
            id="ranked_pairs",
        ),
        pytest.param(
            lambda profile: SimultaneousVeto(profile, n_seats=N_SEATS, tiebreak="random"),
            id="simultaneous_veto",
        ),
        pytest.param(
            lambda profile: PluralityVeto(profile, n_seats=N_SEATS, tiebreak="lex"),
            id="plurality_veto",
        ),
        pytest.param(
            lambda profile: SerialVeto(profile, n_seats=N_SEATS, tiebreak="lex"),
            id="serial_veto",
        ),
        pytest.param(
            lambda profile: RandomDictator(profile, n_seats=N_SEATS),
            id="random_dictator",
        ),
        pytest.param(
            lambda profile: BoostedRandomDictator(profile, n_seats=N_SEATS),
            id="boosted_random_dictator",
        ),
        pytest.param(
            lambda profile: Schulze(profile, n_seats=N_SEATS, tiebreak="lexicographic"),
            id="schulze",
        ),
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
