"""
Reproducibility and non-determinism tests for ballot generators, elections, and core modules.
"""

import multiprocessing
import os
import random

import pytest

from votekit.ballot import RankBallot
from votekit.ballot_generator.bloc_slate_generator.cambridge import (
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.cumulative import (
    name_cumulative_ballot_generator_by_bloc,
    name_cumulative_profile_generator,
)
from votekit.ballot_generator.bloc_slate_generator.name_bradley_terry import (
    name_bt_profile_generator,
    name_bt_profile_generator_using_mcmc,
    name_bt_profiles_by_bloc_generator,
    name_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.name_plackett_luce import (
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
)
from votekit.ballot_generator.bloc_slate_generator.slate_bradley_terry import (
    slate_bt_profile_generator,
    slate_bt_profile_generator_using_mcmc,
    slate_bt_profiles_by_bloc_generator,
    slate_bt_profiles_by_bloc_generator_using_mcmc,
)
from votekit.ballot_generator.bloc_slate_generator.slate_plackett_luce import (
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
)
from votekit.ballot_generator.std_generator.impartial_anon_culture import iac_profile_generator
from votekit.ballot_generator.std_generator.impartial_culture import ic_profile_generator
from votekit.ballot_generator.std_generator.spacial import (
    clustered_spacial_profile_and_positions_generator,
    onedim_spacial_profile_generator,
    spacial_profile_and_positions_generator,
)
from votekit.elections.election_types.ranking.boosted_random_dictator import BoostedRandomDictator
from votekit.elections.election_types.ranking.plurality import SNTV, Plurality
from votekit.elections.election_types.ranking.random_dictator import RandomDictator
from votekit.elections.election_types.ranking.stv.stv import IRV, STV
from votekit.elections.transfers import random_transfer
from votekit.pref_interval import PreferenceInterval
from votekit.pref_profile import RankProfile
from votekit.utils import elect_cands_from_set_ranking, tiebreak_set

NUM_LOOPS = 20
RANDOM_SEED = 10

# =============================================================================
# Ballot Generators
# =============================================================================


ALL_SLATE_GENERATORS = [
    name_cumulative_profile_generator,
    name_cumulative_ballot_generator_by_bloc,
    name_bt_profile_generator,
    name_bt_profiles_by_bloc_generator,
    name_bt_profile_generator_using_mcmc,
    name_bt_profiles_by_bloc_generator_using_mcmc,
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    slate_bt_profile_generator_using_mcmc,
    slate_bt_profiles_by_bloc_generator_using_mcmc,
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
]


@pytest.fixture
def bloc_config():
    return BlocSlateConfig(
        n_voters=400,
        slate_to_candidates={"A": ["A1", 2], "B": [1, "B2"]},
        bloc_proportions={"A": 0.6, "B": 0.4},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"A1": 0.7, 2: 0.3}),
                "B": PreferenceInterval({1: 0.4, "B2": 0.6}),
            },
            "B": {
                "A": PreferenceInterval({"A1": 0.3, 2: 0.7}),
                "B": PreferenceInterval({1: 0.6, "B2": 0.4}),
            },
        },
        cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"A": 0.3, "B": 0.7}},
    )


@pytest.mark.parametrize("fn", ALL_SLATE_GENERATORS, ids=lambda f: f.__name__)
def test_bloc_generator_reproducible(fn, bloc_config):
    result = fn(bloc_config, random_seed=RANDOM_SEED)
    for _ in range(NUM_LOOPS):
        assert result == fn(bloc_config, random_seed=RANDOM_SEED)


@pytest.mark.parametrize("fn", ALL_SLATE_GENERATORS, ids=lambda f: f.__name__)
def test_bloc_generator_nondeterministic(fn, bloc_config):
    results = [fn(bloc_config, random_seed=None) for _ in range(NUM_LOOPS)]
    assert not all(r == results[0] for r in results)


CANDIDATES = ["A", "B", 1, 2]
N_BALLOTS = 200

ALL_STD_GENERATORS = [
    ic_profile_generator,
    iac_profile_generator,
    onedim_spacial_profile_generator,
]


@pytest.mark.parametrize("fn", ALL_STD_GENERATORS, ids=lambda f: f.__name__)
def test_std_generator_reproducible(fn):
    result = fn(candidates=CANDIDATES, number_of_ballots=N_BALLOTS, random_seed=RANDOM_SEED)
    for _ in range(NUM_LOOPS):
        assert result == fn(
            candidates=CANDIDATES, number_of_ballots=N_BALLOTS, random_seed=RANDOM_SEED
        )


@pytest.mark.parametrize("fn", ALL_STD_GENERATORS, ids=lambda f: f.__name__)
def test_std_generator_nondeterministic(fn):
    results = [
        fn(candidates=CANDIDATES, number_of_ballots=N_BALLOTS, random_seed=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


CLUSTERED_N_BALLOTS = {"A": 50, "B": 50, 1: 50, 2: 50}


def test_std_generator_cluster_args_reproducible():
    result = clustered_spacial_profile_and_positions_generator(
        number_of_ballots=CLUSTERED_N_BALLOTS, candidates=CANDIDATES, random_seed=RANDOM_SEED
    )
    for _ in range(NUM_LOOPS):
        assert (
            result[0]
            == clustered_spacial_profile_and_positions_generator(
                number_of_ballots=CLUSTERED_N_BALLOTS,
                candidates=CANDIDATES,
                random_seed=RANDOM_SEED,
            )[0]
        )


def test_std_generator_with_cluster_nondeterministic():
    results = [
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=CLUSTERED_N_BALLOTS, candidates=CANDIDATES, random_seed=None
        )
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r[0] == results[0][0] for r in results)


def test_spacial_positions_generator_reproducible():
    result = spacial_profile_and_positions_generator(
        number_of_ballots=N_BALLOTS, candidates=CANDIDATES, random_seed=RANDOM_SEED
    )
    for _ in range(NUM_LOOPS):
        assert (
            result[0]
            == spacial_profile_and_positions_generator(
                number_of_ballots=N_BALLOTS, candidates=CANDIDATES, random_seed=RANDOM_SEED
            )[0]
        )


def test_spacial_positions_generator_nondeterministic():
    results = [
        spacial_profile_and_positions_generator(
            number_of_ballots=N_BALLOTS, candidates=CANDIDATES, random_seed=None
        )
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r[0] == results[0][0] for r in results)


# =============================================================================
# Elections
# =============================================================================


@pytest.fixture
def tied_profile():
    # All three candidates tied at 3 FPV — forces random tiebreak in every election type.
    return RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=3),
            RankBallot(ranking=[{"B"}, {"A"}, {"C"}], weight=3),
            RankBallot(ranking=[{"C"}, {"A"}, {"B"}], weight=3),
        )
    )


@pytest.fixture
def stv_profile():
    # Integer weights required by random_transfer.
    # 30 total votes, droop quota for 2 seats = 11.
    # A gets 15 FPV → elected with surplus 4, randomly transferred.
    return RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}, {1}, {2}], weight=10),
            RankBallot(ranking=[{"A"}, {1}, {"B"}, {2}], weight=5),
            RankBallot(ranking=[{"B"}, {"A"}, {1}, {2}], weight=8),
            RankBallot(ranking=[{1}, {"A"}, {"B"}, {2}], weight=7),
        )
    )


ELECTION_CASES = [
    pytest.param(Plurality, {"n_seats": 1, "tiebreak": "random"}, id="Plurality"),
    pytest.param(SNTV, {"n_seats": 1, "tiebreak": "random"}, id="SNTV"),
    pytest.param(IRV, {"tiebreak": "random"}, id="IRV"),
    pytest.param(RandomDictator, {"n_seats": 1}, id="RandomDictator"),
    pytest.param(BoostedRandomDictator, {"n_seats": 1}, id="BoostedRandomDictator"),
]


@pytest.mark.parametrize("cls,kwargs", ELECTION_CASES)
def test_election_reproducible(cls, kwargs, tied_profile):
    result = cls(tied_profile, **kwargs, random_seed=RANDOM_SEED).get_elected()
    for _ in range(NUM_LOOPS):
        assert result == cls(tied_profile, **kwargs, random_seed=RANDOM_SEED).get_elected()


@pytest.mark.parametrize("cls,kwargs", ELECTION_CASES)
def test_election_nondeterministic(cls, kwargs, tied_profile):
    results = [
        cls(tied_profile, **kwargs, random_seed=None).get_elected() for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


def test_stv_random_transfer_reproducible(stv_profile):
    result = STV(
        stv_profile, n_seats=2, transfer=random_transfer, random_seed=RANDOM_SEED
    ).get_elected()
    for _ in range(NUM_LOOPS):
        assert (
            result
            == STV(
                stv_profile, n_seats=2, transfer=random_transfer, random_seed=RANDOM_SEED
            ).get_elected()
        )


# =============================================================================
# Module functions: utils, transfers, pref_interval
# =============================================================================
CAND_SET = frozenset({"A", "B", 1, 2})


def test_tiebreak_set_reproducible():
    result = tiebreak_set(CAND_SET, tiebreak="random", rng=random.Random(RANDOM_SEED))
    for _ in range(NUM_LOOPS):
        assert result == tiebreak_set(CAND_SET, tiebreak="random", rng=random.Random(RANDOM_SEED))


def test_tiebreak_set_nondeterministic():
    results = [tiebreak_set(CAND_SET, tiebreak="random", rng=None) for _ in range(NUM_LOOPS)]
    assert not all(r == results[0] for r in results)


def test_elect_cands_from_set_ranking_reproducible():
    result = elect_cands_from_set_ranking(
        [CAND_SET], n_seats=1, tiebreak="random", rng=random.Random(RANDOM_SEED)
    )
    for _ in range(NUM_LOOPS):
        assert result == elect_cands_from_set_ranking(
            [CAND_SET], n_seats=1, tiebreak="random", rng=random.Random(RANDOM_SEED)
        )


def test_elect_cands_from_set_ranking_nondeterministic():
    ranking = [frozenset({"A", "B", "C"})]
    results = [
        elect_cands_from_set_ranking(ranking, n_seats=1, tiebreak="random", rng=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


def test_random_transfer_reproducible():
    ballots = [
        RankBallot(ranking=[{"A"}, {"B"}, {1}], weight=5),
        RankBallot(ranking=[{"A"}, {1}, {"B"}], weight=5),
        RankBallot(ranking=[{"B"}, {"A"}, {1}], weight=3),
    ]
    result = random_transfer(
        "A", fpv=10, ballots=ballots, threshold=8, rng=random.Random(RANDOM_SEED)
    )
    for _ in range(NUM_LOOPS):
        assert result == random_transfer(
            "A", fpv=10, ballots=ballots, threshold=8, rng=random.Random(RANDOM_SEED)
        )


def test_random_transfer_nondeterministic():
    ballots = [
        RankBallot(ranking=[{"A"}, {"B"}, {1}], weight=5),
        RankBallot(ranking=[{"A"}, {1}, {"B"}], weight=5),
        RankBallot(ranking=[{"B"}, {"A"}, {1}], weight=3),
    ]
    results = [
        random_transfer("A", fpv=10, ballots=ballots, threshold=8, rng=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


def test_pref_interval_from_dirichlet_reproducible():
    results = PreferenceInterval.from_dirichlet(CANDIDATES, alpha=1.0, random_seed=RANDOM_SEED)
    for _ in range(NUM_LOOPS):
        assert results == PreferenceInterval.from_dirichlet(
            CANDIDATES, alpha=1.0, random_seed=RANDOM_SEED
        )


def test_pref_interval_from_dirichlet_nondeterministic():
    results = [
        PreferenceInterval.from_dirichlet(CANDIDATES, alpha=1.0, random_seed=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


# =============================================================================
# PYTHONHASHSEED stability: spawn processes with explicit hash seeds
# =============================================================================
# Python randomises the seed used by hash() per interpreter invocation via PYTHONHASHSEED.
# These tests set PYTHONHASHSEED explicitly in os.environ before each spawn so each child
# starts with a known seed and confirms that the result is the same if given the same random_seed.

_HASH_SEEDS = ["0", "1", "10", "100"]


def _spawn_with_seed(fn, seed):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    prev = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = seed
    p = ctx.Process(target=fn, args=(q,))
    p.start()
    if prev is None:
        os.environ.pop("PYTHONHASHSEED")
    else:
        os.environ["PYTHONHASHSEED"] = prev
    p.join()
    return q.get()


def _tiebreak_worker(queue):
    import random

    from votekit.utils import tiebreak_set

    queue.put(
        (
            tiebreak_set(
                frozenset({"A", "B", 1, 2}), tiebreak="random", rng=random.Random(RANDOM_SEED)
            ),
            os.environ.get("PYTHONHASHSEED"),
        )
    )


def _elect_cands_worker(queue):
    import random

    from votekit.utils import elect_cands_from_set_ranking

    queue.put(
        (
            elect_cands_from_set_ranking(
                [frozenset({"A", "B", 1, 2})],
                n_seats=2,
                tiebreak="random",
                rng=random.Random(RANDOM_SEED),
            ),
            os.environ.get("PYTHONHASHSEED"),
        )
    )


def _election_worker(queue):
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}, {1}, {2}], weight=3),
            RankBallot(ranking=[{"B"}, {"A"}, {1}, {2}], weight=3),
            RankBallot(ranking=[{1}, {"A"}, {"B"}, {2}], weight=3),
            RankBallot(ranking=[{2}, {"A"}, {"B"}, {1}], weight=3),
        )
    )
    queue.put(
        (
            Plurality(profile, n_seats=1, tiebreak="random", random_seed=RANDOM_SEED).get_elected(),
            os.environ.get("PYTHONHASHSEED"),
        )
    )


def test_tiebreak_set_hashseed_stable():
    outputs = [_spawn_with_seed(_tiebreak_worker, seed) for seed in _HASH_SEEDS]
    results, hashes = zip(*outputs)
    assert list(hashes) == _HASH_SEEDS
    assert all(r == results[0] for r in results[1:])


def test_elect_cands_hashseed_stable():
    outputs = [_spawn_with_seed(_elect_cands_worker, seed) for seed in _HASH_SEEDS]
    results, hashes = zip(*outputs)
    assert list(hashes) == _HASH_SEEDS
    assert all(r == results[0] for r in results[1:])


def test_election_tiebreak_hashseed_stable():
    outputs = [_spawn_with_seed(_election_worker, seed) for seed in _HASH_SEEDS]
    results, hashes = zip(*outputs)
    assert list(hashes) == _HASH_SEEDS
    assert all(r == results[0] for r in results[1:])
