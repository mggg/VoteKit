"""
Reproducibility and non-determinism tests for ballot generators, elections, and core modules.
"""

import os
import random
import subprocess
import sys
from string import Template

import numpy as np
import pytest

from votekit.ballot import RankBallot, ScoreBallot
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
from votekit.elections.election_types.approval import Approval
from votekit.elections.election_types.block_plurality import BlockPlurality
from votekit.elections.election_types.ranking.alaska import Alaska
from votekit.elections.election_types.ranking.boosted_random_dictator import BoostedRandomDictator
from votekit.elections.election_types.ranking.borda import Borda
from votekit.elections.election_types.ranking.plurality import SNTV, Plurality
from votekit.elections.election_types.ranking.plurality_veto import PluralityVeto, SerialVeto
from votekit.elections.election_types.ranking.random_dictator import RandomDictator
from votekit.elections.election_types.ranking.schulze import Schulze
from votekit.elections.election_types.ranking.simultaneous_veto import SimultaneousVeto
from votekit.elections.election_types.ranking.stv.stv import IRV, STV, AlbanySTV, FastIRV, FastSTV
from votekit.elections.election_types.ranking.top_two import TopTwo
from votekit.elections.election_types.scores.cumulative import Cumulative
from votekit.elections.election_types.scores.limited import Limited
from votekit.elections.transfers import random_transfer
from votekit.pref_interval import PreferenceInterval
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.utils import elect_cands_from_set_ranking, tiebreak_set

NUM_LOOPS = 20
RNG_SEED = 47

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
    result = fn(bloc_config, rng_seed=RNG_SEED)
    for _ in range(NUM_LOOPS):
        assert result == fn(bloc_config, rng_seed=RNG_SEED)


@pytest.mark.parametrize("fn", ALL_SLATE_GENERATORS, ids=lambda f: f.__name__)
def test_bloc_generator_nondeterministic(fn, bloc_config):
    results = [fn(bloc_config, rng_seed=None) for _ in range(NUM_LOOPS)]
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
    result = fn(candidates=CANDIDATES, number_of_ballots=N_BALLOTS, rng_seed=RNG_SEED)
    for _ in range(NUM_LOOPS):
        assert result == fn(candidates=CANDIDATES, number_of_ballots=N_BALLOTS, rng_seed=RNG_SEED)


@pytest.mark.parametrize("fn", ALL_STD_GENERATORS, ids=lambda f: f.__name__)
def test_std_generator_nondeterministic(fn):
    results = [
        fn(candidates=CANDIDATES, number_of_ballots=N_BALLOTS, rng_seed=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


CLUSTERED_N_BALLOTS = {"A": 50, "B": 50, 1: 50, 2: 50}


def test_std_generator_cluster_args_reproducible():
    result = clustered_spacial_profile_and_positions_generator(
        number_of_ballots=CLUSTERED_N_BALLOTS, candidates=CANDIDATES, rng_seed=RNG_SEED
    )
    for _ in range(NUM_LOOPS):
        assert (
            result[0]
            == clustered_spacial_profile_and_positions_generator(
                number_of_ballots=CLUSTERED_N_BALLOTS,
                candidates=CANDIDATES,
                rng_seed=RNG_SEED,
            )[0]
        )


def test_std_generator_with_cluster_nondeterministic():
    results = [
        clustered_spacial_profile_and_positions_generator(
            number_of_ballots=CLUSTERED_N_BALLOTS, candidates=CANDIDATES, rng_seed=None
        )
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r[0] == results[0][0] for r in results)


def test_spacial_positions_generator_reproducible():
    result = spacial_profile_and_positions_generator(
        number_of_ballots=N_BALLOTS, candidates=CANDIDATES, rng_seed=RNG_SEED
    )
    for _ in range(NUM_LOOPS):
        assert (
            result[0]
            == spacial_profile_and_positions_generator(
                number_of_ballots=N_BALLOTS, candidates=CANDIDATES, rng_seed=RNG_SEED
            )[0]
        )


def test_spacial_positions_generator_nondeterministic():
    results = [
        spacial_profile_and_positions_generator(
            number_of_ballots=N_BALLOTS, candidates=CANDIDATES, rng_seed=None
        )
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r[0] == results[0][0] for r in results)


# =============================================================================
# Elections
# =============================================================================


@pytest.fixture
def tied_rank_profile():
    # All three candidates tied at 3 FPV — forces random tiebreak in every election type.
    return RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}, {1}], weight=3),
            RankBallot(ranking=[{"B"}, {1}, {"A"}], weight=3),
            RankBallot(ranking=[{1}, {"A"}, {"B"}], weight=3),
        )
    )


@pytest.fixture
def tied_score_profile():
    # All three candidates tied at 3 FPV — forces random tiebreak in every election type.
    return ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 1, "B": 0, 1: 0}, weight=3),
            ScoreBallot(scores={"A": 0, "B": 1, 1: 0}, weight=3),
            ScoreBallot(scores={"A": 0, "B": 0, 1: 1}, weight=3),
        )
    )


@pytest.fixture
def stv_profile():
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


@pytest.fixture
def stv_transfer_profile():
    # 30 total votes, droop quota for 2 seats = 5.
    # A gets 6 FPV → elected with surplus 1, randomly transferred 50/50 to B or 1.
    return RankProfile(
        ballots=(
            RankBallot(ranking=[{"A"}, {"B"}, {1}], weight=3),
            RankBallot(ranking=[{"A"}, {1}, {"B"}], weight=3),
            RankBallot(ranking=[{"B"}, {"A"}, {1}], weight=3),
            RankBallot(ranking=[{1}, {"A"}, {"B"}], weight=3),
        )
    )


RANK_ELECTION_CASES = [
    pytest.param(Plurality, {"n_seats": 1, "tiebreak": "random"}, id="Plurality"),
    pytest.param(SNTV, {"n_seats": 1, "tiebreak": "random"}, id="SNTV"),
    pytest.param(FastSTV, {"n_seats": 1, "tiebreak": "random"}, id="FastSTV"),
    pytest.param(AlbanySTV, {"n_seats": 1, "tiebreak": "random"}, id="AlbanySTV"),
    pytest.param(IRV, {"tiebreak": "random"}, id="IRV"),
    pytest.param(FastIRV, {"tiebreak": "random"}, id="FastIRV"),
    pytest.param(RandomDictator, {"n_seats": 1}, id="RandomDictator"),
    pytest.param(BoostedRandomDictator, {"n_seats": 1}, id="BoostedRandomDictator"),
    pytest.param(BlockPlurality, {"n_seats": 1, "tiebreak": "random"}, id="BlockPlurality"),
    pytest.param(Alaska, {"m_2": 1, "tiebreak": "random"}, id="Alaska"),
    pytest.param(TopTwo, {"tiebreak": "random"}, id="TopTwo"),
    pytest.param(Schulze, {"n_seats": 1, "tiebreak": "random"}, id="Schulze"),
    pytest.param(Borda, {"n_seats": 1, "tiebreak": "random"}, id="Borda"),
    pytest.param(PluralityVeto, {"n_seats": 1, "tiebreak": "random"}, id="PluralityVeto"),
    pytest.param(SerialVeto, {"n_seats": 1, "tiebreak": "random"}, id="SerialVeto"),
    pytest.param(SimultaneousVeto, {"n_seats": 1, "tiebreak": "random"}, id="SimultaneousVeto"),
]
SCORE_ELECTION_CASES = [
    pytest.param(BlockPlurality, {"n_seats": 1, "tiebreak": "random"}, id="BlockPlurality"),
    pytest.param(Cumulative, {"n_seats": 1, "tiebreak": "random"}, id="Cumulative"),
    pytest.param(Approval, {"n_seats": 1, "tiebreak": "random"}, id="Approval"),
    pytest.param(Limited, {"n_seats": 1, "tiebreak": "random"}, id="Limited"),
]
RANDOM_TRANSFER_STV_CASES = [
    pytest.param(FastSTV, {"transfer": "cambridge_random"}, id="FastSTV, cambridge_random"),
    pytest.param(FastSTV, {"transfer": "fractional_random"}, id="FastSTV, fractional_random"),
    pytest.param(AlbanySTV, {"transfer": "cambridge_random"}, id="AlbanySTV, cambridge_random"),
    pytest.param(AlbanySTV, {"transfer": "fractional_random"}, id="AlbanySTV, fractional_random"),
]


@pytest.mark.parametrize("cls,kwargs", RANK_ELECTION_CASES)
def test_rank_election_reproducible(cls, kwargs, tied_rank_profile):
    result = cls(tied_rank_profile, **kwargs, rng_seed=RNG_SEED).get_elected()
    for _ in range(NUM_LOOPS):
        assert result == cls(tied_rank_profile, **kwargs, rng_seed=RNG_SEED).get_elected()


@pytest.mark.parametrize("cls,kwargs", RANK_ELECTION_CASES)
def test_rank_election_nondeterministic(cls, kwargs, tied_rank_profile):
    results = [
        cls(tied_rank_profile, **kwargs, rng_seed=None).get_elected() for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


@pytest.mark.parametrize("cls,kwargs", SCORE_ELECTION_CASES)
def test_score_election_reproducible(cls, kwargs, tied_score_profile):
    result = cls(tied_score_profile, **kwargs, rng_seed=RNG_SEED).get_elected()
    for _ in range(NUM_LOOPS):
        assert result == cls(tied_score_profile, **kwargs, rng_seed=RNG_SEED).get_elected()


@pytest.mark.parametrize("cls,kwargs", SCORE_ELECTION_CASES)
def test_score_election_nondeterministic(cls, kwargs, tied_score_profile):
    results = [
        cls(tied_score_profile, **kwargs, rng_seed=None).get_elected() for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


def test_stv_random_transfer_reproducible(stv_profile):
    result = STV(stv_profile, n_seats=2, transfer=random_transfer, rng_seed=RNG_SEED).get_elected()
    for _ in range(NUM_LOOPS):
        assert (
            result
            == STV(
                stv_profile, n_seats=2, transfer=random_transfer, rng_seed=RNG_SEED
            ).get_elected()
        )


@pytest.mark.parametrize("cls,kwargs", RANDOM_TRANSFER_STV_CASES)
def test_numpy_random_transfer_in_elections_reproducible(cls, kwargs, stv_transfer_profile):
    result = cls(stv_transfer_profile, n_seats=2, **kwargs, rng_seed=RNG_SEED).get_elected()
    for _ in range(NUM_LOOPS):
        assert (
            result
            == cls(stv_transfer_profile, n_seats=2, **kwargs, rng_seed=RNG_SEED).get_elected()
        )


@pytest.mark.parametrize("cls,kwargs", RANDOM_TRANSFER_STV_CASES)
def test_numpy_random_transfer_in_elections_nondeterministic(cls, kwargs, stv_transfer_profile):
    results = [
        cls(stv_transfer_profile, n_seats=2, **kwargs).get_elected() for _ in range(NUM_LOOPS)
    ]
    assert not all(result == results[0] for result in results)


# =============================================================================
# Module functions: utils, transfers, pref_interval
# =============================================================================
CAND_SET = frozenset({"A", "B", 1, 2})


def test_tiebreak_set_reproducible():
    result = tiebreak_set(CAND_SET, tiebreak="random", rng=random.Random(RNG_SEED))
    for _ in range(NUM_LOOPS):
        assert result == tiebreak_set(CAND_SET, tiebreak="random", rng=random.Random(RNG_SEED))


def test_tiebreak_set_nondeterministic():
    results = [tiebreak_set(CAND_SET, tiebreak="random", rng=None) for _ in range(NUM_LOOPS)]
    assert not all(r == results[0] for r in results)


def test_elect_cands_from_set_ranking_reproducible():
    result = elect_cands_from_set_ranking(
        [CAND_SET], n_seats=1, tiebreak="random", rng=random.Random(RNG_SEED)
    )
    for _ in range(NUM_LOOPS):
        assert result == elect_cands_from_set_ranking(
            [CAND_SET], n_seats=1, tiebreak="random", rng=random.Random(RNG_SEED)
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
    result = random_transfer("A", fpv=10, ballots=ballots, threshold=8, rng=random.Random(RNG_SEED))
    for _ in range(NUM_LOOPS):
        assert result == random_transfer(
            "A", fpv=10, ballots=ballots, threshold=8, rng=random.Random(RNG_SEED)
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
    results = PreferenceInterval.from_dirichlet(CANDIDATES, alpha=1.0, rng_seed=RNG_SEED)
    for _ in range(NUM_LOOPS):
        assert results == PreferenceInterval.from_dirichlet(
            CANDIDATES, alpha=1.0, rng_seed=RNG_SEED
        )


def test_from_dirichlet_respects_passed_numpy_rng():
    for _ in range(10):
        a = PreferenceInterval.from_dirichlet(
            ["A", "B", "C"], alpha=1.0, numpy_rng=np.random.default_rng(42)
        )
        b = PreferenceInterval.from_dirichlet(
            ["A", "B", "C"], alpha=1.0, numpy_rng=np.random.default_rng(42)
        )
        assert a == b


def test_pref_interval_from_dirichlet_nondeterministic():
    results = [
        PreferenceInterval.from_dirichlet(CANDIDATES, alpha=1.0, rng_seed=None)
        for _ in range(NUM_LOOPS)
    ]
    assert not all(r == results[0] for r in results)


# =============================================================================
# PYTHONHASHSEED stability: spawn processes with explicit hash seeds
# =============================================================================
# Python randomises the seed used by hash() per interpreter invocation via PYTHONHASHSEED.
# These tests set PYTHONHASHSEED explicitly in os.environ before each spawn so each child
# starts with a known seed and confirms that the result is the same if given the same rng_seed.

_HASH_SEEDS = ["0", "1", "10", "100"]


def _run_across_hash_seeds(child_code):
    outputs = []
    for hs in _HASH_SEEDS:
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run(
            [sys.executable, "-c", child_code], env=env, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        outputs.append(out.stdout.strip())
    return outputs


_TIEBREAK_SET_CHILD = r"""
import random
from votekit.utils import tiebreak_set

print(repr(tiebreak_set(frozenset({"A", "B", 1, 2}), tiebreak="random", rng=random.Random(10))))
"""


def test_tiebreak_set_hashseed_stable():
    outputs = _run_across_hash_seeds(_TIEBREAK_SET_CHILD)
    assert all(o == outputs[0] for o in outputs)


_ELECT_CANDS_CHILD = r"""
import random
from votekit.utils import elect_cands_from_set_ranking

print(repr(elect_cands_from_set_ranking(
    [frozenset({"A", "B", 1, 2})], n_seats=2, tiebreak="random", rng=random.Random(10)
)[0]))
"""


def test_elect_cands_hashseed_stable():
    outputs = _run_across_hash_seeds(_ELECT_CANDS_CHILD)
    assert all(o == outputs[0] for o in outputs)


_ELECTION_TIEBREAK_CHILD = r"""
from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.elections.election_types.ranking.plurality import Plurality

profile = RankProfile(
    ballots=(
        RankBallot(ranking=[{"A"}, {"B"}, {1}, {2}], weight=3),
        RankBallot(ranking=[{"B"}, {"A"}, {1}, {2}], weight=3),
        RankBallot(ranking=[{1}, {"A"}, {"B"}, {2}], weight=3),
        RankBallot(ranking=[{2}, {"A"}, {"B"}, {1}], weight=3),
    )
)
print(repr(Plurality(profile, n_seats=1, tiebreak="random", rng_seed=10).get_elected()))
"""


def test_election_tiebreak_hashseed_stable():
    outputs = _run_across_hash_seeds(_ELECTION_TIEBREAK_CHILD)
    assert all(o == outputs[0] for o in outputs)


_CUMULATIVE_CHILD = r"""
from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
from votekit.elections.election_types.scores.cumulative import Cumulative

profile = ScoreProfile(
    ballots=(
        ScoreBallot(scores={"A": 1, "B": 0, 1: 0, 2: 0}, weight=3),
        ScoreBallot(scores={"A": 0, "B": 1, 1: 0, 2: 0}, weight=3),
        ScoreBallot(scores={"A": 0, "B": 0, 1: 1, 2: 0}, weight=3),
        ScoreBallot(scores={"A": 0, "B": 0, 1: 0, 2: 1}, weight=3),
    )
)
print(repr(Cumulative(profile, n_seats=1, tiebreak="random", rng_seed=10).get_elected()))
"""


def test_cumulative_hashseed_stable():
    outputs = _run_across_hash_seeds(_CUMULATIVE_CHILD)
    assert all(o == outputs[0] for o in outputs), (
        f"same rng_seed gave different results across hash seeds: {outputs}"
    )


_RANDOM_DICTATOR_CHILD = Template(
    r"""
from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.elections.election_types.ranking.$mod import $cls

cands = ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta", "Eta", "Theta"]
rots = [cands[i:] + cands[:i] for i in range(len(cands))]
profile = RankProfile(
    ballots=tuple(RankBallot(ranking=[{c} for c in rot], weight=3) for rot in rots)
)
print(repr($cls(profile, n_seats=5, rng_seed=10).get_elected()))
"""
)


def test_boosted_random_dictator_hashseed_stable():
    outputs = _run_across_hash_seeds(
        _RANDOM_DICTATOR_CHILD.substitute(
            mod="boosted_random_dictator", cls="BoostedRandomDictator"
        )
    )
    assert all(o == outputs[0] for o in outputs)


def test_random_dictator_hashseed_stable():
    outputs = _run_across_hash_seeds(
        _RANDOM_DICTATOR_CHILD.substitute(mod="random_dictator", cls="RandomDictator")
    )
    assert all(o == outputs[0] for o in outputs)


_PLURALITY_VETO_CHILD = r"""
from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.elections.election_types.ranking.plurality_veto import PluralityVeto

cands = ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta"]
rots = [cands[i:] + cands[:i] for i in range(len(cands))]
profile = RankProfile(
    ballots=tuple(RankBallot(ranking=[{rot[0]}, {rot[1]}], weight=2) for rot in rots),
    candidates=cands,
    max_ranking_length=6,
)
print(repr(PluralityVeto(profile, n_seats=2, tiebreak="random", rng_seed=10).get_elected()))
"""


def test_plurality_veto_random_tiebreak_hashseed_stable():
    outputs = _run_across_hash_seeds(_PLURALITY_VETO_CHILD)
    assert all(o == outputs[0] for o in outputs)


_CUMULATIVE_PROFILE_GENERATOR_CHILD = r"""
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.cumulative import (
    name_cumulative_profile_generator,
)
from votekit.pref_interval import PreferenceInterval

config = BlocSlateConfig(
    n_voters=100,
    slate_to_candidates={"A": ["A1", "A2"], "B": ["B1", "B2"]},
    bloc_proportions={"A": 0.6, "B": 0.4},
    preference_mapping={
        "A": {
            "A": PreferenceInterval({"A1": 0.7, "A2": 0.3}),
            "B": PreferenceInterval({"B1": 0.4, "B2": 0.6}),
        },
        "B": {
            "A": PreferenceInterval({"A1": 0.3, "A2": 0.7}),
            "B": PreferenceInterval({"B1": 0.6, "B2": 0.4}),
        },
    },
    cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"A": 0.3, "B": 0.7}},
)
profile = name_cumulative_profile_generator(config, rng_seed=10)
print(repr(sorted(str(b) for b in profile.ballots)))
"""


def test_cumulative_profile_generator_hashseed_stable():
    outputs = _run_across_hash_seeds(_CUMULATIVE_PROFILE_GENERATOR_CHILD)
    assert all(o == outputs[0] for o in outputs)
