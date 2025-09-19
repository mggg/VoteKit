import math

from votekit.ballot_generator import (
    CambridgeSampler,
    slate_BradleyTerry,
    name_Cumulative,
)
from votekit.pref_profile import RankProfile, ScoreProfile


def test_SBT_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    gen = slate_BradleyTerry.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(gen.pref_intervals_by_bloc[curr_bloc][b].interval.values()), 1)
        for curr_bloc in blocs.keys()
        for b in blocs.keys()
    )

    profile = gen.generate_profile(3)
    assert type(profile) is RankProfile


def test_name_Cumulative_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    gen = name_Cumulative.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
        num_votes=2,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(gen.pref_intervals_by_bloc[curr_bloc][b].interval.values()), 1)
        for curr_bloc in blocs.keys()
        for b in blocs.keys()
    )

    profile = gen.generate_profile(3)
    assert type(profile) is ScoreProfile


def test_CS_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    cs = CambridgeSampler.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(cs.pref_intervals_by_bloc[curr_bloc][b].interval.values()), 1)
        for curr_bloc in blocs.keys()
        for b in blocs.keys()
    )

    profile = cs.generate_profile(3)
    assert type(profile) is RankProfile

    # chekc that W,C bloc assignments work
    cs = CambridgeSampler.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
        W_bloc="R",
        C_bloc="D",
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(cs.pref_intervals_by_bloc[curr_bloc][b].interval.values()), 1)
        for curr_bloc in blocs.keys()
        for b in blocs.keys()
    )

    profile = cs.generate_profile(3)
    assert type(profile) is RankProfile
