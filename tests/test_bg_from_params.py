import math

from votekit.ballot_generator import (
    name_PlackettLuce,
    name_BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    slate_PlackettLuce,
    slate_BradleyTerry,
    name_Cumulative,
)
from votekit.pref_profile import PreferenceProfile


def test_NPL_fron_params():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}

    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    pl = name_PlackettLuce.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=blocs,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(pl.pref_interval_by_bloc[curr_bloc].interval.values()), 1)
        for curr_bloc in blocs.keys()
    )

    profile = pl.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_NBT_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    gen = name_BradleyTerry.from_params(
        slate_to_candidates=slate_to_cands,
        bloc_voter_prop=blocs,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(gen.pref_interval_by_bloc[curr_bloc].interval.values()), 1)
        for curr_bloc in blocs.keys()
    )

    profile = gen.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_AC_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    ac = AlternatingCrossover.from_params(
        bloc_voter_prop=blocs,
        alphas=alphas,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion_parameters,
    )

    # check if intervals add up to one
    assert all(
        math.isclose(sum(ac.pref_intervals_by_bloc[curr_bloc][b].interval.values()), 1)
        for curr_bloc in blocs.keys()
        for b in blocs.keys()
    )

    profile = ac.generate_profile(3)
    assert type(profile) is PreferenceProfile


def test_SPL_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    gen = slate_PlackettLuce.from_params(
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
    assert type(profile) is PreferenceProfile


def test_SPL_from_params_zero_support():
    """
    Ensures that if a candidate has zero support from small alpha, SPL handles it.
    """

    blocs = {"R": 0.6, "D": 0.4}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.0001}}
    cohesion_parameters = {"R": {"R": 0.5, "D": 0.5}, "D": {"D": 0.4, "R": 0.6}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    gen = slate_PlackettLuce.from_params(
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
    assert type(profile) is PreferenceProfile


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
    assert type(profile) is PreferenceProfile


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
    assert type(profile) is PreferenceProfile


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
    assert type(profile) is PreferenceProfile


def test_interval_sum_from_params():
    blocs = {"R": 0.6, "D": 0.4}
    cohesion = {"R": {"R": 0.7, "D": 0.3}, "D": {"D": 0.6, "R": 0.4}}
    alphas = {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}}
    slate_to_cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    npl = name_PlackettLuce.from_params(
        bloc_voter_prop=blocs,
        slate_to_candidates=slate_to_cands,
        cohesion_parameters=cohesion,
        alphas=alphas,
    )
    for curr_b in npl.blocs:
        for b in npl.blocs:
            if not math.isclose(
                sum(npl.pref_intervals_by_bloc[curr_b][b].interval.values()), 1
            ):
                assert False
    assert True
