import pytest
import numpy as np
from votekit.learn import PlackettLuceMixture
from votekit.pref_profile import RankProfile
from votekit.ballot import RankBallot
from votekit.ballot_generator import (
    name_pl_profile_generator,
    BlocSlateConfig,
)
from votekit.pref_interval import PreferenceInterval


def _make_two_bloc_profile(n_voters=2000, seed=108):
    """Generate a profile from two well-separated blocs."""
    np.random.seed(seed)
    config = BlocSlateConfig(
        n_voters=n_voters,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"X": 0.6, "Y": 0.4},
        preference_mapping={
            "X": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "Y": {
                "W": PreferenceInterval({"W1": 0.1, "W2": 0.1}),
                "C": PreferenceInterval({"C1": 0.4, "C2": 0.4}),
            },
        },
        cohesion_mapping={"X": {"W": 0.8, "C": 0.2}, "Y": {"C": 0.9, "W": 0.1}},
    )
    return name_pl_profile_generator(config)


# Initialization tests


def test_init_defaults():
    plm = PlackettLuceMixture()
    assert plm.n_components == 2
    assert plm.max_iter == 500
    assert plm.tol == 1e-6
    assert plm.random_state is None
    assert plm.support_params_ is None


def test_init_custom_params():
    plm = PlackettLuceMixture(n_components=3, max_iter=100, tol=1e-4, random_state=99)
    assert plm.n_components == 3
    assert plm.max_iter == 100
    assert plm.tol == 1e-4
    assert plm.random_state == 99


def test_init_invalid_n_components():
    with pytest.raises(ValueError):
        PlackettLuceMixture(n_components=0)


# Fitting tests


def test_fit_learns_params():
    profile = _make_two_bloc_profile(n_voters=200, seed=108)
    plm = PlackettLuceMixture(n_components=2, random_state=528, max_iter=50)
    result = plm.fit(profile)
    assert result is plm
    assert plm.support_params_ is not None
    assert plm.mixing_weights_ is not None
    assert plm.log_likelihood_ is not None
    assert plm.converged_ is not None
    assert plm.num_iterations_ is not None
    assert plm.responsibilities_ is not None
    assert plm.candidate_names_ is not None
    assert plm._support_params_array_ is not None


def test_fit_result_shapes():
    profile = _make_two_bloc_profile(n_voters=300, seed=4)
    n_components = 2
    plm = PlackettLuceMixture(n_components=n_components, random_state=8, max_iter=100)
    plm.fit(profile)
    n_cands = len(profile.candidates)

    assert plm.mixing_weights_.shape == (n_components,)
    assert plm._support_params_array_.shape == (n_components, n_cands)
    assert plm.responsibilities_.shape[1] == n_components
    assert len(plm.support_params_) == n_cands
    for cand_arr in plm.support_params_.values():
        assert cand_arr.shape == (n_components,)


def test_fit_mixing_weights_sum_to_one():
    profile = _make_two_bloc_profile(n_voters=400, seed=8)
    plm = PlackettLuceMixture(n_components=2, random_state=15, max_iter=100)
    plm.fit(profile)
    assert abs(plm.mixing_weights_.sum() - 1.0) < 1e-10


def test_fit_support_params_sum_to_one():
    profile = _make_two_bloc_profile(n_voters=500, seed=15)
    plm = PlackettLuceMixture(n_components=2, random_state=16, max_iter=100)
    plm.fit(profile)
    row_sums = plm._support_params_array_.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_fit_responsibilities_sum_to_one():
    profile = _make_two_bloc_profile(n_voters=200, seed=16)
    plm = PlackettLuceMixture(n_components=2, random_state=23, max_iter=100)
    plm.fit(profile)
    row_sums = plm.responsibilities_.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_fit_candidate_names_match_profile():
    profile = _make_two_bloc_profile(n_voters=300, seed=23)
    plm = PlackettLuceMixture(n_components=2, random_state=42, max_iter=50)
    plm.fit(profile)
    assert plm.candidate_names_ == profile.candidates


def test_fit_single_component():
    profile = _make_two_bloc_profile(n_voters=400, seed=42)
    plm = PlackettLuceMixture(n_components=1, random_state=4, max_iter=50)
    plm.fit(profile)
    assert plm.mixing_weights_.shape == (1,)
    assert abs(plm.mixing_weights_[0] - 1.0) < 1e-10
    assert plm._support_params_array_.shape == (1, len(profile.candidates))


def test_fit_converges():
    profile = _make_two_bloc_profile(n_voters=500, seed=108)
    plm = PlackettLuceMixture(n_components=2, random_state=8, max_iter=500, tol=1e-6)
    plm.fit(profile)
    assert plm.converged_ is True


def test_fit_reproducible():
    profile = _make_two_bloc_profile(n_voters=200, seed=528)
    plm1 = PlackettLuceMixture(n_components=2, random_state=108, max_iter=100)
    plm1.fit(profile)
    plm2 = PlackettLuceMixture(n_components=2, random_state=108, max_iter=100)
    plm2.fit(profile)
    np.testing.assert_array_equal(
        plm1._support_params_array_, plm2._support_params_array_
    )
    np.testing.assert_array_equal(plm1.mixing_weights_, plm2.mixing_weights_)
    assert plm1.log_likelihood_ == plm2.log_likelihood_


# Tests for name-Plackett-Luce parameters


def test_name_pl_parameters_default_slate():
    profile = _make_two_bloc_profile(n_voters=400, seed=108)
    plm = PlackettLuceMixture(n_components=2, random_state=16, max_iter=100)
    plm.fit(profile)
    params = plm.name_pl_parameters()
    assert "All" in params["slate_to_candidates"]
    assert set(params["slate_to_candidates"]["All"]) == set(profile.candidates)


def test_name_pl_parameters_roundtrip():
    profile = _make_two_bloc_profile(n_voters=400, seed=108)
    plm = PlackettLuceMixture(n_components=2, random_state=8, max_iter=200)
    plm.fit(profile)
    params = plm.name_pl_parameters()
    config2 = BlocSlateConfig(n_voters=1000, **params)
    profile2 = name_pl_profile_generator(config2)
    assert isinstance(profile2, RankProfile)
    assert profile2.total_ballot_wt == 1000


# ---------------------------------------------------------------------------
# Correctness tests — tiny hand-crafted elections
# ---------------------------------------------------------------------------


def test_fit_partial_rankings_symmetric():
    ballots = [
        RankBallot(ranking=[{"A"}, {"B"}], weight=10),
        RankBallot(ranking=[{"B"}, {"A"}], weight=10),
        RankBallot(ranking=[{"A"}], weight=5),
        RankBallot(ranking=[{"B"}], weight=5),
    ]
    profile = RankProfile(ballots=ballots, candidates=("A", "B", "C"))
    plm = PlackettLuceMixture(n_components=1, random_state=15, max_iter=100)
    plm.fit(profile)
    assert plm._support_params_array_.shape == (1, 3)
    assert abs(plm._support_params_array_[0].sum() - 1.0) < 1e-10
    # assert plm.support_params


def test_pure_bloc_cohesion_ones_and_zeros():
    """When each bloc only votes for its own slate, cohesion should be ~1/0.

    Two blocs, two slates: bloc X always ranks W1>W2 (never mentions C
    candidates), bloc Y always ranks C1>C2 (never mentions W candidates).
    The learned support for each component should assign ~all mass to one
    slate and ~none to the other.
    """
    ballots = [
        # Bloc X voters: only rank W candidates
        RankBallot(ranking=[{"W1"}, {"W2"}], weight=3.5),
        RankBallot(ranking=[{"W1"}], weight=5.8),
        # Bloc Y voters: only rank C candidates
        RankBallot(ranking=[{"C1"}, {"C2"}], weight=5),
        RankBallot(ranking=[{"C2"}], weight=7.1),
    ]
    profile = RankProfile(ballots=ballots, candidates=("W1", "W2", "C1", "C2"))

    plm = PlackettLuceMixture(n_components=2, random_state=16, max_iter=500)
    plm.fit(profile)

    # Each component should concentrate support on its own candidates.
    # Component 0 or 1 should have high support for W1,W2 and low for C1,C2
    # (or vice versa).
    s = plm.support_params_
    w_support_0 = float(s["W1"][0]) + float(s["W2"][0])
    w_support_1 = float(s["W1"][1]) + float(s["W2"][1])

    if w_support_0 > w_support_1:
        w_comp, c_comp = 0, 1
    else:
        w_comp, c_comp = 1, 0

    assert float(s["W1"][w_comp]) + float(s["W2"][w_comp]) > 0.95
    assert float(s["C1"][c_comp]) + float(s["C2"][c_comp]) > 0.95


def test_symmetric_two_candidate_profile():
    """When A>B and B>A appear equally, learned parameters should be symmetric.

    One component, one slate, two candidates. Profile has equal weight on
    both orderings, so the MLE support parameters must be equal.
    """
    ballots = [
        RankBallot(ranking=[{"A"}, {"B"}], weight=10),
        RankBallot(ranking=[{"B"}, {"A"}], weight=10),
    ]
    profile = RankProfile(ballots=ballots, candidates=("A", "B"))

    plm = PlackettLuceMixture(n_components=1, random_state=0, max_iter=500)
    plm.fit(profile)

    support = plm._support_params_array_[0]  # shape (2,)
    np.testing.assert_allclose(support[0], support[1], atol=1e-6)
    np.testing.assert_allclose(support[0], 0.5, atol=1e-6)


def test_pure_bloc_three_candidates_per_slate():
    """Same idea as the ones-and-zeros test but with 3 candidates per slate."""
    ballots = [
        RankBallot(ranking=[{"A1"}, {"A2"}, {"A3"}], weight=4),
        RankBallot(ranking=[{"B1"}, {"B2"}, {"B3"}], weight=6),
    ]
    profile = RankProfile(
        ballots=ballots,
        candidates=("A1", "A2", "A3", "B1", "B2", "B3"),
    )

    plm = PlackettLuceMixture(n_components=2, random_state=7, max_iter=500)
    plm.fit(profile)

    # Each component should concentrate support on its own candidates.
    s = plm.support_params_
    a_support_0 = sum(float(s[c][0]) for c in ["A1", "A2", "A3"])
    a_support_1 = sum(float(s[c][1]) for c in ["A1", "A2", "A3"])

    if a_support_0 > a_support_1:
        a_comp, b_comp = 0, 1
    else:
        a_comp, b_comp = 1, 0

    assert sum(float(s[c][a_comp]) for c in ["A1", "A2", "A3"]) > 0.95
    assert sum(float(s[c][b_comp]) for c in ["B1", "B2", "B3"]) > 0.95

    # Mixing weights should roughly reflect the ballot weight ratio (4:6)
    assert abs(float(plm.mixing_weights_[a_comp]) - 0.4) < 0.15
    assert abs(float(plm.mixing_weights_[b_comp]) - 0.6) < 0.15


def test_recovery_from_generated_profile():
    """Generate 100k ballots from known parameters and check recovery.

    We use name_pl_profile_generator with known cohesion and preference
    parameters, then fit a 2-component model and verify the learned
    parameters are close to the ground truth (up to component relabelling).
    """
    true_cohesion = {"X": {"W": 0.9, "C": 0.1}, "Y": {"W": 0.15, "C": 0.85}}
    true_prefs = {
        "X": {
            "W": PreferenceInterval({"W1": 0.7, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.6, "C2": 0.4}),
        },
        "Y": {
            "W": PreferenceInterval({"W1": 0.5, "W2": 0.5}),
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.7}),
        },
    }
    true_bloc_props = {"X": 0.55, "Y": 0.45}

    config = BlocSlateConfig(
        n_voters=100_000,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions=true_bloc_props,
        preference_mapping=true_prefs,
        cohesion_mapping=true_cohesion,
    )
    np.random.seed(12345)
    profile = name_pl_profile_generator(config)

    plm = PlackettLuceMixture(n_components=2, random_state=23, max_iter=500)
    plm.fit(profile)

    # Match learned components to true blocs by W-support
    s = plm.support_params_
    w_support_0 = s["W1"][0] + s["W2"][0]
    w_support_1 = s["W1"][1] + s["W2"][1]
    if w_support_0 > w_support_1:
        x_comp, y_comp = 0, 1
    else:
        x_comp, y_comp = 1, 0

    # True X support: W1=0.9*0.7=0.63, W2=0.9*0.3=0.27, C1=0.1*0.6=0.06, C2=0.1*0.4=0.04
    # Check W-cohesion recovery (within 0.05)
    w_cohesion_x = s["W1"][x_comp] + s["W2"][x_comp]
    assert abs(w_cohesion_x - 0.9) < 0.01
    c_cohesion_y = s["C1"][y_comp] + s["C2"][y_comp]
    assert abs(c_cohesion_y - 0.85) < 0.01

    # Check bloc proportions (within 0.05)
    assert abs(plm.mixing_weights_[x_comp] - 0.55) < 0.01
    assert abs(plm.mixing_weights_[y_comp] - 0.45) < 0.01

    # Check within-W preference recovery for X
    w1_pref_x = s["W1"][x_comp] / w_cohesion_x
    assert abs(w1_pref_x - 0.7) < 0.01
    # Check within-C preference recovery for Y
    c1_pref_y = s["C1"][y_comp] / c_cohesion_y
    assert abs(c1_pref_y - 0.3) < 0.01
