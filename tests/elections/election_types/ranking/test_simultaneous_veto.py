from itertools import product

import pytest

from votekit.ballot import RankBallot
from votekit.elections import SimultaneousVeto
from votekit.pref_profile import RankProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_profile(raw: dict[tuple[str, ...], float], max_ranking_length=None) -> RankProfile:
    """Shorthand: {("A", "B", "C"): 3} -> RankBallot with that ranking and weight."""
    ballots = [RankBallot(ranking=ranking, weight=weight) for ranking, weight in raw.items()]
    if max_ranking_length is None:
        return RankProfile(ballots=ballots)
    else:
        return RankProfile(ballots=ballots, max_ranking_length=max_ranking_length)


def elected_set(election: SimultaneousVeto) -> frozenset[str]:
    """Flatten all elected candidates into a single frozenset."""
    return frozenset(c for s in election.get_elected() for c in s)


def eliminated_order(election: SimultaneousVeto) -> tuple[str, ...]:
    """Return eliminated candidates in order of elimination (first eliminated first)."""
    result = []
    for state in election.election_states:
        if state.eliminated != (frozenset(),):
            for s in state.eliminated:
                result.extend(sorted(s))
    return tuple(result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_profile():
    """3-candidate profile with a clear winner under first_place scoring."""
    return make_profile(
        {
            ("A", "B", "C"): 3.2,
            ("B", "C", "A"): 3,
            ("C", "A", "B"): 2,
        }
    )


@pytest.fixture
def oakland_profile():
    """2022 Oakland School Board District 4 Election."""
    return make_profile(
        {
            ("H",): 2327,
            ("H", "R", "M"): 2337,
            ("H", "M", "R"): 3563,
            ("R",): 3740,
            ("R", "H", "M"): 3095,
            ("R", "M", "H"): 3180,
            ("M",): 1846,
            ("M", "H", "R"): 4194,
            ("M", "R", "H"): 2150,
        }
    )


@pytest.fixture
def mutated_oakland_profile():
    """
    2022 Oakland School Board District 4 Election,
    edited so that H & M would be eliminated simultaneously,
    but H will have higher veto pressure (but more initial score).
    """
    return make_profile(
        {
            ("H",): 2327,
            ("H", "R", "M"): 2337,
            ("H", "M", "R"): 3563,
            ("R",): 3740,
            ("R", "H", "M"): 3095,
            ("R", "M", "H"): 3180 + 380.7446275946277,
            ("M",): 1846,
            ("M", "H", "R"): 4194,
            ("M", "R", "H"): 2150,
        }
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_m_zero(self, basic_profile):
        with pytest.raises(ValueError, match="m must be positive"):
            SimultaneousVeto(basic_profile, m=0)

    def test_m_negative(self, basic_profile):
        with pytest.raises(ValueError, match="m must be positive"):
            SimultaneousVeto(basic_profile, m=-1)

    def test_m_exceeds_candidates(self, basic_profile):
        with pytest.raises(ValueError, match="Not enough candidates"):
            SimultaneousVeto(basic_profile, m=10)

    def test_empty_profile(self):
        with pytest.raises(ValueError, match="Not enough candidates"):
            SimultaneousVeto(RankProfile(), m=1)

    def test_invalid_candidate_weights_string(self, basic_profile):
        with pytest.raises(ValueError, match="invalid input"):
            SimultaneousVeto(basic_profile, candidate_weights="banana")

    def test_invalid_candidate_weights_k_too_large(self, basic_profile):
        with pytest.raises(ValueError, match="not valid for a profile with max_ranking_length"):
            SimultaneousVeto(basic_profile, candidate_weights=42)

    def test_invalid_candidate_weights_k_too_small(self, basic_profile):
        with pytest.raises(ValueError, match="Invalid value for top-k scoring"):
            SimultaneousVeto(basic_profile, candidate_weights=0)

    def test_candidate_weights_dict_missing_candidate(self, basic_profile):
        with pytest.raises(ValueError, match="missing"):
            SimultaneousVeto(basic_profile, candidate_weights={"A": 1.0, "B": 1.0})

    def test_no_ranking(self):
        ballots = [RankBallot(ranking="A", weight=1.0), RankBallot(weight=1.0)]
        profile = RankProfile(ballots=ballots)
        with pytest.raises(ValueError, match="rankings"):
            SimultaneousVeto(profile)


# ---------------------------------------------------------------------------
# All parameter combinations (m=1, str-valued params)
# ---------------------------------------------------------------------------

CANDIDATE_WEIGHTS_STR = ["first_place", "uniform", "borda", "harmonic", 2]
TIEBREAKS = [
    "first_place",
    "random",
    "borda",
    "remaining_score",
    "veto_pressure",
    "lex",
]
SCORING_TIE_CONVENTIONS = ["high", "average", "low"]


class TestParameterCombinations:
    @pytest.mark.parametrize(
        "candidate_weights, tiebreak, scoring_tie_convention",
        list(product(CANDIDATE_WEIGHTS_STR, TIEBREAKS, SCORING_TIE_CONVENTIONS)),
    )
    def test_all_str_param_combos(
        self, basic_profile, candidate_weights, tiebreak, scoring_tie_convention
    ):
        """Election should run without error for all valid str parameter combinations."""
        election = SimultaneousVeto(
            basic_profile,
            m=1,
            candidate_weights=candidate_weights,
            tiebreak=tiebreak,
            scoring_tie_convention=scoring_tie_convention,
        )
        winners = elected_set(election)
        assert len(winners) == 1
        assert winners <= {"A", "B", "C"}


# ---------------------------------------------------------------------------
# candidate_weights
# ---------------------------------------------------------------------------


class TestCandidateWeights:
    def test_first_place(self, basic_profile):
        election = SimultaneousVeto(basic_profile, candidate_weights="first_place")
        # A has 5 first-place votes, B has 3, C has 2
        # C should be eliminated first, then B
        assert elected_set(election) == {"A"}

    def test_borda(self, basic_profile):
        election = SimultaneousVeto(basic_profile, candidate_weights="borda")
        assert election.initial_scores == {"A": 16.6, "B": 17.4, "C": 15.2}
        assert elected_set(election) == {"B"}

    def test_uniform(self, basic_profile):
        election = SimultaneousVeto(basic_profile, candidate_weights="uniform")
        assert election.initial_scores == {"A": 1.0, "B": 1.0, "C": 1.0}
        assert elected_set(election) == {"B"}

    def test_harmonic(self, basic_profile):
        election = SimultaneousVeto(basic_profile, candidate_weights="harmonic")
        expected_scores = {
            "A": 3.2 + 2 * 1 / 2 + 3 * 1 / 3,
            "B": 3 + 3.2 * 1 / 2 + 2 * 1 / 3,
            "C": 2 + 3 * 1 / 2 + 3.2 * 1 / 3,
        }
        assert all(election.initial_scores[c] - expected_scores[c] < 1e-5 for c in "ABC")

    def test_top2(self, basic_profile):
        election = SimultaneousVeto(basic_profile, candidate_weights=2)
        assert election.initial_scores == {"A": 5.2, "B": 6.2, "C": 5.0}

    def test_custom_dict(self, basic_profile):
        weights = {"A": 1.0, "B": 1.0, "C": 100.0}
        election = SimultaneousVeto(basic_profile, candidate_weights=weights)
        assert election.initial_scores == weights
        assert elected_set(election) == {"C"}


# ---------------------------------------------------------------------------
# Multi-winner (m=2)
# ---------------------------------------------------------------------------


class TestMultiWinner:
    def test_m_equals_2(self, basic_profile):
        election = SimultaneousVeto(basic_profile, m=2)
        winners = elected_set(election)
        assert len(winners) == 2

    def test_m_equals_num_candidates(self, basic_profile):
        """When m == n, everyone should be elected immediately."""
        election = SimultaneousVeto(basic_profile, m=3)
        winners = elected_set(election)
        assert winners == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# scoring_tie_convention
# ---------------------------------------------------------------------------


class TestScoringTieConvention:
    def _profile_with_first_place_tie(self):
        """A and B are tied for first place."""
        return make_profile(
            {
                ("AB",): 5,
                ("C",): 4.9,
                ("D",): 1,
            },
            max_ranking_length=2,
        )

    def _profile_with_tie(self):
        """A and B are tied for second place."""
        return make_profile(
            {
                (
                    "C",
                    "AB",
                ): 5,
            }
        )

    def test_high(self):
        profile = self._profile_with_first_place_tie()
        election = SimultaneousVeto(
            profile,
            m=1,
            candidate_weights="first_place",
            scoring_tie_convention="high",
            tiebreak="lex",
        )
        # "high": A and B each get 5, so A wins by lexicographic tiebreak
        assert elected_set(election) == {"A"}

    def test_average(self):
        profile = self._profile_with_first_place_tie()
        election = SimultaneousVeto(
            profile,
            candidate_weights="first_place",
            scoring_tie_convention="average",
        )
        # "average": A and B each get 2.5
        assert elected_set(election) == {"C"}

    def test_low(self):
        profile = self._profile_with_first_place_tie()
        election = SimultaneousVeto(
            profile,
            m=1,
            candidate_weights="first_place",
            scoring_tie_convention="low",
            tiebreak="lex",
        )
        assert election.initial_scores == {"A": 0, "B": 0, "C": 4.9, "D": 1}


# ---------------------------------------------------------------------------
# Partial ballots
# ---------------------------------------------------------------------------


class TestPartialBallots:
    def test_truncated_ballots(self):
        """Voters who only rank some candidates. Unlisted candidates tie for last."""
        profile = make_profile(
            {
                ("A",): 5,
                ("B",): 3,
                ("C", "A", "B"): 2,
            }
        )
        election = SimultaneousVeto(profile)
        # A has 5 fpv, B has 3, C has 2
        # unlisted candidates get vetoed first
        # TODO: fill in expected winner
        assert elected_set(election) == {"A"}

    def test_single_candidate_ballots(self):
        """Every ballot only ranks one candidate."""
        profile = make_profile(
            {
                ("A",): 5,
                ("B",): 3,
                ("C",): 2,
            }
        )
        election = SimultaneousVeto(profile)
        # A: 5 fpv, B: 3, C: 2
        # each ballot vetoes the two unlisted candidates equally
        assert elected_set(election) == {"A"}


# ---------------------------------------------------------------------------
# Zero-weight ballots
# ---------------------------------------------------------------------------


class TestZeroWeightBallots:
    def test_zero_weight_ballot_ignored(self):
        """A zero-weight ballot should not affect the outcome."""
        profile_without = make_profile(
            {
                ("A", "B", "C"): 5,
                ("B", "C", "A"): 3,
            }
        )
        profile_with = make_profile(
            {
                ("A", "B", "C"): 5,
                ("B", "C", "A"): 3,
                ("C", "A", "B"): 0,
            }
        )
        e1 = SimultaneousVeto(profile_without)
        e2 = SimultaneousVeto(profile_with)
        assert elected_set(e1) == elected_set(e2)


# ---------------------------------------------------------------------------
# Ballots with ties
# ---------------------------------------------------------------------------


class TestBallotTies:
    def test_tie_for_first_place_in_ballot(self):
        """Ballot where two candidates are tied at rank 1."""
        ballots = [
            RankBallot(ranking=(frozenset({"A", "B"}), frozenset({"C"})), weight=5),
            RankBallot(ranking=(frozenset({"C"}), frozenset({"A"}), frozenset({"B"})), weight=3),
        ]
        profile = RankProfile(ballots=ballots)
        election = SimultaneousVeto(profile, candidate_weights="first_place")
        # A and B split the 5 first-place votes -> 2.5 each (average convention)
        # C gets 3 first-place votes
        assert len(elected_set(election)) == 1

    def test_tie_in_middle_position(self):
        """Ballot where two candidates are tied at rank 2."""
        ballots = [
            RankBallot(ranking=(frozenset({"A"}), frozenset({"B", "C"})), weight=5),
            RankBallot(ranking=(frozenset({"B"}), frozenset({"C"}), frozenset({"A"})), weight=3),
            RankBallot(ranking=(frozenset({"C"}), frozenset({"A"}), frozenset({"B"})), weight=2),
        ]
        profile = RankProfile(ballots=ballots)
        election = SimultaneousVeto(profile)
        # For the first ballot, B and C are tied for last -> veto split between them
        assert len(elected_set(election)) == 1


# ---------------------------------------------------------------------------
# Candidate with no first-place votes
# ---------------------------------------------------------------------------


class TestNoFirstPlaceVotes:
    def test_candidate_with_no_fpv(self):
        """D never appears in first place but is listed on ballots."""
        profile = make_profile(
            {
                ("A", "B", "D", "C"): 5,
                ("B", "C", "D", "A"): 3,
                ("C", "A", "D", "B"): 2,
            }
        )
        election = SimultaneousVeto(profile, candidate_weights="first_place")
        elim = eliminated_order(election)
        assert elim == ("C", "D", "B")

    def test_candidate_with_no_fpv_uniform(self):
        """
        Under uniform weights, having no first-place votes doesn't matter.
        D is a Condorcet-loser but wins by having no last-place votes.
        """
        profile = make_profile(
            {
                ("A", "B", "D", "C"): 1,
                ("B", "C", "D", "A"): 1,
                ("C", "A", "D", "B"): 1,
            }
        )
        election = SimultaneousVeto(profile, candidate_weights="uniform")
        assert elected_set(election) == {"D"}


# ---------------------------------------------------------------------------
# Candidate with no initial veto pressure
# ---------------------------------------------------------------------------


class TestNoInitialVetoPressure:
    def test_universally_first_ranked(self):
        """A candidate ranked first by everyone receives no veto pressure."""
        profile = make_profile(
            {
                ("A", "B", "C"): 5,
                ("A", "C", "B"): 3,
            }
        )
        election = SimultaneousVeto(profile, candidate_weights="first_place")
        assert elected_set(election) == {"A"}

    def test_no_pressure_with_uniform(self):
        """Under uniform weights, a candidate ranked first by everyone still wins."""
        profile = make_profile(
            {
                ("A", "B", "C"): 5,
                ("A", "C", "B"): 3,
            }
        )
        election = SimultaneousVeto(profile, candidate_weights="uniform")
        assert elected_set(election) == {"A"}


# ---------------------------------------------------------------------------
# Simultaneous eliminations (tiebreak tests)
# ---------------------------------------------------------------------------


class TestSimultaneousEliminations:
    def _symmetric_profile(self):
        """B and C have identical scores and veto pressure."""
        return make_profile(
            {
                ("A",): 10,
                ("B", "C", "A"): 3,
                ("C", "B", "A"): 3,
            }
        )

    def test_tie_for_first_eliminated(self):
        """Two candidates tie for first eliminated; tiebreak decides which goes first."""
        # B and C both have 3 fpv and symmetric veto pressure
        profile = self._symmetric_profile()
        election = SimultaneousVeto(
            profile,
            candidate_weights="first_place",
            tiebreak="lex",
        )
        winners = elected_set(election)
        assert len(winners) == 1

        # there should be a tiebreak recorded in the first elimination round
        tiebreak_states = [s for s in election.election_states if s.tiebreaks]
        assert len(tiebreak_states) >= 1

        # the tied set should be {B, C}
        first_tiebreak = tiebreak_states[0].tiebreaks
        tied_candidates = list(first_tiebreak.keys())[0]
        assert tied_candidates == frozenset({"B", "C"})

    def test_tie_for_last_eliminated_decides_election(self):
        """Two candidates tie for the final elimination, so the tiebreak decides the winner."""
        # A, B have equal scores
        profile = make_profile(
            {
                ("A", "B"): 5,
                ("B", "A"): 5,
            }
        )
        election = SimultaneousVeto(
            profile,
            m=1,
            candidate_weights="first_place",
            tiebreak="lex",
        )
        winners = elected_set(election)
        assert len(winners) == 1

        # the deciding tiebreak should involve {A, B}
        tiebreak_states = [s for s in election.election_states if s.tiebreaks]
        assert len(tiebreak_states) >= 1

        last_tiebreak = tiebreak_states[-1].tiebreaks
        tied_candidates = list(last_tiebreak.keys())[0]
        assert tied_candidates == frozenset({"A", "B"})

        # "lex" tiebreak: alphabetical ordering -> A ranked higher -> B eliminated
        resolution = last_tiebreak[tied_candidates]
        assert resolution == (frozenset({"A"}), frozenset({"B"}))
        assert winners == {"A"}

    def test_veto_pressure_tiebreak(self, mutated_oakland_profile):
        """Veto pressure tiebreak eliminates the candidate with higher veto pressure."""
        election = SimultaneousVeto(
            mutated_oakland_profile,
            candidate_weights="first_place",
            tiebreak="veto_pressure",
        )
        assert eliminated_order(election)[0] == "H"

    def test_remaining_score_tiebreak(self, mutated_oakland_profile):
        """remaining_score tiebreak keeps the candidate with higher pre-step score."""
        election = SimultaneousVeto(
            mutated_oakland_profile,
            candidate_weights="first_place",
            tiebreak="remaining_score",
        )
        assert eliminated_order(election)[0] == "M"

    def test_first_place_tiebreak(self, mutated_oakland_profile):
        """remaining_score tiebreak keeps the candidate with higher pre-step score."""
        election = SimultaneousVeto(
            mutated_oakland_profile,
            candidate_weights="first_place",
            tiebreak="first_place",
        )
        assert eliminated_order(election)[0] == "M"

    def test_three_way_tie_with_m(self):
        """With m=1, a 3-way tie should be broken via tiebreak."""
        profile = make_profile(
            {
                ("A", "C", "B"): 1,
                ("B", "C", "A"): 1,
            }
        )
        election = SimultaneousVeto(
            profile, m=1, candidate_weights="first_place", tiebreak="first_place"
        )
        # this should be a 3-way tie
        assert frozenset({"A", "B", "C"}) in election.election_states[-1].tiebreaks.keys()
        assert len(elected_set(election)) == 1

    def test_three_way_tie(self):
        """With return_all_tied_winners=True, a 3-way tie should return all tied candidates."""
        profile = make_profile(
            {
                ("A", "C", "B"): 1,
                ("B", "C", "A"): 1,
            }
        )
        election = SimultaneousVeto(
            profile,
            candidate_weights="first_place",
            tiebreak="first_place",
            return_all_tied_winners=True,
        )
        assert elected_set(election) == {"A", "B", "C"}
        assert election.election_states[-1].tiebreaks == {}


# ---------------------------------------------------------------------------
# Oakland election (integration / regression test)
# ---------------------------------------------------------------------------


class TestOakland:
    def test_oakland_first_place(self, oakland_profile):
        election = SimultaneousVeto(
            oakland_profile,
            m=1,
            candidate_weights="first_place",
            tiebreak="veto_pressure",
        )
        assert elected_set(election) == {"R"}

    def test_oakland_borda(self, oakland_profile):
        election = SimultaneousVeto(
            oakland_profile,
            m=1,
            candidate_weights="borda",
            tiebreak="veto_pressure",
        )
        assert elected_set(election) == {"H"}


# ---------------------------------------------------------------------------
# get_profile override
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_get_profile_round_0(self, basic_profile):
        election = SimultaneousVeto(basic_profile)
        p0 = election.get_profile(0)
        assert p0.total_ballot_wt == basic_profile.total_ballot_wt

    def test_get_profile_negative_index(self, basic_profile):
        election = SimultaneousVeto(basic_profile)
        p_last = election.get_profile(-1)
        # final profile should be empty (all candidates elected or eliminated)
        assert isinstance(p_last, RankProfile)

    def test_get_profile_out_of_range(self, basic_profile):
        election = SimultaneousVeto(basic_profile)
        with pytest.raises(IndexError):
            election.get_profile(999)

    def test_get_profile(self, basic_profile):
        """Calling get_profile multiple times should return consistent results."""
        election = SimultaneousVeto(basic_profile)
        state0 = election._sv_states[1]
        _ = election.get_profile(0)
        _ = election.get_profile(1)
        state0_after_get_prof = election._sv_states[1]
        assert state0 == state0_after_get_prof
