from votekit.election_types import (
    STV,
    Plurality,
    # Borda,
    # SequentialRCV,
)  # type:ignore
from votekit.cvr_loaders import rank_column_csv, blt  # type:ignore
from pathlib import Path
import pytest
from fractions import Fraction
from votekit.ballot import Ballot
from votekit.profile import PreferenceProfile
from votekit.utils import (
    fractional_transfer,
    random_transfer,
    remove_cand,
    compute_votes,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/csv"
BLT_DIR = BASE_DIR / "data/txt/"


test_profile = rank_column_csv(DATA_DIR / "ten_ballot.csv")
mn_profile = rank_column_csv(DATA_DIR / "mn_clean_ballots.csv")


def test_droop_default_parameter():

    pp, seats = blt(BLT_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats)

    droop_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / (4 + 1)) + 1

    assert election.threshold == droop_quota


def test_droop_inputed_parameter():

    pp, seats = blt(BLT_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="Droop")

    droop_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / (4 + 1)) + 1

    assert election.threshold == droop_quota


def test_quota_misspelled_parameter():

    pp, seats = blt(BLT_DIR / "edinburgh17-01_abridged.blt")

    with pytest.raises(ValueError):
        _ = STV(pp, fractional_transfer, seats=seats, quota="droops")


def test_hare_quota():

    pp, seats = blt(BLT_DIR / "edinburgh17-01_abridged.blt")

    election = STV(pp, fractional_transfer, seats=seats, quota="hare")

    hare_quota = int((8 + 14 + 1 + 13 + 1 + 1 + 2) / 4)

    assert election.threshold == hare_quota


def test_max_votes_toy():
    max_cand = "a"
    cands = test_profile.get_candidates()
    ballots = test_profile.get_ballots()
    results = {cand: votes for cand, votes in compute_votes(cands, ballots)}
    max_votes = [
        candidate
        for candidate, votes in results.items()
        if votes == max(results.values())
    ]
    assert results[max_cand] == 6
    assert max_votes[0] == max_cand


def test_min_votes_mn():
    min_cand = "JOHN CHARLES WILSON"
    cands = mn_profile.get_candidates()
    ballots = mn_profile.get_ballots()
    results = {cand: votes for cand, votes in compute_votes(cands, ballots)}
    max_votes = [
        candidate
        for candidate, votes in results.items()
        if votes == min(results.values())
    ]
    assert max_votes[0] == min_cand


def test_remove_cand_not_inplace():
    remove = "a"
    ballots = test_profile.get_ballots()
    new_ballots = remove_cand(remove, ballots)
    assert ballots != new_ballots


def test_remove_fake_cand():
    remove = "z"
    ballots = test_profile.get_ballots()
    new_ballots = remove_cand(remove, ballots)
    assert ballots == new_ballots


def test_remove_and_shift():
    remove = "a"
    ballots = test_profile.get_ballots()
    new_ballots = remove_cand(remove, ballots)
    for ballot in new_ballots:
        if len(ballot.ranking) == len(ballots[0].ranking):
            assert len(ballot.ranking) == len(ballots[0].ranking)


def test_irv_winner_mn():
    irv = STV(mn_profile, fractional_transfer, 1)
    outcome = irv.run_election()
    winner = "BETSY HODGES"
    assert [winner] == outcome.elected


def test_stv_winner_mn():
    irv = STV(mn_profile, fractional_transfer, 3)
    outcome = irv.run_election()
    winners = ["BETSY HODGES", "MARK ANDREW", "DON SAMUELS"]
    assert winners == outcome.get_all_winners()


# def test_runstep_seats_full_at_start():
#     mock = STV(test_profile, fractional_transfer, 9)
#     step = mock.__profile
#     assert step == test_profile


def test_runstep_update_inplace_mn():
    irv = STV(mn_profile, fractional_transfer, 1)
    out = irv.run_step()
    step = out.profile
    last = "JOHN CHARLES WILSON"
    assert step != mn_profile
    assert last not in step.get_candidates()
    assert last == out.get_all_eliminated()[0]


def test_rand_transfer_func_mock_data():
    winner = "A"
    ballots = [
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(2)),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1)),
    ]
    votes = {"A": 3}
    threshold = 1

    ballots_after_transfer = random_transfer(
        winner=winner, ballots=ballots, votes=votes, threshold=threshold
    )

    counts = compute_votes(candidates=["B", "C"], ballots=ballots_after_transfer)

    assert counts[0].votes == Fraction(1) or counts[0].votes == Fraction(2)


def test_rand_transfer_assert():
    winner = "A"
    ballots = [
        Ballot(ranking=({"A"}, {"C"}, {"B"}), weight=Fraction(1000)),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=Fraction(1000)),
    ]
    votes = {"A": 2000}
    threshold = 1000

    ballots_after_transfer = random_transfer(
        winner=winner, ballots=ballots, votes=votes, threshold=threshold
    )
    counts = compute_votes(candidates=["B", "C"], ballots=ballots_after_transfer)

    assert 400 < counts[0].votes < 600


def test_plurality():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(1), voters={"tom"}),
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1), voters={"andy"}),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voters={"andy"}),
        ]
    )
    election = Plurality(profile, seats=1)
    results = election.run_election()
    assert results.get_all_winners() == ["A"]


def test_plurality_multi_winner():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"D"}, {"B"}], weight=Fraction(1), voters={"tom"}),
            Ballot(ranking=[{"C"}, {"B"}, {"C"}], weight=Fraction(2), voters={"andy"}),
            Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(3), voters={"andy"}),
        ]
    )
    election = Plurality(profile, seats=3)
    results = election.run_election()
    assert results.get_all_winners() == ["A", "C", "D"]


# ---------------------------------------------------------------------------
#                         Borda Election Tests
# ---------------------------------------------------------------------------


# def test_toy_Borda():
#     known_winners = ["{'a'}", "{'d'}", "{'b'}", "{'c'}", "{'e'}"]
#     ballot_list = [
#         Ballot(ranking=[{"a"}, {"b"}, {"c"}, {"d"}, {"e"}], weight=Fraction(100)),
#         Ballot(ranking=[{"a"}, {"b"}], weight=Fraction(300)),
#         Ballot(ranking=[{"d"}], weight=Fraction(400)),
#     ]
#     toy_pp = PreferenceProfile(ballots=ballot_list)
#     borda_election = Borda(toy_pp, seats=5)
#     toy_winners = borda_election.run_borda_election().get_all_winners()
#     assert known_winners == toy_winners


# ---------------------------------------------------------------------------
#                        Sequential RCV Tests
# ---------------------------------------------------------------------------


# def test_toy_rcv():
#     """
#     example toy election taken from David McCune's code with known winners c and d
#     """
#     known_winners = ["c", "d"]
#     ballot_list = [
#         Ballot(ranking=[{"a"}, {"b"}], weight=Fraction(1799)),
#         Ballot(ranking=[{"a"}, {"b"}, {"c"}, {"d"}], weight=Fraction(1801)),
#         Ballot(ranking=[{"a"}, {"c"}, {"d"}], weight=Fraction(100)),
#         Ballot(ranking=[{"b"}, {"c"}, {"a"}, {"d"}], weight=Fraction(901)),
#         Ballot(ranking=[{"b"}, {"d"}], weight=Fraction(900)),
#         Ballot(ranking=[{"c"}, {"b"}, {"d"}, {"a"}], weight=Fraction(498)),
#         Ballot(ranking=[{"c"}, {"d"}, {"a"}], weight=Fraction(2000)),
#         Ballot(ranking=[{"d"}, {"b"}], weight=Fraction(1400)),
#         Ballot(ranking=[{"d"}, {"c"}], weight=Fraction(601)),
#     ]
#     toy_pp = PreferenceProfile(ballots=ballot_list)
#     seq_RCV = SequentialRCV(profile=toy_pp, seats=2)
#     toy_winners = seq_RCV.run_election().get_all_winners()
#     assert known_winners == toy_winners
