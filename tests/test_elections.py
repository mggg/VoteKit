from votekit.election_types import compute_votes, remove_cand, fractional_transfer, STV
from votekit.cvr_loaders import rank_column_csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

test_profile = rank_column_csv(DATA_DIR / "ten_ballot.csv")
mn_profile = rank_column_csv(DATA_DIR / "mn_clean_ballots.csv")


def test_max_votes_toy():
    max_cand = "a"
    cands = test_profile.get_candidates()
    ballots = test_profile.get_ballots()
    results = compute_votes(cands, ballots)
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
    results = compute_votes(cands, ballots)
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
    assert {winner} == outcome.elected


def test_stv_winner_mn():
    irv = STV(mn_profile, fractional_transfer, 3)
    outcome = irv.run_election()
    winners = {"BETSY HODGES", "MARK ANDREW", "DON SAMUELS"}
    assert winners == outcome.elected


def test_runstep_seats_full_at_start():
    mock = STV(test_profile, fractional_transfer, 9)
    step, out = mock.run_step(test_profile)
    assert step == test_profile
