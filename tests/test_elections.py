from votekit.election_types import compute_votes
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
