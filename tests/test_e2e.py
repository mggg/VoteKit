from pathlib import Path
from fractions import Fraction

from votekit.cvr_loaders import load_scottish
import votekit.cleaning as clean
from votekit.election_state import ElectionState
import votekit.ballot_generator as bg
import votekit.elections.election_types as elections
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.elections.transfers import fractional_transfer
from votekit.pref_interval import PreferenceInterval

# TODO:
# need to do one with visualizations,
# need to test other elections,
# need to test cleaning methods,
# need to add cleaning methods (ballot truncation for ex)
# need to test ballot generation models


def test_load_clean_completion():
    """simple example of what a "full" use would look like"""

    # load CVR -> PP representation
    BASE_DIR = Path(__file__).resolve().parent
    CSV_DIR = BASE_DIR / "data/csv/"

    pp, seats, cand_list, cand_to_party, ward = load_scottish(
        CSV_DIR / "scot_wardy_mc_ward.csv"
    )

    # apply rules to get new PP
    cleaned_pp = clean.remove_noncands(pp, ["Paul"])

    # write intermediate output for inspection
    # cleaned_pp.save("cleaned.cvr")

    # run election using a configured RCV step object
    election_borda = elections.Borda(cleaned_pp, 1, score_vector=None)

    outcome_borda = election_borda.run_election()

    assert isinstance(outcome_borda, ElectionState)

    # plot_results(outcome)


def test_generate_election_completion():
    number_of_ballots = 100
    candidates = ["W1", "W2", "C1", "C2"]
    slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
    cohesion_parameters = {"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.6, "W": 0.4}}
    pref_interval_by_bloc = {
        "W": {
            "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
            "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
        },
        "C": {
            "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
            "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
        },
    }
    bloc_voter_prop = {"W": 0.7, "C": 0.3}

    DATA_DIR = "src/votekit/data/"
    path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    ballot_model = bg.CambridgeSampler(
        candidates=candidates,
        pref_intervals_by_bloc=pref_interval_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        path=path,
        cohesion_parameters=cohesion_parameters,
        slate_to_candidates=slate_to_candidate,
    )

    pp = ballot_model.generate_profile(number_of_ballots=number_of_ballots)

    election_borda = elections.Borda(pp, 1, score_vector=None)

    outcome_borda = election_borda.run_election()

    assert isinstance(outcome_borda, ElectionState)


def make_ballot(ranking, weight):
    ballot_rank = []
    for cand in ranking:
        ballot_rank.append({cand})

    return Ballot(ranking=ballot_rank, weight=Fraction(weight))


def test_generate_election_diff_res():
    b1 = make_ballot(ranking=["A", "D", "E", "C", "B"], weight=18)
    b2 = make_ballot(ranking=["B", "E", "D", "C", "A"], weight=12)
    b3 = make_ballot(ranking=["C", "B", "E", "D", "A"], weight=10)
    b4 = make_ballot(ranking=["D", "C", "E", "B", "A"], weight=4)
    b5 = make_ballot(ranking=["E", "B", "D", "C", "A"], weight=4)
    b6 = make_ballot(ranking=["E", "C", "D", "B", "A"], weight=2)
    pp = PreferenceProfile(ballots=[b1, b2, b3, b4, b5, b6])

    election_borda = elections.Borda(pp, 1, score_vector=None)
    election_irv = elections.STV(pp, fractional_transfer, 1)
    election_plurality = elections.Plurality(pp, seats=1, ballot_ties=False)
    election_seq = elections.SequentialRCV(pp, seats=1)
    # election_sntv = elections.SNTV(pp, seats=1)

    outcome_borda = election_borda.run_election().winners()
    outcome_irv = election_irv.run_election().winners()
    outcome_plurality = election_plurality.run_election().winners()
    outcome_seq = election_seq.run_election().winners()
    # outcome_sntv = election_sntv.run_election().get_all_winners()

    print(outcome_borda)
    print(outcome_irv)
    print(outcome_plurality)
    print(outcome_seq)

    assert outcome_borda != outcome_irv != outcome_plurality != outcome_seq
