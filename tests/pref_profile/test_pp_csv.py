from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile

filepath= "tests/data/csv"
def test_csv_bijection_rankings():
    profile_rankings = PreferenceProfile(ballots=(Ballot(ranking = ({"A", "B"}, frozenset(), {"C"}), id="u", voter_set={"Chris", "Peter"}, weight = 3/2),
                                     Ballot(ranking = ({"A", "B"}, frozenset(), {"C"}), id="v", voter_set={"Moon"}, weight = 1/2),
                                     Ballot(ranking = ({"A"}, {"B"},), id="yes"),
                                     Ballot(ranking = ({"A"}, {"B"},), id = "hi"),
                                     Ballot(ranking = ({"A"}, {"B"},), id = "X29"))*5,

                                    max_ballot_length=3,
                                    candidates=["A", "B", "C", "D", "E"])

    
    profile_rankings.to_csv(f"{filepath}/test_csv_pp_rankings.csv")
    read_profile =  PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_rankings.csv")
    assert profile_rankings == read_profile


def test_csv_bijection_scores():
    profile_scores = PreferenceProfile(ballots=(
                                     Ballot(scores = {"A": 2, "B": 4, "D": 1}, ),
                                     Ballot(scores = {"A": 2, "B": 4, "D": 1}),
                                     Ballot(scores = {"A": 2, "B": 4, "C": 1}, id="yes"),
                                     Ballot(scores = {"A": 2, "B": 4, "C": 1}, id = "hi"),
                                     Ballot(scores = {"A": 5, "B": 4, "C": 1}, id = "X29"))*5,
                                    candidates=["A", "B", "C", "D", "E"])
    
    profile_scores.to_csv(f"{filepath}/test_csv_pp_scores.csv")
    read_profile =  PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_scores.csv")
    assert profile_scores == read_profile
    
     

def test_csv_bijection_mixed():
    profile_mixed = PreferenceProfile(ballots=(Ballot(ranking = ({"A", "B"}, frozenset(), {"C"}), id="u", voter_set={"Chris", "Peter"}, weight = 3/2),
                                     Ballot(ranking = ({"A", "B"}, frozenset(), {"C"}), id="v", voter_set={"Moon"}, weight = 1/2),
                                     Ballot(scores = {"A": 2, "B": 4, "D": 1}),
                                     Ballot(scores = {"A": 2, "B": 4, "D": 1}),
                                     Ballot(ranking = ({"A"}, {"B"},), scores = {"A": 2, "B": 4, "C": 1}, id="yes"),
                                     Ballot(ranking = ({"A"}, {"B"},), scores = {"A": 2, "B": 4, "C": 1}, id = "hi"),
                                     Ballot(ranking = ({"A"}, {"B"},), scores = {"A": 5, "B": 4, "C": 1}, id = "X29"))*5,

                                    max_ballot_length=3,
                                    candidates=["A", "B", "C", "D", "E"])
    
    profile_mixed.to_csv(f"{filepath}/test_csv_pp_mixed.csv")
    read_profile =  PreferenceProfile.from_csv(f"{filepath}/test_csv_pp_mixed.csv")
    assert profile_mixed == read_profile 