from votekit.visuals.mds import plot_MDS  # type: ignore
from votekit.ballot import Ballot  # type: ignore
from votekit.profile import PreferenceProfile  # type: ignore
from votekit.ballot_generator_ import BradleyTerry, PlackettLuce  # type: ignore
from votekit.metrics.distances import earth_mover_dist  # type: ignore
from fractions import Fraction

test_BL1 = [
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(10)),
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(10)),
    Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"a"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"b"}, {"c"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"a"}, {"b"}, {"c"}], weight=Fraction(0)),
]
test_BL2 = [
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(5)),
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(8)),
    Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"a"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=Fraction(9)),
    Ballot(ranking=[{"b"}, {"c"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"a"}, {"b"}, {"c"}], weight=Fraction(11)),
]

test_BL3 = [
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(5)),
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(10)),
    Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"a"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=Fraction(9)),
    Ballot(ranking=[{"b"}, {"c"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"a"}, {"b"}, {"c"}], weight=Fraction(3)),
]
test_BL4 = [
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(0)),
    Ballot(ranking=[{"b"}, {"a"}, {"c"}], weight=Fraction(8)),
    Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=Fraction(6)),
    Ballot(ranking=[{"c"}, {"a"}, {"b"}], weight=Fraction(0)),
    Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"b"}, {"c"}, {"a"}], weight=Fraction(0)),
    Ballot(ranking=[{"a"}, {"b"}, {"c"}], weight=Fraction(12)),
]

pp1 = PreferenceProfile(ballots=test_BL1)
pp2 = PreferenceProfile(ballots=test_BL2)
pp3 = PreferenceProfile(ballots=test_BL3)
pp4 = PreferenceProfile(ballots=test_BL4)


# Testing ballot generated code
voter_proportion_by_race = {"X": 0.5, "Y": 0.5}
interval1 = {
    "X": {"a": 0.40, "b": 0.40, "c": 0.2},
    "Y": {"a": 0.20, "b": 0.20, "c": 0.60},
}
interval2 = {
    "X": {"a": 0.10, "b": 0.30, "c": 0.60},
    "Y": {"a": 0.90, "b": 0.05, "c": 0.05},
}

pl1 = PlackettLuce(
    number_of_ballots=100,
    candidates=["a", "b", "c"],
    ballot_length=3,
    pref_interval_by_slate=interval1,
    slate_voter_prop=voter_proportion_by_race,
)

pl2 = PlackettLuce(
    number_of_ballots=100,
    candidates=["a", "b", "c"],
    ballot_length=3,
    pref_interval_by_slate=interval2,
    slate_voter_prop=voter_proportion_by_race,
)

bt1 = BradleyTerry(
    number_of_ballots=100,
    candidates=["a", "b", "c"],
    ballot_length=3,
    pref_interval_by_slate=interval1,
    slate_voter_prop=voter_proportion_by_race,
)

bt2 = BradleyTerry(
    number_of_ballots=100,
    candidates=["a", "b", "c"],
    ballot_length=3,
    pref_interval_by_slate=interval2,
    slate_voter_prop=voter_proportion_by_race,
)

pl1_list = []

pl2_list = []
bt1_list = []
bt2_list = []

for i in range(100):
    pl1_list.append(pl1.generate_profile())
    pl2_list.append(pl2.generate_profile())
    bt1_list.append(bt1.generate_profile())
    bt2_list.append(bt2.generate_profile())

election = {"red": pl1_list, "purple": bt1_list, "blue": pl2_list, "green": bt2_list}

plot_MDS(data=election, distance=earth_mover_dist, marker_size=10)
