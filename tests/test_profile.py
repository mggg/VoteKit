from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from fractions import Fraction


def test_unique_cands():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(1)),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=Fraction(1)),
        ]
    )
    cands = profile.get_candidates()
    unique_cands = {"A", "B", "C", "E"}
    assert unique_cands == set(cands)
    assert len(cands) == len(unique_cands)
