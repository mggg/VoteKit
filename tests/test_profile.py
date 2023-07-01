from unnamed_rcv_thing.profile import PreferenceProfile
from unnamed_rcv_thing.ballot import Ballot


def test_unique_cands():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=1),
            Ballot(ranking=[{"B"}, {"C"}, {"E"}], weight=1),
        ]
    )
    cands = profile.get_candidates()
    unique_cands = {"A", "B", "C", "E"}
    assert unique_cands == set(cands)
    assert len(cands) == len(unique_cands)
