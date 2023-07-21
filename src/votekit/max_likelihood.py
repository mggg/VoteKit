from ballot import Ballot
from profile import PreferenceProfile
import ballot_graph as bg

di = {
    (2,): 52,
    (1,): 36,
    (1, 2): 23,
    (1, 2, 3): 39,
    (1, 3): 5,
    (1, 3, 2): 28,
    (2, 1): 16,
    (2, 1, 3): 38,
    (2, 3): 94,
    (2, 3, 1): 76,
    (3,): 42,
    (3, 1): 17,
    (3, 1, 2): 25,
    (3, 2): 89,
    (3, 2, 1): 81,
}

ballots_1 = []
for b in di.keys():
    rank = [set([i]) for i in b]
    bal = Ballot(ranking=rank, weight=di[b])
    ballots_1.append(bal)

election = PreferenceProfile(ballots=ballots_1)
vis = bg.BallotGraph(election)
