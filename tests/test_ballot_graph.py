# import sys

# sys.path.insert(0, r"C:\Users\malav\OneDrive\Desktop\mggg\VoteKit\src\votekit")
# import pytest


from ballot import Ballot
from profile import PreferenceProfile
from ballot_graph import BallotGraph

# import networkx as nx
# import matplotlib.pyplot as plt


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


def setinitial(election):
    global vis
    vis = BallotGraph(election)
    return


def test_visualize():
    vis.visualize()


def test_heavy_nbs():
    nbs = vis.k_heaviest_neighborhoods(k=1, radius=2)
    vis.visualize(nbs)


setinitial(PreferenceProfile(ballots=ballots_1))
test_visualize()
test_heavy_nbs()
