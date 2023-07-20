#import sys
#sys.path.insert(0, r"C:\Users\malav\OneDrive\Desktop\mggg\VoteKit\src\votekit")
import pytest

from outcome_model import Outcome
from profile import PreferenceProfile
from ballot import Ballot

##TODO: this is from Scottish 3-cand ward_03 data, pref should be tested later after we integrate Outcome with election_types
di = {(2,): 52, (1,): 36, (1, 2): 23, (1, 2, 3): 39, (1, 3): 5, (1, 3, 2): 28, (2, 1): 16, (2, 1, 3): 38, (2, 3): 94, (2, 3, 1): 76, (3,): 42, (3, 1): 17, (3, 1, 2): 25, (3, 2): 89, (3, 2, 1): 81}

ballots_1 = []
for b in di.keys():
    rank = [set([i]) for i in b]
    bal = Ballot(ranking = rank, weight = di[b])
    ballots_1.append(bal)

    
#pref = PreferenceProfile(ballots = ballots_1)

b1 = Ballot(ranking = [{'A'}, {'B'}, {'C'}], weight =  250)
b2 = Ballot(ranking = [{'B'}, {'A'}, {'C'}], weight = 200)
b3 = Ballot(ranking = [{'C'}, {'B'}, {'A'}], weight = 100)
ballots_2 = [b1, b2, b3]
pref_2 = PreferenceProfile(ballots = ballots_2)

round_0 = Outcome(curr_round = 0, elected = [], eliminated = [], remaining = ['A', 'B', 'C'], profile = pref_2, previous= None)
round_1 = Outcome(curr_round = 1, elected = [], eliminated = ['C'], remaining = ['B','A'], profile = None, previous = round_0)
round_2 = Outcome(curr_round = 2, elected = ['B'], eliminated = ['A'], remaining = [], winner_votes = {'B': [{'C'}, {'B'}, {'A'}]}, profile = None, previous = round_1)
    
rounds = [round_0, round_1, round_2]
rds = [0,1,2]
elects = [[], [], ['B']]
elims = [[], ['C'], ['A']]
remains = [['A', 'B', 'C'], ['B', 'A'], []]
wins = [[], [], ['B']]
los = [[], ['C'], ['A', 'C']]
ranks = [['A', 'B', 'C'], ['B', 'A', 'C'], ['B', 'A', 'C']]

changes = [{'A':(0,1), 'B':(1,0)}, None]
def test_outcome_get_attributes():
    
    for i in range(3):
        assert rounds[i].curr_round == rds[i]
        assert rounds[i].elected == elects[i]
        assert rounds[i].eliminated == elims[i]
        assert rounds[i].remaining == remains[i]
    
    #orig = Outcome(remaining=["A", "B", "C"])
    #new = orig.add_winners_and_losers({"B"}, {"A"})
    #assert new.elected == {"B"}
    #assert new.eliminated == {"A"}
    #assert new.remaining == {"C"}

def test_outcome_get_alls():
    for i in range(3):
        assert rounds[i].get_all_winners() == wins[i]
        assert rounds[i].get_all_eliminated() == los[i]
        assert rounds[i].get_rankings() == ranks[i]
        assert rounds[i].get_profile() == pref_2
        
def test_outcome_changed_rankings():
    for i in [1,2]:
        assert rounds[i].changed_rankings() == changes[i-1]

        
##TODO: the rest of this
def test_outcome_round_outcome():
    return

def test_outcome_difference_remaining_candidates():
    return

    
test_outcome_get_attributes()
test_outcome_get_alls()
test_outcome_changed_rankings()
test_outcome_round_outcome()
test_outcome_difference_remaining_candidates()

#def test_outcome_promote_winners_error(): 
#    orig = Outcome(remaining=["A", "B", "C"])
#    with pytest.raises(ValueError) as exc:
#        orig.add_winners_and_losers({"D"}, {"E"})

    # ensure error message mentions both bad candidates
    #assert "D" in str(exc.value)
    #assert "E" in str(exc.value)


#def test_outcome_immutable():
    # not that important to test this since it is implemented by pydantic
    # but provided as a check that we remembered to set allow_mutation = False
    #orig = Outcome(remaining={"A"})
#    with pytest.raises(TypeError):
#        orig.remaining = {"A", "B", "C"}
        

