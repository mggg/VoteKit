import pytest

from votekit.ballot import ScoreBallot
from votekit.elections.election_types.scores.quadratic import Quadratic
from votekit.pref_profile import ScoreProfile

profile_no_tied_votes = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 2, "B": 1, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 1, "B": 0, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 2, "B": 1, "C": 1}),
    ]
)
# votes = 4, 2, 4
# credits = 6, 2, 6

profile_no_tied_credits = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 4, "B": 1, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 1, "B": 0, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 4, "B": 1, "C": 1}),
    ]
)
# votes = 4, 2, 4
# credits = 6, 2, 6

profile_tied_votes = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 2, "C": 1}),
        ScoreBallot(scores={"A": 2, "B": 1, "C": 2}),
    ]
)
# votes = 4, 5
# credits = 6, 9

# TODO
profile_tied_credits = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 4, "C": 1}),
        ScoreBallot(scores={"A": 4, "B": 1, "C": 4}),
    ]
)
# votes = 4, 5
# credits = 6, 9


def test_init():
    # Votes
    e = Quadratic(profile_no_tied_votes, k=6)
    assert e.get_elected() == (frozenset({"A"}),)

    e = Quadratic(profile_no_tied_votes, m=2, k=6)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"C"}))

    # Credits
    e = Quadratic(profile_no_tied_credits, k=6, is_credits=True)
    assert e.get_elected() == (frozenset({"A"}),)

    e = Quadratic(profile_no_tied_credits, m=2, k=6, is_credits=True)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"C"}))


def test_ties():

    # Votes
    e_random = Quadratic(profile_tied_votes, m=1, k=10, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1

    e_random = Quadratic(profile_tied_votes, m=2, k=10, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 2

    e_random = Quadratic(profile_tied_votes, m=3, k=10, tiebreak="random")
    assert e_random.get_elected() == (frozenset({"A", "C", "B"}),)

    # Credits
    e_random = Quadratic(profile_tied_credits, m=1, k=10, is_credits=True, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1

    e_random = Quadratic(profile_tied_credits, m=2, k=10, is_credits=True, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 2

    e_random = Quadratic(profile_tied_credits, m=3, k=10, is_credits=True, tiebreak="random")
    assert e_random.get_elected() == (frozenset({"A", "C", "B"}),)


def test_errors():
    # Votes
    # with pytest.raises(ValueError, match="Credit budget k must be a whole number."):
    #    Quadratic(profile_no_tied_votes, k = 1.5)

    # Credits
    with pytest.raises(ValueError, match="Credit budget k must be a whole number."):
        Quadratic(profile_no_tied_credits, k=1.5)


def test_validate_profile():
    # Both votes and credits
    with pytest.raises(ValueError, match="Scores must be whole numbers."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3.5})])
        Quadratic(profile, m=1, k=5)

    with pytest.raises(ValueError, match="Scores must be whole numbers."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3.5})])
        Quadratic(profile, m=1, k=5)

    # Credits
    with pytest.raises(ValueError, match="is above the credit budget."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 9})])
        Quadratic(profile, m=1, k=2, is_credits=True)

    with pytest.raises(ValueError, match="score violates credit's perfect squares requirement."):
        profile = ScoreProfile(ballots=[ScoreBallot(), ScoreBallot(scores={"A": 3})])
        Quadratic(profile, m=1, k=4, is_credits=True)

    # Votes
    with pytest.raises(ValueError, match="is above the credit budget."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 2})])
        Quadratic(profile, m=1, k=3)
