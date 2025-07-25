from votekit.animations import STVAnimation
from votekit import Ballot, PreferenceProfile
from votekit.elections import STV

# modified from STV wiki
# Election following the "happy path". One elimination or election per round. No ties. No exact quota matches. No funny business.
test_profile_happy = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"Orange"}, {"Pear"}), weight=3),
        Ballot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
        Ballot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1),
        Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
        Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
        Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
        Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
    ),
    max_ranking_length=3,
)

test_election_happy = STV(test_profile_happy, m=3)


def test_init():
    animation = STVAnimation(test_election_happy)
    assert animation.candidates is not None
    assert animation.rounds is not None