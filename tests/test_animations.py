from votekit.animations import STVAnimation
from votekit import Ballot, PreferenceProfile
from votekit.elections import STV
import pytest

# modified from STV wiki
# Election following the "happy path". One elimination or election per round. No ties. No exact quota matches. No funny business.


@pytest.fixture
def election_happy():
    profile_happy = PreferenceProfile(
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
    return STV(profile_happy, m=3)


def test_STVAnimation_init(election_happy):
    animation = STVAnimation(election_happy)
    assert isinstance(animation.candidate_dict, dict)
    assert isinstance(animation.events, list)
    assert "Pear" in animation.candidate_dict.keys()
    assert animation.candidate_dict["Pear"]["support"] == 8
