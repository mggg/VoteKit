from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.cleaning import remove_empty_ballots


def test_remove_empty_ballots():
    profile = PreferenceProfile(
        ballots=[
            Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=1),
            Ballot(ranking=[], weight=1),
        ]
    )
    profile_cleaned = remove_empty_ballots(profile)
    assert len(profile_cleaned.get_ballots()) == 1
    ballot = profile_cleaned.get_ballots()[0]
    assert ballot.ranking == [{'A'}, {'B'}, {'C'}]
