from votekit.pref_profile import RankProfile
from votekit.ballot import RankBallot
from votekit.cleaning import condense_ranked_profile, remove_cand_from_rank_profile


def test_cleaning_series():
    profile = RankProfile(
        ballots=[
            RankBallot(
                ranking=[
                    {"A"},
                ],
                weight=1,
            ),
            RankBallot(weight=0),
            RankBallot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        ]
    )

    profile_no_A = remove_cand_from_rank_profile(["A"], profile)
    condensed = condense_ranked_profile(profile_no_A)
    assert condensed.group_ballots().ballots == (
        RankBallot(ranking=[{"C"}, {"B"}], weight=3),
    )
