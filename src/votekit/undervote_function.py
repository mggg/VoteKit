from profile import PreferenceProfile

from ballot import Ballot


profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {" "}, {"C"}], weight=10),
        Ballot(ranking=[{" "}, {" "}, {" "}, {"D"}, {"F"}], weight=5),
        Ballot(ranking=[{"A"}, {" "}, {" "}, {"D"}, {" "}], weight=6),
    ]
)


ballots_list = profile.ballots


def clean(ballots_list):
    cleaned_rank_list = []
    for ballot in ballots_list:
        rank_list = ballot.ranking
        cleaned_rank_list_temp = [rank for rank in rank_list if " " not in rank]
        cleaned_rank_list.append(cleaned_rank_list_temp)
    return cleaned_rank_list


result = clean(ballots_list)
ballots_list = result
