from profile import PreferenceProfile
from ballot import Ballot

profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {" "}, {"C"}], weight=10),
        Ballot(ranking=[{" "}, {" "}, {" "}, {"D"}, {"F"}], weight=5),
        Ballot(ranking=[{"A"}, {" "}, {" "}, {"D"}, {" "}], weight=6),
    ]
)
