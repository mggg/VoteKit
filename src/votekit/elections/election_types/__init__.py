from .ranking import (  # noqa
    RankingElection,
    Plurality,
    SNTV,
    Borda,
    STV,
    IRV,
    Alaska,
    DominatingSets,
    SequentialRCV,
    CondoBorda,
    TopTwo,
    PluralityVeto,
    RandomDictator,
    BoostedRandomDictator,
)


from .scores import GeneralRating, Rating, Limited, Cumulative  # noqa
from .scores.star import Star # noqa
from .approval import Approval, BlocPlurality  # noqa
from .approval.open_list_pr import OpenListPR # noqa
