from .abstract_ranking import RankingElection  # noqa

from .plurality import (  # noqa
    Plurality,
    SNTV,
)

from .borda import Borda  # noqa
from .stv import STV, IRV, SequentialRCV  # noqa
from .alaska import Alaska  # noqa
from .top_two import TopTwo  # noqa
from .dominating_sets import DominatingSets  # noqa
from .condo_borda import CondoBorda  # noqa
from .top_two import TopTwo  # noqa
from .plurality_veto import PluralityVeto  # noqa
from .random_dictator import RandomDictator  # noqa
from .boosted_random_dictator import BoostedRandomDictator  # noqa
