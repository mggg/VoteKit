from .abstract_ranking import RankingElection
from .alaska import Alaska
from .boosted_random_dictator import BoostedRandomDictator
from .borda import Borda
from .condo_borda import CondoBorda
from .dominating_sets import DominatingSets
from .plurality import (
    SNTV,
    Plurality,
)
from .plurality_veto import PluralityVeto, SerialVeto
from .random_dictator import RandomDictator
from .ranked_pairs import RankedPairs
from .schulze import Schulze
from .simultaneous_veto import SimultaneousVeto
from .stv import IRV, STV, FastSTV, SequentialRCV
from .top_two import TopTwo

__all__ = [
    "RankingElection",
    "Plurality",
    "SNTV",
    "Borda",
    "STV",
    "FastSTV",
    "IRV",
    "SequentialRCV",
    "Alaska",
    "DominatingSets",
    "CondoBorda",
    "TopTwo",
    "PluralityVeto",
    "SerialVeto",
    "SimultaneousVeto",
    "RandomDictator",
    "BoostedRandomDictator",
    "RankedPairs",
    "Schulze",
]
