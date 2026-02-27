from .abstract_ranking import RankingElection

from .plurality import (
    Plurality,
    SNTV,
)

from .borda import Borda
from .stv import FastSTV, STV, IRV, SequentialRCV
from .alaska import Alaska
from .top_two import TopTwo
from .dominating_sets import DominatingSets
from .condo_borda import CondoBorda
from .plurality_veto import PluralityVeto
from .plurality_veto import SerialVeto
from .simultaneous_veto import SimultaneousVeto
from .random_dictator import RandomDictator
from .boosted_random_dictator import BoostedRandomDictator
from .ranked_pairs import RankedPairs
from .schulze import Schulze


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
