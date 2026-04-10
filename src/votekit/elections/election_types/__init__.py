from .approval import Approval
from .block_plurality import BlockPlurality
from .ranking import (
    IRV,
    SNTV,
    STV,
    MeekSTV,
    Alaska,
    BoostedRandomDictator,
    Borda,
    CondoBorda,
    DominatingSets,
    FastSTV,
    Plurality,
    PluralityVeto,
    RandomDictator,
    RankedPairs,
    RankingElection,
    Schulze,
    SequentialRCV,
    SerialVeto,
    SimultaneousVeto,
    TopTwo,
)
from .scores import Cumulative, GeneralRating, Limited, Rating
from .scores.block_plurality import BlocPlurality

__all__ = [
    "RankingElection",
    "Plurality",
    "SNTV",
    "Borda",
    "STV",
    "MeekSTV",
    "FastSTV",
    "IRV",
    "SequentialRCV",
    "Alaska",
    "DominatingSets",
    "CondoBorda",
    "TopTwo",
    "GeneralRating",
    "Rating",
    "Limited",
    "Cumulative",
    "Approval",
    "BlockPlurality",
    "BlocPlurality",
    "PluralityVeto",
    "SerialVeto",
    "SimultaneousVeto",
    "RandomDictator",
    "BoostedRandomDictator",
    "RankedPairs",
    "Schulze",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
