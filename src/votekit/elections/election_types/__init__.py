from .approval import Approval, BlocPlurality
from .ranking import (
    IRV,
    SNTV,
    STV,
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
    "GeneralRating",
    "Rating",
    "Limited",
    "Cumulative",
    "Approval",
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
