from votekit.elections.election_state import ElectionState
from votekit.elections.election_types import (
    IRV,
    SNTV,
    STV,
    MeekSTV,
    Alaska,
    Approval,
    BlockPlurality,
    BlocPlurality,
    BoostedRandomDictator,
    Borda,
    CondoBorda,
    Cumulative,
    DominatingSets,
    FastSTV,
    GeneralRating,
    Limited,
    Plurality,
    PluralityVeto,
    RandomDictator,
    RankedPairs,
    RankingElection,
    Rating,
    Schulze,
    SequentialRCV,
    SerialVeto,
    SimultaneousVeto,
    TopTwo,
)
from votekit.elections.transfers import fractional_transfer, random_transfer
from votekit.models import Election

__all__ = [
    "ElectionState",
    "Election",
    "fractional_transfer",
    "random_transfer",
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
    "Cumulative",
    "Limited",
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
# the canonical public import path (e.g. votekit.elections.RankingElection)
# instead of the full internal path where each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
