from .ballot import Ballot, RankBallot, ScoreBallot
from .pref_interval import PreferenceInterval
from .pref_profile import PreferenceProfile, RankProfile, ScoreProfile

__all__ = [
    "Ballot",
    "RankBallot",
    "ScoreBallot",
    "PreferenceProfile",
    "RankProfile",
    "ScoreProfile",
    "PreferenceInterval",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path (e.g. votekit.Ballot) instead of the full
# internal path where each object is defined (e.g. votekit.ballot.Ballot).
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
