from .boost import boost_matrix, boost_prob
from .candidate_distance import candidate_distance, candidate_distance_matrix
from .comentions import comention, comention_above, comentions_matrix

__all__ = [
    "comention",
    "comention_above",
    "comentions_matrix",
    "boost_prob",
    "boost_matrix",
    "candidate_distance",
    "candidate_distance_matrix",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
