from .distances import earth_mover_dist, euclidean_dist, lp_dist

__all__ = [
    "earth_mover_dist",
    "lp_dist",
    "euclidean_dist",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
