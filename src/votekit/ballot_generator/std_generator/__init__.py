from votekit.ballot_generator.std_generator.impartial_anon_culture import (
    iac_profile_generator,
)
from votekit.ballot_generator.std_generator.impartial_culture import (
    ic_profile_generator,
)
from votekit.ballot_generator.std_generator.spacial import (
    clustered_spacial_profile_and_positions_generator,
    onedim_spacial_profile_generator,
    spacial_profile_and_positions_generator,
)

__all__ = [
    "ic_profile_generator",
    "iac_profile_generator",
    "spacial_profile_and_positions_generator",
    "onedim_spacial_profile_generator",
    "clustered_spacial_profile_and_positions_generator",
]

# Patch __module__ on every exported symbol so that Sphinx autodoc displays
# the canonical public import path instead of the full internal path where
# each object is defined.
for _name in __all__:
    _obj = globals().get(_name)
    if _obj is not None and hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
