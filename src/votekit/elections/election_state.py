from dataclasses import dataclass
from dataclasses import field
from fractions import Fraction
from typing import Union


@dataclass
class ElectionState:
    """
    Class for storing information about a round of an election. Round 0 should be
    the initial state of the election. To save memory, the PreferenceProfile is
    not carried by the ElectionState class.

    Attributes:
        round_number (int, optional): Round number, defaults to 0.
        remaining (tuple[frozenset[str],...], optional): Remaining candidates, ordered to indicate
            ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
        elected (tuple[frozenset[str],...], optional): Elected candidates, ordered to indicate
            ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
        eliminated (tuple[frozenset[str],...], optional): Eliminated candidates, ordered to indicate
            ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
        tiebreaks (dict[frozenset[str], tuple[frozenset[str],...]], optional): Stores
            tiebreak resolutions. Keys are frozensets of tied candidates and values are resolutions
            of tiebreak. Defaults to empty dictionary.
        scores(Union[dict[str, Fraction], dict[str, float], optional): Stores score information.
            Keys are candidates, values are scores. Only remaining candidates should be stored.

    """

    round_number: int = 0
    remaining: tuple[frozenset[str], ...] = (frozenset(),)
    elected: tuple[frozenset[str], ...] = (frozenset(),)
    eliminated: tuple[frozenset[str], ...] = (frozenset(),)
    tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = field(
        default_factory=dict
    )
    scores: Union[dict[str, Fraction], dict[str, float]] = field(default_factory=dict)
