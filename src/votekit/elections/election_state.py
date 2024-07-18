from dataclasses import dataclass
from dataclasses import field


@dataclass
class ElectionState:
    """
    Class for storing information about a round of an election. Round 0 should be
    the initial state of the election. To save memory, the PreferenceProfile is
    not carried by the ElectionState class.

    tiebreak_winners is a tuple[frozenset] so that you can store, in order, the candidates
    who won a tiebreak within a round.
    """

    round_number: int = 0
    remaining: frozenset = frozenset()
    elected: frozenset = frozenset()
    eliminated: frozenset = frozenset()
    tiebreak_winners: tuple[frozenset] = tuple([frozenset()])
    scores: dict[str, float] = field(default_factory=dict)
