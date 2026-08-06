from dataclasses import dataclass, field
from typing import Any

from votekit.types import Candidate, CandidateNumericDict


@dataclass
class ElectionState:
    """
    Class for storing information about a round of an election. Round 0 should be
    the initial state of the election. To save memory, the PreferenceProfile is
    not carried by the ElectionState class.

    Attributes:
        round_number (int, optional): Round number, defaults to 0.
        remaining (tuple[frozenset[Candidate],...], optional): Remaining candidates, ordered to
            indicate ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
            Candidates can be strings, integers, or mix of both.
        elected (tuple[frozenset[Candidate],...], optional): Elected candidates, ordered to
            indicate ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
            Candidates can be strings, integers, or mix of both.
        eliminated (tuple[frozenset[Candidate],...], optional): Eliminated candidates, ordered to
            indicate ranking, frozensets to indicate ties. Defaults to tuple with one empty set.
            Candidates can be strings, integers, or mix of both.
        tiebreaks (dict[frozenset[Candidate], tuple[frozenset[Candidate],...]], optional): Stores
            tiebreak resolutions. Keys are frozensets of tied candidates and values are resolutions
            of tiebreak. Defaults to empty dictionary.
            Candidates can be strings, integers, or mix of both.
        scores (CandidateNumericDict, optional): Stores score information.
            Keys are candidates, values are scores. Only remaining candidates should be stored.
            Candidates can be strings, integers, or mix of both.

    """

    round_number: int = 0
    remaining: tuple[frozenset[Candidate], ...] = (frozenset(),)
    elected: tuple[frozenset[Candidate], ...] = (frozenset(),)
    eliminated: tuple[frozenset[Candidate], ...] = (frozenset(),)
    tiebreaks: dict[frozenset[Candidate], tuple[frozenset[Candidate], ...]] = field(
        default_factory=dict
    )
    scores: CandidateNumericDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ElectionState to a Python dictionary representation.

        Score values are returned unchanged and may not be JSON serializable.
        """
        return {
            "round_number": self.round_number,
            "remaining": tuple(map(tuple, self.remaining)),
            "elected": tuple(map(tuple, self.elected)),
            "eliminated": tuple(map(tuple, self.eliminated)),
            "tiebreaks": {
                tuple(tie): tuple(map(tuple, resolution))
                for tie, resolution in self.tiebreaks.items()
            },
            "scores": self.scores,
        }
