from dataclasses import dataclass, field
from typing import Any
import math
import pandas as pd


def _format_ranked_sets(ranked: tuple[frozenset[str], ...]) -> str:
    """
    Format something like ({"A"}, {"B","C"}, {"D"}) as: "A > B=C > D"
    Empty/placeholder frozensets are ignored.
    """
    groups: list[str] = []
    for s in ranked:
        if not s:
            continue
        groups.append("=".join(sorted(s)))
    return " > ".join(groups)


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
        scores(dict[str, float], optional): Stores score information.
            Keys are candidates, values are scores. Only remaining candidates should be stored.
        precision (float, optional): Used only for approximate score equality. Defaults to 1e-6. 
    """
    round_number: int = 0
    remaining: tuple[frozenset[str], ...] = (frozenset(),)
    elected: tuple[frozenset[str], ...] = (frozenset(),)
    eliminated: tuple[frozenset[str], ...] = (frozenset(),)
    tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = field(
        default_factory=dict
    )
    scores: dict[str, float] = field(default_factory=dict)

    # Used only for approximate score equality. Not part of structural equality / repr.
    precision: float = field(default=1e-6, repr=False, compare=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElectionState):
            return NotImplemented

        if (
            self.round_number != other.round_number
            or self.remaining != other.remaining
            or self.elected != other.elected
            or self.eliminated != other.eliminated
            or self.tiebreaks != other.tiebreaks
        ):
            return False

        if self.scores.keys() != other.scores.keys():
            return False

        tol = min(self.precision, other.precision)
        for k in self.scores.keys():
            a = self.scores[k]
            b = other.scores[k]

            if (
                isinstance(a, float)
                and isinstance(b, float)
                and math.isnan(a)
                and math.isnan(b)
            ):
                continue

            if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
                return False

        return True

    def as_df(self) -> pd.DataFrame:
        """
        Return the same DataFrame that __repr__ renders.

        Index:
          - optionally "elected" row (string value like "A > B=C")
          - one row per candidate score
          - optionally "eliminated" row (string value like "D")
        Column: "value"
        """
        elected_str = _format_ranked_sets(self.elected)
        eliminated_str = _format_ranked_sets(self.eliminated)

        # Sort scores for readability (highest first, then name).
        score_items = sorted(self.scores.items(), key=lambda kv: (-kv[1], kv[0]))

        idx: list[str] = []
        vals: list[object] = []

        if elected_str:
            idx.append(elected_str)
            vals.append("Elected")

        for cand, score in score_items:
            idx.append(cand)
            vals.append(score)

        if eliminated_str:
            idx.append(eliminated_str)
            vals.append("Eliminated")

        s = pd.Series(vals, index=idx, name=f"Round {self.round_number}")
        return s.to_frame("value")

    @property
    def df(self) -> pd.DataFrame:
        """Convenience accessor for the repr DataFrame."""
        return self.as_df()

    def __repr__(self) -> str:
        remaining_str = _format_ranked_sets(self.remaining)
        df = self.as_df()

        header = f"ElectionState(round={self.round_number}"
        if remaining_str:
            header += f", remaining={remaining_str}"
        header += ")"

        return header + "\n" + df.to_string()

    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class old_ElectionState:
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
        scores(dict[str, float], optional): Stores score information.
            Keys are candidates, values are scores. Only remaining candidates should be stored.

    """

    round_number: int = 0
    remaining: tuple[frozenset[str], ...] = (frozenset(),)
    elected: tuple[frozenset[str], ...] = (frozenset(),)
    eliminated: tuple[frozenset[str], ...] = (frozenset(),)
    tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = field(
        default_factory=dict
    )
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ElectionState to a dictionary representation.
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
