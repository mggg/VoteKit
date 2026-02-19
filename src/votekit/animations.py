from copy import deepcopy
import textwrap
import manim  # type: ignore
from manim import (
    Rectangle,
    SurroundingRectangle,
    Line,
    Create,
    Uncreate,
    FadeIn,
    FadeOut,
    Paragraph,
    Text,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ManimColor,
    ParsableManimColor,
)
from votekit.cleaning.rank_ballots_cleaning import (
    condense_rank_ballot,
    remove_cand_rank_ballot,
)
from votekit.utils import ballots_by_first_cand, mentions
from votekit.elections.election_types.ranking.stv import STV
from typing import Literal, List, Optional, Sequence, Mapping
from collections import defaultdict
import logging
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ColorPalette:
    """
    A color palette for STV animations. Colors may be provided in any format that the Manim
    library can handle. For instance, colors may be
    - Hex code strings
    - rbg triples of floats
    - RGB triples of integers
    - ManimColor objects
    Note that any provided alpha channel information will be ignored.

    Attributes:
        bar_fills (List[ParsableManimColor]): Colors for candidate bars (cycles through list).
        bar_outline (ParsableManimColor): Color for bar outlines.
        winner (ParsableManimColor): Color for boxes around winner names and winner bars.
        offscreen_candidate_fill (ParsableManimColor): Color for candidates not shown on screen.
        background (ParsableManimColor): Background color.
        elimination_line (ParsableManimColor): Color for candidate name strikethrough lines.
        text_frosted (ParsableManimColor): Color for de-emphasized text.
        text_regular (ParsableManimColor): Color for regular text.
    """

    bar_fills: List[ParsableManimColor]
    bar_outline: ParsableManimColor
    winner: ParsableManimColor
    offscreen_candidate_fill: ParsableManimColor
    background: ParsableManimColor
    elimination_line: ParsableManimColor
    text_frosted: ParsableManimColor
    text_regular: ParsableManimColor


DARK_PALETTE = ColorPalette(
    bar_fills=[
        "#68359B",
        "#006B3D",
        "#E52121",
        "#1460BC",
        "#FFB7C4",
        "#FFA811",
        "#FFE035",
        "#8CD1C4",
    ],
    bar_outline="#BBBBBB",
    winner="#8CB500",
    offscreen_candidate_fill="#888888",
    background="#000000",
    elimination_line="#D11942",
    text_frosted="#444444",
    text_regular="#FFFFFF",
)

LIGHT_PALETTE = ColorPalette(
    bar_fills=[
        "#68359B",
        "#006B3D",
        "#E52121",
        "#1460BC",
        "#FFB7C4",
        "#FFA811",
        "#FFE035",
        "#8CD1C4",
    ],
    bar_outline="#000000",
    winner="#8CB500",
    offscreen_candidate_fill="#888888",
    background="#FFFFFF",
    elimination_line="#D11942",
    text_frosted="#BBBBBB",
    text_regular="#000000",
)


@dataclass
class AnimationEvent(ABC):
    """
    An abstract class representing a single step of the animation, usually a single election round.

    Attributes:
        quota (float): The current election threshold at the start of this event.
    """

    quota: float

    @abstractmethod
    def get_message(self) -> str:
        """Generate a message describing the event for the viewer of the animation."""
        pass


@dataclass
class EliminationEvent(AnimationEvent):
    """
    An animation event representing a round in which a candidate was eliminated.

    Attributes:
        candidate (str): The name of the eliminated candidate.
        display_name (str): The candidate name to use for display purposes, such as a
            nickname.
        support_transferred (Mapping[str, float]): A dictionary mapping names of candidates
            to the amount of support they received from the elimination.
        round_number (int): The round of the election process associated to this event.
    """

    candidate: str
    display_name: str
    support_transferred: Mapping[str, float]
    round_number: int

    def get_message(self) -> str:
        return f"Round {self.round_number}: {self.display_name} eliminated."


@dataclass
class EliminationOffscreenEvent(AnimationEvent):
    """
    An animation event representing some number of rounds in which offscreen candidates
        were eliminated.

    Attributes:
        support_transferred (Mapping[str, float]): A dictionary mapping names of candidates
            to the total amount of support they received from the eliminations.
        round_numbers (List[int]): The rounds of the election process associated to this event.
    """

    support_transferred: Mapping[str, float]
    round_numbers: List[int]

    def get_message(self) -> str:
        if len(self.round_numbers) == 1:
            return f"Round {self.round_numbers[0]}: 1 candidate eliminated."
        else:
            message = f"Rounds {self.round_numbers[0]}-{self.round_numbers[-1]}: {len(self.round_numbers)} candidates eliminated."
            return message


@dataclass
class WinEvent(AnimationEvent):
    """
    An animation event representing a round in which some number of candidates were elected.

    Attributes:
        candidates (Sequence[str]): The names of the elected candidates.
        display_names (Sequence[str]): The candidate names to use for display purposes,
            such as nicknames.
        support_transferred (Mapping[str, Mapping[str, float]]): A dictionary mapping
            pairs of candidate names to the amount of support transferred between them
            this round. For instance, if ``c1`` was elected this round, then
            ``support_transferred[c1][c2]`` will represent the amount of support
            that ran off from ``c1`` to candidate ``c2``.
        round_number (int): The round of the election process associated to this event.
    """

    candidates: Sequence[str]
    display_names: Sequence[str]
    support_transferred: Mapping[str, Mapping[str, float]]
    round_number: int

    def get_message(self) -> str:
        candidate_string = ", ".join(self.display_names)
        return f"Round {self.round_number}: {candidate_string} elected."


class STVAnimation:
    """
    A class which creates round-by-round animations of STV elections.

    Args:
        election (STV): An STV election to animate.
        title (str, optional): Text to be displayed at the beginning of the animation as
            a title screen. If ``None``, the title screen will be skipped. Defaults to ``None``.
        focus (set[str], list[str], "winners", "viable", or "all", optional): A set or list of
            names of candidates that should appear on-screen. This is useful for elections
            with many candidates. Note that any candidates that won the election are on-screen
            automatically, so passing an empty set will result in only elected candidates
            appearing on-screen. If ``"winners"``, focus only the elected candidates.
            If ``"viable"``, focus only the candidates with more mentions than the election
            threshold. If ``"all"``, focus all candidates. Defaults to ``"viable"``.
        nicknames (Optional[dict[str,str]], optional): A dictionary mapping candidate names to candidate
            "nicknames" to be used in the animation instead. The keys of ``nicknames``
            need not contain every candidate, only the ones for which the user would like to
            provide a nickname.
        candidate_colors (Optional[Mapping[str, ParsableManimColor]], optional): A dictionary
            mapping candidate names to colors that should represent them in the animation.
            The colors in ``candidate_colors`` will override the bar fill colors provided by
            ``color_palette``. The keys of ``candidate_colors`` need not contain
            every candidate, only the ones for which the user would like to provide
            a specific color. Defaults to the empty dictionary.
        color_palette (ColorPalette, optional): A color palette to use for the animation.
            Defaults to `DARK_PALETTE`.


    Attributes:
        title (str, optional): Text to be displayed at the beginning of the animation as
            a title screen.
        focus (set[str]): A set of names of candidates that should appear on-screen.
        nicknames (dict[str,str], optional): A dictionary mapping candidate names to candidate
            "nicknames" to be used in the animation instead.
        color_palette (ColorPalette, optional): A color palette to use for the animation.
        candidate_dict (dict[str, dict[str, object]]): A dictionary mapping each candidate's
            name to a dictionary recording that candidate's support, display name, and color.
        events (List[AnimationEvent]): A list of animation events in order of occurrence.

    Raises:
        TypeError: ``focus`` was not a set, list, or recognized string literal.
        ValueError: ``focus``, ``nicknames``, or ``candidate_colors`` contained
            candidate names not present in the election.
    """

    def __init__(
        self,
        election: STV,
        title: Optional[str] = None,
        focus: set[str] | List[str] | Literal["winners", "viable", "all"] = "viable",
        nicknames: Optional[dict[str, str]] = None,
        candidate_colors: Optional[Mapping[str, ParsableManimColor]] = None,
        color_palette: ColorPalette = DARK_PALETTE,
    ):
        if nicknames is None:
            nicknames = {}
        if candidate_colors is None:
            candidate_colors = {}

        user_provided_focus = isinstance(focus, (list, set))
        match focus:
            case "winners":
                focus = {c for s in election.get_elected() for c in s}
            case "viable":
                total_mentions = mentions(election.get_profile(0))
                focus = {
                    candidate
                    for candidate, ment in total_mentions.items()
                    if ment >= election.threshold
                }
            case "all":
                focus = {c for s in election.get_remaining(0) for c in s}
            case _:
                if not user_provided_focus:
                    raise TypeError(f"{focus} was not a recognized literal for focus")
                focus = set(focus)

        all_candidates = {c for s in election.get_remaining(0) for c in s}

        if user_provided_focus:
            invalid_names = focus - all_candidates
            if invalid_names:
                raise ValueError(
                    f"The following names in focus are not candidates "
                    f"in the election: {invalid_names}"
                )

        invalid_nicknames = nicknames.keys() - all_candidates
        if invalid_nicknames:
            raise ValueError(
                f"The following keys in nicknames are not candidates "
                f"in the election: {invalid_nicknames}"
            )

        invalid_colors = candidate_colors.keys() - all_candidates
        if invalid_colors:
            raise ValueError(
                f"The following keys in candidate_colors are not candidates "
                f"in the election: {invalid_colors}"
            )

        # Election winners must all be onscreen. Ensure it is so.
        elected_candidates = {c for s in election.get_elected() for c in s}
        missing_winners = elected_candidates - focus
        if user_provided_focus and missing_winners:
            warnings.warn(
                f"The focus list did not include all election winners. "
                f"Missing winners {missing_winners} have been added automatically.",
                UserWarning,
            )
        focus = focus | missing_winners
        self.focus = focus

        self.nicknames = nicknames
        self.color_palette = color_palette
        self.candidate_dict = self._make_candidate_dict(election, candidate_colors)
        self.events = self._make_event_list(election)
        if len(self.candidate_dict) == 0:
            raise ValueError("Tried creating animation with no candidates.")
        if len(self.events) == 0:
            raise ValueError("Tried creating animation with no animation event.")
        self.title = title

    def _make_candidate_dict(
        self, election: STV, candidate_colors: Mapping[str, ParsableManimColor]
    ) -> dict[str, dict[str, object]]:
        """
        Create a dictionary sending candidate names to dictionaries recording that candidate's
        support, display name, and color.

        Args:
            election (STV): An STV election from which to extract the candidates.
            candidate_colors (Mapping[str, ParsableManimColor]): A dictionary mapping candidate names
                to codes for colors to which they should be associated with in the
                candidate dictionary.

        Returns:
            dict[str, dict[str,object]]: A dictionary whose keys are candidate names and whose
                values are themselves dictionaries with details about each candidate.
        """
        # Initialize dictionary and add "support" key for each candidate.
        candidate_dict: dict[str, dict[str, object]] = {
            name: {"support": support}
            for name, support in election.election_states[0].scores.items()
            if name in self.focus
        }
        # Add display names
        for name in candidate_dict.keys():
            if name in self.nicknames.keys():
                display_name = self.nicknames[name]
            else:
                display_name = name
            candidate_dict[name]["display_name"] = display_name

        # Determine candidate color
        num_default_colors = len(self.color_palette.bar_fills)
        color_index = 0
        for name in candidate_dict.keys():
            if name in candidate_colors.keys():
                candidate_dict[name]["color"] = candidate_colors[name]
            else:
                candidate_dict[name]["color"] = self.color_palette.bar_fills[
                    color_index % num_default_colors
                ]
                color_index += 1
        return candidate_dict

    def _make_event_list(self, election: STV) -> List[AnimationEvent]:
        """
        Processes an STV election into a condensed list of animation events which roughly
        correspond to election rounds.

        Args:
            election (STV): The STV election to process.

        Returns:
            List[AnimationEvent]: A list of the events of the election which are
                worthy of animation.

        Raises:
            ValueError: If multiple candidates are eliminated in the same election round.
        """
        events: List[AnimationEvent] = []
        for round_number, election_round in enumerate(
            election.election_states[1:], start=1
        ):
            # Nothing happens in election round 0
            elected_candidates = [c for s in election_round.elected for c in s]
            eliminated_candidates = [c for s in election_round.eliminated for c in s]

            if len(elected_candidates) > 0:  # Win round
                support_transferred: dict[str, dict[str, float]] = {}
                if round_number == len(election):
                    # If it's the last round, don't worry about the transferred votes
                    support_transferred = {cand: {} for cand in elected_candidates}
                else:
                    support_transferred = self._get_transferred_votes(
                        election, round_number, elected_candidates, "win"
                    )
                display_names = [
                    str(self.candidate_dict[name]["display_name"])
                    for name in elected_candidates
                ]
                events.append(
                    WinEvent(
                        quota=election.threshold,
                        candidates=elected_candidates,
                        display_names=display_names,
                        support_transferred=support_transferred,
                        round_number=round_number,
                    )
                )
            elif len(eliminated_candidates) > 0:  # Elimination round
                if len(eliminated_candidates) > 1:
                    raise ValueError(
                        "Rounds with multiple eliminations are not supported. "
                        "At most one candidate should be eliminated in each round. "
                        f"Candidates eliminated in round {round_number}: {eliminated_candidates}."
                    )
                eliminated_candidate = eliminated_candidates[0]
                support_transferred = self._get_transferred_votes(
                    election, round_number, eliminated_candidates, "elimination"
                )
                if eliminated_candidate in self.focus:
                    display_name = str(
                        self.candidate_dict[eliminated_candidate]["display_name"]
                    )
                    events.append(
                        EliminationEvent(
                            quota=election.threshold,
                            candidate=eliminated_candidate,
                            display_name=display_name,
                            support_transferred=support_transferred[
                                eliminated_candidate
                            ],
                            round_number=round_number,
                        )
                    )
                else:
                    events.append(
                        EliminationOffscreenEvent(
                            quota=election.threshold,
                            support_transferred=support_transferred[
                                eliminated_candidate
                            ],
                            round_numbers=[round_number],
                        )
                    )

        events = self._condense_offscreen_events(events)

        return events

    def _get_transferred_votes(
        self,
        election: STV,
        round_number: int,
        cands_transferred_from: List[str],
        event_type: Literal["win", "elimination"],
    ) -> dict[str, dict[str, float]]:
        """
        Compute the number of votes transferred from each elected or eliminated candidate
        to each remaining candidate.

        Args:
            election (STV): The election.
            round_number (int): The number of the round in question.
            cands_transferred_from (List[str]): A list of the names of the elected or
                eliminated candidates.
            event_type (Literal["win", "elimination"]): ``"win"`` if candidates were elected this round,
                ``"elimination"`` otherwise.

        Returns:
            dict[str, dict[str, float]]: A nested dictionary. If ``d`` is the return value,
                ``c1`` was a candidate eliminated this round, and ``c2`` is a remaining candidate,
                then ``d[c1][c2]`` will be the total support transferred this round from
                candidate ``c1`` to candidate ``c2``.

        Notes:
            This function supports the election, but not the elimination, of multiple candidates
                in one round. If ``event_type`` is ``"elimination"`` then ``cands_transferred_from``
                should have length 1.
        """
        prev_profile, prev_state = election.get_step(round_number - 1)
        current_state = election.election_states[round_number]

        transfers: dict[str, dict[str, float]] = {}
        if event_type == "elimination":
            assert len(cands_transferred_from) == 1, (
                "Tried to compute transferred votes in a round "
                "with multiple eliminated candidates, which "
                "is not supported."
            )
            cand_transferred_from = cands_transferred_from[0]
            transfers = {cand_transferred_from: {}}
            for to_candidate in [
                c for s in current_state.remaining for c in s if c in self.focus
            ]:
                prev_score = prev_state.scores[to_candidate]
                current_score = current_state.scores[to_candidate]
                transfers[cand_transferred_from][to_candidate] = (
                    current_score - prev_score
                )
        elif event_type == "win":
            ballots_by_fpv = ballots_by_first_cand(prev_profile)
            for cand_transferred_from in cands_transferred_from:
                new_ballots = election.transfer(
                    cand_transferred_from,
                    prev_state.scores[cand_transferred_from],
                    ballots_by_fpv[cand_transferred_from],
                    election.threshold,
                )
                clean_ballots = [
                    condense_rank_ballot(
                        remove_cand_rank_ballot(cands_transferred_from, b)
                    )
                    for b in new_ballots
                ]
                transfer_weights_from_candidate: dict[str, float] = defaultdict(float)
                for ballot in clean_ballots:
                    if ballot.ranking is not None:
                        (to_candidate,) = ballot.ranking[0]
                        if to_candidate in self.focus:
                            transfer_weights_from_candidate[
                                to_candidate
                            ] += ballot.weight

                transfers[cand_transferred_from] = transfer_weights_from_candidate

        return transfers

    def _condense_offscreen_events(
        self, events: List[AnimationEvent]
    ) -> List[AnimationEvent]:
        """
        Take a list of events and condense any consecutive offscreen events into one summarizing
        event. For instance, if ``events`` contains three offscreen eliminations in a row,
        this function will condense them into one offscreen elimination of three candidates.

        Args:
            events (List[AnimationEvent]): A list of animation events to be condensed.

        Returns:
            List[AnimationEvent]: A condensed list of animation events.
        """
        if len(events) == 0:
            return []
        return_events: List[AnimationEvent] = [events[0]]
        for event in events[1:]:
            # Unless the next and previous events were both offscreen, just add
            # the next event to the list
            if not isinstance(
                return_events[-1], EliminationOffscreenEvent
            ) or not isinstance(event, EliminationOffscreenEvent):
                return_events.append(event)
            # If both events are offscreen, condense them.
            else:
                return_events[-1] = self._compose_offscreen_eliminations(
                    return_events[-1], event
                )
        return return_events

    def _compose_offscreen_eliminations(
        self, event1: EliminationOffscreenEvent, event2: EliminationOffscreenEvent
    ) -> EliminationOffscreenEvent:
        """
        Take two offscreen eliminations and "compose" them into a single offscreen elimination
        event summarizing both.

        Args:
            event1 (EliminationOffscreenEvent): The first offscreen elimination event to compose.
            event2 (EliminationOffscreenEvent): The second offscreen elimination event to compose.

        Returns:
            EliminationOffscreenEvent: One offscreen elimination event summarizing ``event1`` and
            ``event2``.
        """
        support_transferred: dict[str, float] = defaultdict(float)
        for key, value in event1.support_transferred.items():
            support_transferred[key] += value
        for key, value in event2.support_transferred.items():
            support_transferred[key] += value
        round_numbers = event1.round_numbers + event2.round_numbers
        quota = event1.quota
        return EliminationOffscreenEvent(
            quota=quota,
            support_transferred=support_transferred,
            round_numbers=round_numbers,
        )

    def render(
        self,
        preview: bool = False,
        render_dir: str = "media",
    ) -> None:
        """
        Renders the STV animation using Manim.

        The completed video will appear in the ``videos`` subdirectory within ``render_dir``.

        Args:
            preview (bool, optional): If ``True``, display the result in a video player
                immediately upon completing the render. Defaults to False.
            render_dir (str, optional): Directory in which the rendering files will appear.
        """
        # Set up necessary manim configurations.
        background_color = self.color_palette.background
        with manim.tempconfig(
            {"media_dir": render_dir, "background_color": background_color}
        ):
            # Animate
            manimation = ElectionScene(
                deepcopy(
                    self.candidate_dict
                ),  # deepcopy because this argument is mutated
                deepcopy(self.events),  # deepcopy because this argument is mutated
                title=self.title,
                color_palette=self.color_palette,
            )
            manimation.render(preview=preview)


class ElectionScene(manim.Scene):
    """
    Class for Manim animation of an STV election.

    Notes:
        This class is instantiated by the class ``STVAnimation``. It should not be
            instantiated directly.

    Args:
        candidate_dict (dict[str,dict]): A dictionary mapping each candidate to a dictionary of
            attributes of the candidate.
        events (List[AnimationEvent]): A list of animation events to be constructed and rendered.
        title (Optional[str], optional): A string to be displayed at the beginning of the animation as a title screen.
            If ``None``, the animation will skip the title screen. Defaults to ``None``.
        color_palette (ColorPalette, optional): A color scheme to use in the animation.
            Defaults to `DARK_PALETTE`.

    """

    def __init__(
        self,
        candidate_dict: dict[str, dict],
        events: List[AnimationEvent],
        title: Optional[str] = None,
        color_palette: ColorPalette = DARK_PALETTE,
    ):
        super().__init__()
        self.candidate_dict = candidate_dict
        self.events = events
        self.title = title
        self.color_palette = color_palette

        # Sizing and spacing for various aspects of the animation
        self.width = 8
        self.bar_height = 3.5 / len(self.candidate_dict)
        self.font_size = 3 * 40 / len(self.candidate_dict)
        self.bar_opacity = 1
        self.ghost_opacity = 0.3
        self.bar_buffer_size = self.bar_height
        self.strikethrough_thickness = self.font_size / 5
        self.ticker_tape_height = 2
        self.title_font_size = 120
        self.name_bar_spacing = 0.2
        self.winner_box_buffer = 0.1

        self.max_support = 1.1 * max([event.quota for event in self.events])

        self.quota_line = None
        self.ticker_tape_line = None
        self.ticker_tape: List[Text] = []

    def _make_text(self, text: str, font_size: float, color: ManimColor) -> Text:
        """
        Create a Text object with improved kerning for small font sizes.

        Manim/Pango has poor kerning at small font sizes. This method renders
        text at a larger size and scales it down to work around the issue.
        See: https://github.com/ManimCommunity/manim/issues/2844
        """
        SCALE_FACTOR = 0.1
        THRESHOLD = 32  # Only apply workaround for small text

        if font_size < THRESHOLD:
            scaled_text = Text(text, font_size=font_size / SCALE_FACTOR, color=color)
            scaled_text.scale(SCALE_FACTOR)
            return scaled_text
        else:
            return Text(text, font_size=font_size, color=color)

    def construct(self) -> None:
        """
        Constructs the animation.
        """

        # Manim produces a lot of logging output. Set the logging level to WARNING.
        logging.getLogger("manim").setLevel(logging.WARNING)

        if self.title is not None:
            self._draw_title(self.title)
        self._draw_initial_bars()
        self._initialize_ticker_tape()

        # Animate each event in turn
        for event_number, event in enumerate(self.events):
            self.wait(2)
            # Draw or move the quota line
            self._update_quota_line(event.quota)

            self._ticker_animation_shift(event_number)
            self._ticker_animation_highlight(event_number)

            if isinstance(event, EliminationEvent):  # Onscreen candidate eliminated
                # Remove the candidate from the candidate list
                eliminated_candidates = {
                    event.candidate: self.candidate_dict.pop(event.candidate)
                }
                self._animate_elimination(eliminated_candidates, event)
            elif isinstance(
                event, EliminationOffscreenEvent
            ):  # Offscreen candidate eliminated
                self._animate_elimination_offscreen(event)
            elif isinstance(event, WinEvent):  # Election round
                # Remove the candidates from the candidate list
                elected_candidates = {}
                for name in event.candidates:
                    elected_candidates[name] = self.candidate_dict.pop(name)
                self._animate_win(elected_candidates, event)
            else:
                raise Exception(f"Invalid type for event {event}.")
        self.wait(2)

    def _draw_title(self, message: str) -> None:
        """
        Draw the title screen.

        Args:
            message (str): String that the title screen will display.
        """
        lines = textwrap.wrap(message, width=40)
        text = Paragraph(
            *lines,
            alignment="center",
            font_size=self.title_font_size,
            color=ManimColor(self.color_palette.text_regular),
        )
        # Scale down to fit within 90 % of the frame if needed.
        max_width = manim.config.frame_width * 0.9
        max_height = manim.config.frame_height * 0.9
        if text.width > max_width:
            text.scale_to_fit_width(max_width)
        if text.height > max_height:
            text.scale_to_fit_height(max_height)

        self.play(FadeIn(text))
        self.wait(3)
        self.play(FadeOut(text))

    def _draw_initial_bars(self) -> None:
        """
        Instantiate and draw the names and bars for each candidate.
        """
        # Sort candidates by starting first place votes
        sorted_candidates = sorted(
            self.candidate_dict.keys(),
            key=lambda x: self.candidate_dict[x]["support"],
            reverse=True,
        )

        # Create candidate name text
        for i, name in enumerate(sorted_candidates):
            candidate = self.candidate_dict[name]
            candidate["name_text"] = self._make_text(
                str(candidate["display_name"]),
                font_size=self.font_size,
                color=ManimColor(candidate["color"]),
            )
            if i == 0:
                # First candidate goes at the top
                candidate["name_text"].to_edge(UP, buff=self.bar_buffer_size)
            else:
                # The rest of the candidates go below, right justified
                candidate["name_text"].next_to(
                    self.candidate_dict[sorted_candidates[i - 1]]["name_text"],
                    DOWN,
                    buff=self.bar_buffer_size,
                ).align_to(
                    self.candidate_dict[sorted_candidates[0]]["name_text"], RIGHT
                )
        # Align candidate names to the left
        group = manim.Group().add(
            *[candidate["name_text"] for candidate in self.candidate_dict.values()]
        )
        group.to_edge(LEFT)

        # Create bars
        for candidate in self.candidate_dict.values():
            candidate["bars"] = [
                Rectangle(
                    width=self._support_to_bar_width(candidate["support"]),
                    height=self.bar_height,
                    color=ManimColor(candidate["color"]),
                    fill_color=ManimColor(candidate["color"]),
                    fill_opacity=self.bar_opacity,
                ).next_to(candidate["name_text"], RIGHT, buff=self.name_bar_spacing)
            ]

        # Draw a large black rectangle for the background
        # so that the ticker tape vanishes behind it
        frame_width = manim.config.frame_width
        frame_height = manim.config.frame_height
        background = (
            Rectangle(
                width=frame_width,
                height=frame_height,
                fill_color=ManimColor(self.color_palette.background),
                color=ManimColor(self.color_palette.background),
                fill_opacity=1,
            ).shift(UP * self.ticker_tape_height)
            # Background must be in the back, but not behind the play-by-play
            .set_z_index(-1)
        )

        # Draw the bars and names
        self.play(
            *[
                FadeIn(self.candidate_dict[name]["name_text"])
                for name in sorted_candidates
            ],
            FadeIn(background),
        )
        self.play(
            *[
                Create(self.candidate_dict[name]["bars"][0])
                for name in sorted_candidates
            ]
        )

    def _initialize_ticker_tape(self) -> None:
        """
        Instantiate and draw the ticker tape line and text.
        """
        line_length = manim.config.frame_width
        ticker_line = Line(
            start=LEFT * line_length / 2,
            end=RIGHT * line_length / 2,
            color=ManimColor(self.color_palette.bar_outline),
        )
        ticker_line.to_edge(DOWN, buff=0).shift(UP * self.ticker_tape_height)
        # Keep this line in front of the bars and the quota line
        ticker_line.set_z_index(2)
        self.ticker_tape_line = ticker_line
        self.ticker_tape = []
        for i, event in enumerate(self.events):
            new_message = self._make_text(
                event.get_message(),
                font_size=24,
                color=ManimColor(self.color_palette.text_frosted),
            )
            if i == 0:
                new_message.to_edge(DOWN, buff=0).shift(DOWN)
            else:
                new_message.next_to(self.ticker_tape[-1], DOWN)
            # Messages need to disappear behind the background rectangle as they scroll by.
            new_message.set_z_index(-2)
            self.ticker_tape.append(new_message)

        self.play(Create(ticker_line))
        self.play(*[Create(message) for message in self.ticker_tape])

    def _ticker_animation_shift(self, event_number: int) -> None:
        """
        Animate the shifting of the ticker tape to display the message for a given round.

        Args:
            event_number (int): The index of the event whose message will shift into view.
        """
        shift_to_event = (
            self.ticker_tape[event_number]
            .animate.to_edge(DOWN, buff=0)
            .shift(UP * self.ticker_tape_height / 3)
        )
        drag_other_messages = [
            manim.MaintainPositionRelativeTo(
                self.ticker_tape[i], self.ticker_tape[event_number]
            )
            for i in range(len(self.ticker_tape))
            if i != event_number
        ]
        self.play(shift_to_event, *drag_other_messages)

    def _ticker_animation_highlight(self, event_number: int) -> None:
        """
        Play an animation graying out all ticker tape message but one.

        Args:
            event_number (int): The index of the event whose message will be highlighted.
        """
        highlight_message = self.ticker_tape[event_number].animate.set_color(
            ManimColor(self.color_palette.text_regular)
        )
        unhighlight_other_messages = [
            self.ticker_tape[i].animate.set_color(
                ManimColor(self.color_palette.text_frosted)
            )
            for i in range(len(self.ticker_tape))
            if i != event_number
        ]
        self.play(highlight_message, *unhighlight_other_messages)

    def _update_quota_line(self, quota: float) -> None:
        """
        Update the position of the quota line to reflect the given quota. If no quota line
        exists, create it and animate its creation.

        Args:
            quota (float): The threshold number of votes necessary to be elected in the
                current round.
        """
        some_candidate = list(self.candidate_dict.values())[0]
        if not self.quota_line:
            # If the quota line doesn't exist yet, draw it.
            assert self.ticker_tape_line is not None, (
                "Tried to draw the quota line before " "the ticker tape line."
            )
            line_bottom = self.ticker_tape_line.get_top()[1]
            line_top = manim.config.frame_height / 2
            self.quota_line = Line(
                start=UP * line_top,
                end=UP * line_bottom,
                color=ManimColor(self.color_palette.winner),
            )
            self.quota_line.align_to(some_candidate["bars"][0], LEFT)
            self.quota_line.shift((self.width * quota / self.max_support) * RIGHT)
            self.quota_line.set_z_index(1)  # Keep the quota line in the front

            self.play(Create(self.quota_line))
            self.wait(2)
        else:
            self.play(
                self.quota_line.animate.align_to(some_candidate["bars"][0], LEFT).shift(
                    (self.width * quota / self.max_support) * RIGHT
                )
            )

    def _animate_win(
        self, cands_transferred_from: dict[str, dict], event: WinEvent
    ) -> None:
        """
        Animate a round in which one or more candidates are elected.

        Args:
            cands_transferred_from (dict[str,dict]): A dictionary in which the keys are the
                candidates elected this round and the values are dictionaries recording
                the candidate's attributes.
            event (WinEvent): The event to be animated.
        """
        # Box the winners' names
        winner_boxes = [
            SurroundingRectangle(
                cand_transferred_from["name_text"],
                color=ManimColor(self.color_palette.winner),
                buff=self.winner_box_buffer,
            )
            for cand_transferred_from in cands_transferred_from.values()
        ]

        # Animate the box around the candidate name and the message text
        self.play(*[Create(box) for box in winner_boxes])

        # Create and animate the subdivision and redistribution of winners' leftover votes
        for (
            from_candidate_name,
            cand_transferred_from,
        ) in cands_transferred_from.items():
            old_bars: List[Rectangle] = cand_transferred_from["bars"]
            new_bars: List[Rectangle] = []
            transformations = []
            destinations = event.support_transferred[from_candidate_name]
            candidate_color = cand_transferred_from["color"]

            used_votes = min(event.quota, cand_transferred_from["support"])
            winner_bar = (
                Rectangle(
                    width=self._support_to_bar_width(used_votes),
                    height=self.bar_height,
                    color=ManimColor(self.color_palette.winner),
                    fill_color=ManimColor(candidate_color),
                    fill_opacity=self.bar_opacity,
                )
                .align_to(cand_transferred_from["bars"][0], LEFT)
                .align_to(cand_transferred_from["bars"][0], UP)
            )

            # Create a sub-bar for each destination
            for destination, votes in destinations.items():
                if votes <= 0:
                    continue
                sub_bar = Rectangle(
                    width=self._support_to_bar_width(votes),
                    height=self.bar_height,
                    color=ManimColor(candidate_color),
                    fill_color=ManimColor(candidate_color),
                    fill_opacity=self.bar_opacity,
                )
                # The first sub-bar should start at the right end of the
                # eliminated candidate's stack. The rest should be arranged
                # to the left of that one
                if len(new_bars) == 0:
                    sub_bar.align_to(cand_transferred_from["bars"][-1], RIGHT).align_to(
                        cand_transferred_from["bars"][-1], UP
                    )
                else:
                    sub_bar.next_to(new_bars[-1], LEFT, buff=0)
                new_bars.append(sub_bar)
                self.candidate_dict[destination]["support"] += votes

                # The sub-bars will move to be next to the bars of their destination candidates
                transformation = sub_bar.animate.next_to(
                    self.candidate_dict[destination]["bars"][-1], RIGHT, buff=0
                )
                transformations.append(transformation)

                # Let the new sub-bar be owned by its destination candidate
                self.candidate_dict[destination]["bars"].append(sub_bar)

            # Create a final short bar representing the exhausted votes
            exhausted_votes = (
                cand_transferred_from["support"]
                - used_votes
                - sum(list(destinations.values()))
            )
            exhausted_bar = Rectangle(
                width=self._support_to_bar_width(exhausted_votes),
                height=self.bar_height,
                color=ManimColor(candidate_color),
                fill_color=ManimColor(candidate_color),
                fill_opacity=self.bar_opacity,
            )
            assert self.quota_line is not None, (
                "Tried to animate a win event, but the quota " "line was never drawn."
            )
            if new_bars:
                exhausted_bar.next_to(new_bars[-1], LEFT, buff=0)
            else:
                exhausted_bar.next_to(winner_bar, RIGHT, buff=0)
            # Keep this bar behind the others
            # This helps things look clean in edge cases when there are few exhausted votes
            exhausted_bar.set_z_index(-1)
            transformations.append(Uncreate(exhausted_bar))

            # Animate the splitting of the old bar into the new sub_bars
            self.play(
                *[FadeOut(bar) for bar in old_bars],
                FadeIn(winner_bar),
                *[FadeIn(bar) for bar in new_bars],
                FadeIn(exhausted_bar),
            )

            # Animate moving the sub-bars to the destination bars,
            # and the destruction of the exhausted votes
            if len(transformations) > 0:
                self.play(*transformations)

    def _animate_elimination(
        self, cands_transferred_from: dict[str, dict], event: EliminationEvent
    ) -> None:
        """
        Animate a round in which a candidate was eliminated.

        Args:
            cands_transferred_from (dict[str,dict]): A dictionary in which the keys are the
                candidates eliminated this round and the values are dictionaries recording
                the candidate's attributes.
            event (EliminationEvent): The event to be animated.

        Notes:
            While the interface supports multiple candidate eliminations in one round for
                future extensibility, this function currently only supports elimination of one
                candidate at a time. The cands_transferred_from argument should have exactly one entry.

        Raises:
            ValueError: If the length of ``cands_transferred_from`` is not 1.
        """
        num_eliminated_candidates = len(list(cands_transferred_from.values()))
        if num_eliminated_candidates != 1:
            raise ValueError(
                "Elimination round animations only support one eliminated candidate at a time. "
                f"Attempted to animate {num_eliminated_candidates} "
                "eliminations in one election round."
            )

        cand_transferred_from = list(cands_transferred_from.values())[0]
        destinations = event.support_transferred

        # Cross out the candidate name
        cross = Line(
            cand_transferred_from["name_text"].get_left(),
            cand_transferred_from["name_text"].get_right(),
            color=ManimColor(self.color_palette.elimination_line),
        )
        cross.set_stroke(width=self.strikethrough_thickness)
        self.play(Create(cross))

        # Create short bars that will replace the candidate's current bars
        candidate_color = cand_transferred_from["color"]
        old_bars = cand_transferred_from["bars"]
        new_bars = []  # The bits to be redistributed
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width=self._support_to_bar_width(votes),
                height=self.bar_height,
                color=ManimColor(candidate_color),
                fill_color=ManimColor(candidate_color),
                fill_opacity=self.bar_opacity,
            )
            self.candidate_dict[destination]["support"] += votes
            new_bars.append(sub_bar)
            transformations.append(
                new_bars[-1].animate.next_to(
                    self.candidate_dict[destination]["bars"][-1], RIGHT, buff=0
                )
            )  # The sub-bars will move to be next to the bars of their destination candidates
            self.candidate_dict[destination]["bars"].append(sub_bar)
        # Create a final short bar representing the exhausted votes
        exhausted_votes = cand_transferred_from["support"] - sum(
            list(destinations.values())
        )
        exhausted_bar = Rectangle(
            width=self._support_to_bar_width(exhausted_votes),
            height=self.bar_height,
            color=ManimColor(candidate_color),
            fill_color=ManimColor(candidate_color),
            fill_opacity=self.bar_opacity,
        )
        exhausted_bar.align_to(old_bars[0], LEFT).align_to(old_bars[0], UP)

        if len(new_bars) > 0:
            # The short bars should start in the same place as the old bars. Place them.
            rightmost_old_bar = old_bars[-1]
            new_bars[0].align_to(rightmost_old_bar, RIGHT).align_to(
                rightmost_old_bar, UP
            )
            for i, sub_bar in enumerate(new_bars[1:], start=1):
                sub_bar.next_to(new_bars[i - 1], LEFT, buff=0)

        # Animate the splitting of the old bar into sub-bars
        self.play(
            *[bar.animate.set_opacity(self.ghost_opacity) for bar in old_bars],
            *[FadeIn(bar) for bar in new_bars],
            FadeIn(exhausted_bar),
        )
        # Animate the exhaustion of votes and moving the sub-bars to the destination bars
        self.play(Uncreate(exhausted_bar), *transformations)

    def _animate_elimination_offscreen(self, event: EliminationOffscreenEvent) -> None:
        """
        Animate a round in which offscreen candidates were eliminated.

        Args:
            event (EliminationOffscreenEvent): The event to be animated.
        """
        destinations = event.support_transferred
        # Create short bars that will begin offscreen
        new_bars = []
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width=self._support_to_bar_width(votes),
                height=self.bar_height,
                color=ManimColor(self.color_palette.offscreen_candidate_fill),
                fill_color=ManimColor(self.color_palette.offscreen_candidate_fill),
                fill_opacity=self.bar_opacity,
            )
            self.candidate_dict[destination]["support"] += votes
            new_bars.append(sub_bar)
            transformations.append(
                new_bars[-1].animate.next_to(
                    self.candidate_dict[destination]["bars"][-1], RIGHT, buff=0
                )
            )  # The sub-bars will move to be next to the bars of their destination candidates
            self.candidate_dict[destination]["bars"].append(sub_bar)

        # Place these bars well offscreen
        for bar in new_bars:
            bar.to_edge(DOWN).shift((self.bar_height + 2) * DOWN)

        # Animate the exhaustion of votes and moving the sub-bars to the destination bars
        if len(transformations) > 0:
            self.play(*transformations)

    def _support_to_bar_width(self, support: float) -> float:
        """
        Convert a number of votes to the width of a bar in manim coordinates
        representing that many votes.

        Args:
            support (float): A number of votes.

        Returns:
            float: The width, in manim coordinates, of a bar representing the support.
        """
        return self.width * support / self.max_support
