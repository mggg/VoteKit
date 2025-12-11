from copy import deepcopy
import manim  # type: ignore
from manim import (
    Rectangle,
    SurroundingRectangle,
    Line,
    Create,
    Uncreate,
    FadeIn,
    FadeOut,
    Text,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)
from .cleaning.rank_ballots_cleaning import (
    condense_rank_ballot,
    remove_cand_rank_ballot,
)
from .utils import ballots_by_first_cand
from .elections.election_types.ranking.stv import STV
from typing import Literal, List, Optional, Sequence, Mapping
from collections import defaultdict
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
        support_transferred (Mapping[str,float]): A dictionary mapping names of candidates to the amount of support they received from the elimination.
        round_number (int): The round of the election process associated to this event.
    """

    candidate: str
    support_transferred: Mapping[str, float]
    round_number: int

    def get_message(self) -> str:
        return f"Round {self.round_number}: {self.candidate} eliminated."


@dataclass
class EliminationOffscreenEvent(AnimationEvent):
    """
    An animation event representing some number of rounds in which offscreen candidates were eliminated.

    Attributes:
        support_transferred (Mapping[str,float]): A dictionary mapping names of candidates to the total amount of support they received from the eliminations.
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
        candidates (str): The names of the elected candidates.
        support_transferred (Mapping[str, Mapping[str,float]]): A dictionary mapping pairs of candidate names to the amount of support transferred between them this round. For instance, if ``c1`` was elected this round, then ``support_transferred[c1][c2]`` will represent the amount of support that ran off from ``c1`` to candidate ``c2``.
        round_number (int): The round of the election process associated to this event.
    """

    candidates: Sequence[str]
    support_transferred: Mapping[str, Mapping[str, float]]
    round_number: int

    def get_message(self) -> str:
        candidate_string = ", ".join(self.candidates)
        return f"Round {self.round_number}: {candidate_string} elected."


class STVAnimation:
    """
    A class which creates round-by-round animations of STV elections.

    Args:
        election (STV): An STV election to animate.
        title (str, optional): Text to be displayed at the beginning of the animation as a title screen. If ``None``, the title screen will be skipped. Defaults to ``None``.
        focus (List[str], optional): A list of names of candidates that should appear on-screen. This is useful for elections with many candidates. Note that any candidates that won the election are on-screen automatically, so passing an empty list will result in only elected candidates appearing on-screen. If ``None``, focus only the elected candidates. Defaults to ``None``.
    """

    def __init__(
        self,
        election: STV,
        title: Optional[str] = None,
        focus: Optional[List[str]] = None,
    ):
        if focus is None:
            focus = []
        self.focus = focus
        elected_candidates = [c for s in election.get_elected() for c in s]
        focus += [name for name in elected_candidates if name not in focus]
        self.candidate_dict = self._make_candidate_dict(election)
        self.events = self._make_event_list(election)
        if len(self.candidate_dict) == 0:
            raise ValueError("Tried creating animation with no candidates.")
        if len(self.events) == 0:
            raise ValueError("Tried creating animation with no animation event.")
        self.title = title

    def _make_candidate_dict(self, election: STV) -> dict[str, dict[str, object]]:
        """
        Create the dictionary of candidates and relevant facts about each one.

        Args:
            election (STV): An STV election from which to extract the candidates.

        Returns:
            dict[str, dict[str,object]]: A dictionary whose keys are candidate names and whose values are themselves dictionaries with details about each candidate.
        """
        candidate_dict: dict[str, dict[str, object]] = {
            name: {"support": support}
            for name, support in election.election_states[0].scores.items()
            if name in self.focus
        }
        return candidate_dict

    def _make_event_list(self, election: STV) -> List[AnimationEvent]:
        """
        Processes an STV election into a condensed list of animation events which roughly correspond to election rounds.

        Args:
            election (STV): The STV election to process.

        Returns:
            List[AnimationEvent]: A list of the events of the election which are worthy of animation.

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
                events.append(
                    WinEvent(
                        quota=election.threshold,
                        candidates=elected_candidates,
                        support_transferred=support_transferred,
                        round_number=round_number,
                    )
                )
            elif len(eliminated_candidates) > 0:  # Elimination round
                if len(eliminated_candidates) > 1:
                    raise ValueError(
                        f"Multiple-elimination rounds not supported. At most one candidate should be eliminated in each round. Candidates eliminated in round {round_number}: {eliminated_candidates}."
                    )
                eliminated_candidate = eliminated_candidates[0]
                support_transferred = self._get_transferred_votes(
                    election, round_number, eliminated_candidates, "elimination"
                )
                if eliminated_candidate in self.focus:
                    events.append(
                        EliminationEvent(
                            quota=election.threshold,
                            candidate=eliminated_candidate,
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
        from_candidates: List[str],
        event_type: Literal["win", "elimination"],
    ) -> dict[str, dict[str, float]]:
        """
        Compute the number of votes transferred from each elected or eliminated candidate to each remaining candidate.

        Args:
            election (STV): The election.
            round_number (int): The number of the round in question.
            from_candidates (List[str]): A list of the names of the elected or eliminated candidates.
            event_type (str): ``"win"`` if candidates were elected this round, ``"elimination"`` otherwise.

        Returns:
            dict[str, dict[str, float]]: A nested dictionary. If ``d`` is the return value, ``c1`` was a candidate eliminated this round, and ``c2`` is a remaining candidate, then ``d[c1][c2]`` will be the total support transferred this round from candidate ``c1`` to candidate ``c2``.

        Notes:
            This function supports the election, but not the elimination, of multiple candidates in one round. If ``event_type`` is ``"elimination"`` then ``from_candidates`` should have length 1.
        """
        prev_profile, prev_state = election.get_step(round_number - 1)
        current_state = election.election_states[round_number]

        transfers: dict[str, dict[str, float]] = {}
        if event_type == "elimination":
            assert len(from_candidates) == 1
            from_candidate = from_candidates[0]
            transfers = {from_candidate: {}}
            for to_candidate in [
                c for s in current_state.remaining for c in s if c in self.focus
            ]:
                prev_score = prev_state.scores[to_candidate]
                current_score = current_state.scores[to_candidate]
                transfers[from_candidate][to_candidate] = current_score - prev_score
        elif event_type == "win":
            ballots_by_fpv = ballots_by_first_cand(prev_profile)
            for from_candidate in from_candidates:
                new_ballots = election.transfer(
                    from_candidate,
                    prev_state.scores[from_candidate],
                    ballots_by_fpv[from_candidate],
                    election.threshold,
                )
                clean_ballots = [
                    condense_rank_ballot(remove_cand_rank_ballot(from_candidates, b))
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

                transfers[from_candidate] = transfer_weights_from_candidate

        return transfers

    def _condense_offscreen_events(
        self, events: List[AnimationEvent]
    ) -> List[AnimationEvent]:
        """
        Take a list of events and condense any consecutive offscreen events into one summarizing event. For instance, if ``events`` contians three offscreen eliminations in a row, this function will condense them into one offscreen elimination of three candidates.

        Args:
            events (List[AnimationEvent]): A list of animation events to be condensed.

        Returns:
            List[AnimationEvent]: A condensed list of animation events.
        """
        return_events: List[AnimationEvent] = [events[0]]
        for event in events[1:]:
            if isinstance(return_events[-1], EliminationOffscreenEvent) and isinstance(
                event, EliminationOffscreenEvent
            ):
                return_events[-1] = self._compose_offscreen_eliminations(
                    return_events[-1], event
                )
            else:
                return_events.append(event)
        return return_events

    def _compose_offscreen_eliminations(
        self, event1: EliminationOffscreenEvent, event2: EliminationOffscreenEvent
    ) -> EliminationOffscreenEvent:
        """
        Take two offscreen eliminations and "compose" them into a single offscreen elimination event summarizing both.

        Args:
            event1 (EliminationOffscreenEvent): The first offscreen elimination event to compose.
            event2 (EliminationOffscreenEvent): The second offscreen elimination event to compose.

        Returns:
            EliminationOffscreenEvent: One offscreen elimination event summarizing ``event1`` and ``event2``.
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

    def render(self, preview: bool = False) -> None:
        """
        Renders the STV animation using Manim.

        The completed video will appear in the directory ``media/videos``.

        Args:
            preview (bool, optional): If ``True``, display the result in a video player immediately upon completing the render. Defaults to False.
        """
        manimation = ElectionScene(
            deepcopy(self.candidate_dict), deepcopy(self.events), title=self.title
        )
        manimation.render(preview=preview)


class ElectionScene(manim.Scene):
    """
    Class for Manim animation of an STV election.

    Notes:
        This class is instantiated by the class ``STVAnimation``. It should not be instantiated directly.

    Args:
        candidate_dict (dict[str,dict]): A dictionary mapping each candidate to a dictionary of attributes of the candidate.
        events (List[AnimationEvent]): A list of animation events to be constructed and rendered.
        title (str): A string to be displayed at the beginning of the animation as a title screen. If ``None``, the animation will skip the title screen.
    """

    colors = [
        manim.color.ManimColor(hex)
        for hex in [
            "#16DEBD",
            "#163EDE",
            "#9F34F6",
            "#FF6F00",
            "#8F560C",
            "#E2AD00",
            "#8AD412",
        ]
    ]
    bar_color = manim.LIGHT_GRAY
    bar_opacity = 1
    win_bar_color = manim.GREEN
    eliminated_bar_color = manim.RED
    ghost_opacity = 0.3
    ticker_tape_height = 2
    offscreen_sentinel = "__offscreen__"
    offscreen_candidate_color = manim.GRAY
    title_font_size = 48
    name_bar_spacing = 0.2
    winner_box_buffer = 0.1

    def __init__(
        self,
        candidate_dict: dict[str, dict],
        events: List[AnimationEvent],
        title: Optional[str] = None,
    ):
        super().__init__()
        self.candidate_dict = candidate_dict
        self.events = events
        self.title = title

        self.width = 8
        self.bar_height = 3.5 / len(self.candidate_dict)
        self.font_size = 3 * 40 / len(self.candidate_dict)
        self.bar_opacity = 1
        self.bar_buffer_size = self.bar_height
        self.strikethrough_thickness = self.font_size / 5
        self.max_support = 1.1 * max([event.quota for event in self.events])

        self.quota_line = None
        self.ticker_tape_line = None
        self.ticker_tape: List[Text] = []

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
        text = manim.Tex(
            r"{7cm}\centering " + message, tex_environment="minipage"
        ).scale_to_fit_width(
            10
        )  # We do this one with a TeX minipage to get the text to wrap if it's too long.
        self.play(Create(text))
        self.wait(3)
        self.play(Uncreate(text))

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

        # Assign colors
        for i, name in enumerate(sorted_candidates):
            color = self.colors[i % len(self.colors)]
            self.candidate_dict[name]["color"] = color

        # Create candidate name text
        for i, name in enumerate(sorted_candidates):
            candidate = self.candidate_dict[name]
            candidate["name_text"] = Text(
                name, font_size=self.font_size, color=candidate["color"]
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
        del group

        # Create bars
        for candidate in self.candidate_dict.values():
            candidate["bars"] = [
                Rectangle(
                    width=self._support_to_bar_width(candidate["support"]),
                    height=self.bar_height,
                    color=self.bar_color,
                    fill_color=candidate["color"],
                    fill_opacity=self.bar_opacity,
                ).next_to(candidate["name_text"], RIGHT, buff=self.name_bar_spacing)
            ]

        # Draw a large black rectangle for the background so that the ticker tape vanishes behind it
        frame_width = manim.config.frame_width
        frame_height = manim.config.frame_height
        background = (
            Rectangle(
                width=frame_width,
                height=frame_height,
                fill_color=manim.BLACK,
                color=manim.BLACK,
                fill_opacity=1,
            )
            .shift(UP * self.ticker_tape_height)
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
        """Instantiate and draw the ticker tape line and text."""
        line_length = manim.config.frame_width
        ticker_line = Line(
            start=LEFT * line_length / 2,
            end=RIGHT * line_length / 2,
        )
        ticker_line.to_edge(DOWN, buff=0).shift(UP * self.ticker_tape_height)
        ticker_line.set_z_index(
            2
        )  # Keep this line in front of the bars and the quota line
        self.ticker_tape_line = ticker_line
        self.ticker_tape = []
        for i, event in enumerate(self.events):
            new_message = Text(event.get_message(), font_size=24, color=manim.DARK_GRAY)
            if i == 0:
                new_message.to_edge(DOWN, buff=0).shift(DOWN)
            else:
                new_message.next_to(self.ticker_tape[-1], DOWN)
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
            manim.WHITE
        )
        unhighlight_other_messages = [
            self.ticker_tape[i].animate.set_color(manim.DARK_GRAY)
            for i in range(len(self.ticker_tape))
            if i != event_number
        ]
        self.play(highlight_message, *unhighlight_other_messages)

    def _update_quota_line(self, quota: float) -> None:
        """
        Update the position of the quota line to reflect the given quota. If no quota line exists, create it and animate its creation.

        Args:
            quota (float): The threshold number of votes necessary to be elected in the current round.
        """
        some_candidate = list(self.candidate_dict.values())[0]
        if not self.quota_line:
            # If the quota line doesn't exist yet, draw it.
            assert self.ticker_tape_line is not None
            line_bottom = self.ticker_tape_line.get_top()[1]
            line_top = manim.config.frame_height / 2
            self.quota_line = Line(
                start=UP * line_top, end=UP * line_bottom, color=self.win_bar_color
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

    def _animate_win(self, from_candidates: dict[str, dict], event: WinEvent) -> None:
        """
        Animate a round in which one or more candidates are elected.

        Args:
            from_candidates (dict[str,dict]): A dictionary in which the keys are the candidates elected this round and the values are dictionaries recording the candidate's attributes.
            event (WinEvent): The event to be animated.
        """
        # Box the winners' names
        winner_boxes = [
            SurroundingRectangle(
                from_candidate["name_text"],
                color=manim.GREEN,
                buff=self.winner_box_buffer,
            )
            for from_candidate in from_candidates.values()
        ]

        # Animate the box around the candidate name and the message text
        self.play(*[Create(box) for box in winner_boxes])

        # Create and animate the subdivision and redistribution of winners' leftover votes
        for from_candidate_name, from_candidate in from_candidates.items():
            old_bars: List[Rectangle] = from_candidate["bars"]
            new_bars: List[Rectangle] = []
            transformations = []
            destinations = event.support_transferred[from_candidate_name]
            candidate_color = from_candidate["color"]

            used_votes = min(event.quota, from_candidate["support"])
            winner_bar = (
                Rectangle(
                    width=self._support_to_bar_width(used_votes),
                    height=self.bar_height,
                    color=self.bar_color,
                    fill_color=self.win_bar_color,
                    fill_opacity=self.bar_opacity,
                )
                .align_to(from_candidate["bars"][0], LEFT)
                .align_to(from_candidate["bars"][0], UP)
            )

            # Create a sub-bar for each destination
            for destination, votes in destinations.items():
                if votes <= 0:
                    continue
                sub_bar = Rectangle(
                    width=self._support_to_bar_width(votes),
                    height=self.bar_height,
                    color=self.bar_color,
                    fill_color=candidate_color,
                    fill_opacity=self.bar_opacity,
                )
                # The first sub-bar should start at the right end of the eliminated candidate's stack. The rest should be arranged to the left of that one
                if len(new_bars) == 0:
                    sub_bar.align_to(from_candidate["bars"][-1], RIGHT).align_to(
                        from_candidate["bars"][-1], UP
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
                from_candidate["support"]
                - used_votes
                - sum(list(destinations.values()))
            )
            exhausted_bar = Rectangle(
                width=self._support_to_bar_width(exhausted_votes),
                height=self.bar_height,
                color=self.bar_color,
                fill_color=candidate_color,
                fill_opacity=self.bar_opacity,
            )
            assert self.quota_line is not None
            if len(new_bars) > 0:
                exhausted_bar.next_to(new_bars[0], RIGHT, buff=0)
            else:
                exhausted_bar.next_to(winner_bar, RIGHT, buff=0)
            exhausted_bar.set_z_index(1)
            transformations.append(Uncreate(exhausted_bar))

            # Animate the splitting of the old bar into the new sub_bars
            self.play(
                *[FadeOut(bar) for bar in old_bars],
                FadeIn(winner_bar),
                *[FadeIn(bar) for bar in new_bars],
                FadeIn(exhausted_bar),
            )

            # Animate moving the sub-bars to the destination bars, and the destruction of the exhausted votes
            if len(transformations) > 0:
                self.play(*transformations)

    def _animate_elimination(
        self, from_candidates: dict[str, dict], event: EliminationEvent
    ) -> None:
        """
        Animate a round in which a candidate was eliminated.

        Args:
            from_candidates (dict[str,dict]): A dictionary in which the keys are the candidates eliminated this round and the values are dictionaries recording the candidate's attributes.
            event (EliminationEvent): The event to be animated.

        Notes:
            While the interface supports multiple candidate eliminations in one round for future extensibility, this function currently only supports elimination of one candidate at a time. The from_candidates argument should have exactly one entry.

        Raises:
            ValueError: If the length of ``from_candidates`` is not 1.
        """
        num_eliminated_candidates = len(list(from_candidates.values()))
        if num_eliminated_candidates != 1:
            raise ValueError(
                f"Elimination round animations only support one eliminated candidate at a time. Attempted to animate {num_eliminated_candidates} eliminations in one election round."
            )
        del num_eliminated_candidates

        from_candidate = list(from_candidates.values())[0]
        destinations = event.support_transferred

        # Cross out the candidate name
        cross = Line(
            from_candidate["name_text"].get_left(),
            from_candidate["name_text"].get_right(),
            color=manim.RED,
        )
        cross.set_stroke(width=self.strikethrough_thickness)
        self.play(Create(cross))

        # Create short bars that will replace the candidate's current bars
        candidate_color = from_candidate["color"]
        old_bars = from_candidate["bars"]
        new_bars = []  # The bits to be redistributed
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width=self._support_to_bar_width(votes),
                height=self.bar_height,
                color=self.bar_color,
                fill_color=candidate_color,
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
        exhausted_votes = from_candidate["support"] - sum(list(destinations.values()))
        exhausted_bar = Rectangle(
            width=self._support_to_bar_width(exhausted_votes),
            height=self.bar_height,
            color=self.bar_color,
            fill_color=candidate_color,
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
            event (EliminationOffscreenEvent) The event to be animated.
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
                color=self.bar_color,
                fill_color=self.offscreen_candidate_color,
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

        for bar in new_bars:
            bar.to_edge(DOWN).shift((self.bar_height + 2) * DOWN)

        # Animate the exhaustion of votes and moving the sub-bars to the destination bars
        if len(transformations) > 0:
            self.play(*transformations)

    def _support_to_bar_width(self, support: float) -> float:
        """
        Convert a number of votes to the width of a bar in manim coordinates representing that many votes.

        Args:
            support (float): A number of votes.

        Returns:
            float: The width, in manim coordinates, of a bar representing the support.
        """
        return self.width * support / self.max_support
