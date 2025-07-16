from copy import deepcopy
from manim import *

class ElectionScene(Scene):
    colors = [color.ManimColor(hex) for hex in [
        "#16DEBD",
        "#163EDE",
        "#9F34F6",
        "#FF6F00",
        "#8F560C",
        "#E2AD00",
        "#8AD412"]]
    bar_color = LIGHT_GRAY
    bar_opacity = 1
    win_bar_color = GREEN
    eliminated_bar_color = RED
    ghost_opacity = 0.3
    ticker_tape_height = 2
    offscreen_sentinel = '__offscreen__'
    offscreen_candidate_color = GRAY
    title_font_size = 48

    def __init__(self, candidates, rounds, title=None):
        Scene.__init__(self)
        self.candidates = candidates
        self.rounds = rounds
        self.title = title

        self.width = 8
        self.bar_height = 3.5 / len(self.candidates)
        self.font_size = 3*40 / len(self.candidates)
        self.bar_buffer_size = 1 / len(self.candidates)
        self.strikethrough_width = self.font_size / 5
        self.max_support = 1.1 * max([round['quota'] for round in self.rounds])

        self.quota_line = None
        self.ticker_tape_line = None


    def construct(self):
        if self.title is not None:
            self.__draw_title__(self.title)
        self.__draw_initial_bars__()
        self.__initialize_ticker_tape__()
        
        # Go round by round and animate the events
        for round_number, round in enumerate(self.rounds):     
            print(f"Animating round: {round['event']}, {round['candidate']}")

            # Remove the candidate from the candidate list
            if round['candidate'] == self.offscreen_sentinel: #If the eliminated candidate is offscreen
                eliminated_candidate = None
            else:
                eliminated_candidate = self.candidates[round['candidate']]
                self.candidates.pop(round['candidate'])

            self.wait(2)

            # Draw or move the quota line
            self.__update_quota_line__(round['quota'])

            self.__ticker_animation_shift__(round)
            self.__ticker_animation_highlight__(round)

            if round['event'] == 'elimination':
                if eliminated_candidate is None: #Offscreen candidate eliminated
                    self.__animate_elimination_offscreen__(round)
                else: #Onscreen candidate eliminated
                    self.__animate_elimination__(eliminated_candidate, round)
            elif round['event'] == 'win':
                self.__animate_win__(eliminated_candidate, round)
            else:
                raise Exception(f"Event type {round['event']} not recognized.")
    
        self.wait(2)

    def __draw_title__(self, message):
        text = Tex(
            r"{7cm}\centering " + message,
            tex_environment="minipage"
        ).scale_to_fit_width(10)
        self.play(Create(text))
        self.wait(3)
        self.play(Uncreate(text))

    def __draw_initial_bars__(self):
        # Sort candidates by starting first place votes
        sorted_candidates = sorted(self.candidates.keys(), key=lambda x: self.candidates[x]['support'], reverse=True)

        # Create bars
        for i, name in enumerate(sorted_candidates):
            color = self.colors[i % len(self.colors)]
            self.candidates[name]['color'] = color
            self.candidates[name]['bars'] = [Rectangle(
                width=self.__support_to_bar_width__(self.candidates[name]['support']),
                height=self.bar_height,
                color=self.bar_color,
                fill_color=color,
                fill_opacity=self.bar_opacity
                )]
        
        self.candidates[sorted_candidates[0]]['bars'][0].to_edge(UP) #First candidate goes at the top
        for i, name in enumerate(sorted_candidates[1:], start=1):
            self.candidates[name]['bars'][0].next_to(
                self.candidates[sorted_candidates[i-1]]['bars'][0],
                DOWN,
                buff=self.bar_buffer_size
            ).align_to(
                self.candidates[sorted_candidates[0]]['bars'][0],
                LEFT
            ) #The rest of the candidates go below

        # Draw a large black rectangle for the background so that the ticker tape vanishes behind it
        background = Rectangle(width = self.camera.frame_width,
                               height = self.camera.frame_height,
                               fill_color = BLACK,
                               color = BLACK,
                               fill_opacity = 1).shift(UP * self.ticker_tape_height).set_z_index(-1)

        # Create and place candidate names
        for name in sorted_candidates:
            candidate = self.candidates[name]
            candidate['name_text'] = Text(
                name,
                font_size=self.font_size,
                color=candidate['color']
                ).next_to(candidate['bars'][0], LEFT, buff=0.2)

        # Draw the bars and names
        self.play(*[FadeIn(self.candidates[name]['name_text']) for name in sorted_candidates], FadeIn(background))
        self.play(*[Create(self.candidates[name]['bars'][0]) for name in sorted_candidates])

    def __initialize_ticker_tape__(self):
        line_length = self.camera.frame_width
        ticker_line = Line(start = [-line_length/2, 0., 0.],
                           end = [line_length/2, 0., 0.])
        ticker_line.to_edge(DOWN, buff=0).shift(UP * self.ticker_tape_height)
        ticker_line.set_z_index(2) #Keep this line in front of the bars and the quota line
        self.ticker_tape_line = ticker_line

        for i, round in enumerate(self.rounds):
            new_message = Text(
                round['message'],
                font_size = 24,
                color=DARK_GRAY)
            round['ticker_text'] = new_message
            if i == 0:
                new_message.to_edge(DOWN,buff=0).shift(DOWN)
            else:
                new_message.next_to(self.rounds[i-1]['ticker_text'], DOWN)
            new_message.set_z_index(-2)
        
        self.play(Create(ticker_line))
        self.play(*[Create(round['ticker_text']) for round in self.rounds])

    def __ticker_animation_shift__(self, round):
        shift_to_round = round['ticker_text'].animate.to_edge(DOWN, buff=0).shift(UP * self.ticker_tape_height/3)
        drag_other_messages = [
            MaintainPositionRelativeTo(other_round['ticker_text'], round['ticker_text'])
            for other_round in self.rounds if not other_round == round
        ]
        self.play(shift_to_round, *drag_other_messages)

    def __ticker_animation_highlight__(self, round):
        highlight_message = round['ticker_text'].animate.set_color(WHITE)
        unhighlight_other_messages = [
            other_round['ticker_text'].animate.set_color(DARK_GRAY)
            for other_round in self.rounds if not other_round == round
        ]
        self.play(highlight_message, *unhighlight_other_messages)

        
    def __update_quota_line__(self, quota):
        some_candidate = list(self.candidates.values())[0]

        if not self.quota_line:
            # If the quota line doesn't exist yet, draw it.
            line_top = self.camera.frame_height / 2
            line_bottom = self.ticker_tape_line.get_top()[1]
            self.quota_line = Line(
                start = [0., line_top, 0.],
                end = [0., line_bottom, 0.],
                color=self.win_bar_color)
            self.quota_line.align_to(some_candidate['bars'][0], LEFT)
            self.quota_line.shift((self.width * quota/self.max_support) * RIGHT)
            self.quota_line.set_z_index(1) # Keep the quota line in the front

            self.play(Create(self.quota_line))
            self.wait(2)
        else:
            self.play(self.quota_line.animate.align_to(some_candidate['bars'][0], LEFT).shift((self.width * quota/self.max_support) * RIGHT))


    def __animate_win__(self, from_candidate, round):
        assert from_candidate is not None

        #Box the winner's name
        winner_box = SurroundingRectangle(from_candidate['name_text'], color=GREEN, buff=0.1)

        #Save the winner's votes as green votes
        winner_rectangle = Rectangle(
            width=self.__support_to_bar_width__(round['quota']),
            height=self.bar_height,
            color=self.bar_color,
            fill_color=self.win_bar_color,
            fill_opacity=self.bar_opacity
        ).align_to(from_candidate['bars'][0], LEFT).align_to(from_candidate['bars'][0], UP)
        
        # Create short bars that will represent the transferred votes
        destinations = round['support_transferred']
        candidate_color = from_candidate['color']
        old_bars = from_candidate['bars']
        new_bars = []
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width = self.__support_to_bar_width__(votes),
                height=self.bar_height,
                color=self.bar_color, 
                fill_color=candidate_color,
                fill_opacity=self.bar_opacity)
            self.candidates[destination]['support'] += votes
            new_bars.append(sub_bar)
            transformations.append(
                new_bars[-1].animate.next_to(
                    self.candidates[destination]['bars'][-1],
                    RIGHT,
                    buff=0)
            ) #The sub-bars will move to be next to the bars of their destination candidates
            self.candidates[destination]['bars'].append(sub_bar)
        # Create a final short bar representing the exhausted votes
        exhausted_votes = from_candidate['support'] - round['quota'] - np.sum(list(destinations.values()))
        exhausted_bar = Rectangle(
            width = self.__support_to_bar_width__(exhausted_votes),
            height=self.bar_height,
            color=self.bar_color, 
            fill_color=candidate_color,
            fill_opacity=self.bar_opacity)
        assert self.quota_line is not None
        exhausted_bar.align_to(self.quota_line, LEFT).align_to(old_bars[0], UP)

        # The short bars should start at the right side of the old bar
        if len(new_bars) > 0:
            new_bars[0].align_to(old_bars[-1], RIGHT).align_to(old_bars[-1],UP)
            for i, sub_bar in enumerate(new_bars[1:], start=1):
                sub_bar.next_to(new_bars[i-1], LEFT, buff=0)


        # Animate the box around the candidate name and the message text
        self.play(Create(winner_box))
        # Animate the splitting of the old bar into sub-bars
        self.play(*[FadeOut(bar) for bar in old_bars],
                  *[FadeIn(bar) for bar in new_bars],
                  FadeIn(exhausted_bar),
                  FadeIn(winner_rectangle))
        # Animate moving the sub-bars to the destination bars, and the destruction of the exhausted votes
        transformations.append(Uncreate(exhausted_bar))
        if len(transformations) > 0:
            self.play(*transformations)


    def __animate_elimination__(self, from_candidate, round):
        destinations = round['support_transferred']

        #Cross out the candidate name
        cross = Line(
            from_candidate['name_text'].get_left(),
            from_candidate['name_text'].get_right(),
            color=RED,
            )
        cross.set_stroke(width=self.strikethrough_width)
        self.play(Create(cross))

        # Create short bars that will replace the candidate's current bars
        candidate_color = from_candidate['color']
        old_bars = from_candidate['bars']
        new_bars = [] #The bits to be redistributed
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width = self.__support_to_bar_width__(votes),
                height=self.bar_height,
                color=self.bar_color, 
                fill_color=candidate_color,
                fill_opacity=self.bar_opacity)
            self.candidates[destination]['support'] += votes
            new_bars.append(sub_bar)
            transformations.append(
                new_bars[-1].animate.next_to(
                    self.candidates[destination]['bars'][-1],
                    RIGHT,
                    buff=0)
            ) #The sub-bars will move to be next to the bars of their destination candidates
            self.candidates[destination]['bars'].append(sub_bar)
        # Create a final short bar representing the exhausted votes
        exhausted_votes = from_candidate['support'] - np.sum(list(destinations.values()))
        exhausted_bar = Rectangle(
            width = self.__support_to_bar_width__(exhausted_votes),
            height=self.bar_height,
            color=self.bar_color, 
            fill_color=candidate_color,
            fill_opacity=self.bar_opacity)
        exhausted_bar.align_to(old_bars[0], LEFT).align_to(old_bars[0], UP)

        if len(new_bars) > 0:
            # The short bars should start in the same place as the old bars. Place them.
            rightmost_old_bar = old_bars[-1]
            new_bars[0].align_to(rightmost_old_bar, RIGHT).align_to(rightmost_old_bar,UP)
            for i, sub_bar in enumerate(new_bars[1:], start=1):
                sub_bar.next_to(new_bars[i-1], LEFT, buff=0)
        

        # Animate the splitting of the old bar into sub-bars
        self.play(*[
            bar.animate.set_opacity(self.ghost_opacity)
            for bar in old_bars],
                  *[FadeIn(bar) for bar in new_bars],
                  FadeIn(exhausted_bar))
        # Animate the exhaustion of votes and moving the sub-bars to the destination bars
        self.play(Uncreate(exhausted_bar), *transformations)

    

    def __animate_elimination_offscreen__(self, round):
        destinations = round['support_transferred']

        # Create short bars that will replace the candidate's current bars
        new_bars = [] #The bits to be redistributed
        transformations = []
        for destination, votes in destinations.items():
            if votes <= 0:
                continue
            sub_bar = Rectangle(
                width = self.__support_to_bar_width__(votes),
                height=self.bar_height,
                color=self.bar_color, 
                fill_color=self.offscreen_candidate_color,
                fill_opacity=self.bar_opacity,
            )
            self.candidates[destination]['support'] += votes
            new_bars.append(sub_bar)
            transformations.append(
                new_bars[-1].animate.next_to(
                    self.candidates[destination]['bars'][-1],
                    RIGHT,
                    buff=0)
            ) #The sub-bars will move to be next to the bars of their destination candidates
            self.candidates[destination]['bars'].append(sub_bar)
        
        for bar in new_bars:
            bar.to_edge(DOWN).shift((self.bar_height + 2)*DOWN)

        # Animate the exhaustion of votes and moving the sub-bars to the destination bars
        self.play(*transformations)

            
    def __rescale_bars__(self):
        #Re-scale the bars so they're not out of frame
        self.max_support = max([candidate['support'] for candidate in self.candidates.values()])

        transformations = []
        for candidate in self.candidates.values():
            old_bar = candidate['bar']
            new_bar = Rectangle(
                width=self.__support_to_bar_width__(candidate['support']),
                height=self.bar_height,
                color=self.bar_color, 
                fill_color=candidate['color'],
                fill_opacity=self.bar_opacity).next_to(candidate['name_text'], RIGHT)
            bar_shortening_transformation = Transform(
                old_bar,
                new_bar,
            )
            transformations.append(bar_shortening_transformation) 
        self.play(*[transformations])

    def __support_to_bar_width__(self, support):
        return self.width * support / self.max_support



class STVAnimation():
    def __init__(self, election, title = None):
        self.candidates = self.__make_candidate_dict__(election)
        self.rounds = self.__make_event_list__(election)
        self.title = title

    def __make_candidate_dict__(self, election): #this creates the candidate dictionary
        viz_candidates = {name : {'support' : support} for name, support in election.election_states[0].scores.items()}
        return viz_candidates
    
    def __get_transferred_votes__(self, election, candidate, round_number):
        prev_state = election.election_states[round_number-1]
        current_state = election.election_states[round_number]
        prev_score = int(prev_state.scores[candidate])
        current_score = int(current_state.scores[candidate])
        return current_score - prev_score

    def __make_event_list__(self, election):
        events = []
        for round_number, election_round in enumerate(election.election_states[1:], start = 1): #Nothing happens in election round 0

            remaining_candidates = []
            for fset in election_round.remaining:
                remaining_candidates += list(fset)

            elected_candidates = []
            for fset in election_round.elected:
                if len(fset) == 0:
                    break
                name, = fset
                elected_candidates.append(name)

            eliminated_candidates = []
            for fset in election_round.eliminated:
                if len(fset) == 0:
                    break
                name, = fset
                eliminated_candidates.append(name)
            
            for name in elected_candidates[:1]: #TODO This code currently skips all but the first elected candidate. It will take some work to deduce how support transferred if multiple candidates were elected at once.
                #Create a new event for each candidate elected this round
                event_type = 'win'
                message = f'Round {round_number}: {name} Elected'
                support_transferred = {}
                if round_number < len(election): #If it's the last round, don't worry about the transferred votes
                    for candidate in remaining_candidates:
                        if candidate == name:
                            continue
                        support_transferred[candidate] = self.__get_transferred_votes__(election, candidate, round_number)
                events.append(dict(
                    event = event_type,
                    candidate = name,
                    support_transferred = support_transferred,
                    quota = election.threshold,
                    message = message
                ))
            
            for name in eliminated_candidates[:1]: #TODO This code currently skips all but the first eliminated candidate. It will take some work to deduce how support transferred if multiple candidates were eliminated at once.
                #Create a new event for each candidate eliminated this round
                event_type = 'elimination'
                message = f'Round {round_number}: {name} Eliminated'
                support_transferred = {}

                if round_number < len(election): #If it's the last round, don't worry about the transferred votes
                    for candidate in remaining_candidates:
                        if candidate == name:
                            continue
                        support_transferred[candidate] = self.__get_transferred_votes__(election, candidate, round_number)
                events.append(dict(
                    event = event_type,
                    candidate = name,
                    support_transferred = support_transferred,
                    quota = election.threshold,
                    message = message
                ))
        return events

    def render(self, preview=False):
        manimation = ElectionScene(self.candidates, self.rounds, title=self.title)
        manimation.render(preview=preview)
