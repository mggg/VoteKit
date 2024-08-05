---
title: "VoteKit: A Python package for computational social choice research"
tags:
  - Python
authors: 
  - name: Christopher Donnay
  - name: Moon Duchin
  - name: Jack Gibson
  - name: Zach Glaser
  - name: Andrew Hong
  - name: Malavika Mukundan
  - name: Jennifer Wang
affiliations:
  - name: Metric Geometry and Gerrymandering Group (MGGG)
    index: 1
bibliography: paper.bib
---

# Summary

The scholarly study of elections, known as *social choice theory*,
centers on the provable properties of voting rules. Practical work
in democracy reform focuses on designing or selecting systems of
election to produce electoral outcomes that promote legitimacy and
broad-based representation. For instance, the dominant electoral system
in the United States is a one-person-one-vote/winner-take-all system,
sometimes known as PSMD (plurality in single member districts); today,
there is considerable reform momentum in favor of ranked choice voting
because it is thought to mitigate the effects of vote-splitting and
to strengthen prospects for minority representation, among other
claimed properties.[^1]  Across the world,
systems of election---and prospects for system change---vary
substantially. From both a scholarly and a practical perspective, many
questions arise about comparing the properties and tendencies of diverse
systems of election in a rigorous manner.

`VoteKit` [@VoteKit] is a Python package designed to facilitate just that
kind of analysis, bringing together multiple types of functionality.
Users can:

1.  Create synthetic *preference profiles* (collections of ballots) with
    a choice of generative models and behavioral parameters;

2.  Read in real-world *cast vote records* (CVRs) as observed examples
    of preference profiles; clean and process ballots, including by
    deduplication and handling of undervotes and overvotes;

3.  Run a variety of *voting rules* to ingest preference profiles and
    output winner sets and rankings; and

4.  Produce a wide range of *summary statistics* and *data
    visualizations* to compare and analyze profiles and election
    outcomes.

# Statement of need

Social choice theory grew out of welfare economics in the mid-twentieth
century and has been recognized as a deep and highly applicable area of
economic theory, forming part of the basis for at least four Nobel Prize
awards.[^2] Since the 1990s, a new fusion of economics and computer
science has emerged under the name of *computational social choice*,
studying questions of complexity and design and further advancing the
axiomatic study of elections.[^3] But most of these innovations have
been highly abstract, and there has been a significant gap in the
literature---and in the landscape of software---between the theory and
the practice of democracy.

On the software side, researchers have built a multitude of different
packages for generating and analyzing elections, and users have had to
invest substantial work in cleaning CVRs to make them usable across
multiple packages.[^4] `VoteKit` is built to provide an end-to-end
pipeline.

## Area of need: Generative models

For one concrete example of a literature and software gap, consider the
construction of *generative models*. This term is often associated with
large language models as paradigms of artificial intelligence; here,
what is being generated is realistic voting rather than realistic
language. In this setting, a generative model of voting is a probability
distribution on the set of all possible ballots that can be cast in a
given election style; profiles can be sampled from a generative model to
produce simulated or synthetic elections. Having sources of rich,
varied, and realistic data is essential to an empirically grounded
research program to probe the properties of voting rules. Good
generative models are also essential to advise reformers deciding
between options in a new locality, as they enable generation of
synthetic profiles keyed to the scale, demographics, and election specs
of that specific place. But most of the models in the literature, like
the Impartial Culture model (all permutations of candidates are equally
likely) or the Impartial Anonymous Culture model (sampling proportional
to volume measure on the simplex of weighted averages of permutations)
are mathematically tractable but highly unrealistic. This is bluntly
described by Tideman and Plassman in a survey of generative methods: in
their words, "None of the 11 models discussed so far are based on the
belief that the associated distributions \[\...\] might actually
describe rankings in actual elections\" [@Tideman2010TheSO]. They
therefore recommend *spatial models* instead, which themselves are of
dubious realism for the selection of political candidates.[^5]

`VoteKit` implements many of the models described in those surveys, as
well as newer mathematical models that give users the ability to
generate profiles that are designed to comport with real-world ranking
behavior and particularly to generate polarized elections. Two leading
choices are based on classic statistical ranking mechanisms, called the
Plackett--Luce (PL) and Bradley--Terry (BT) models; another model called
the Cambridge Sampler (CS) draws from historical ranking data in
Cambridge, MA city council elections. These models have flexible
parameters, allowing users to vary voting bloc proportions, candidate
strength within slates, and polarization between blocs. These parameters
can be specified or randomly sampled.

## Area of need: Comparison and communication

In the realm of democracy reform, groups of stakeholders often ask
researchers to provide modeling studies to decide on what shift to make
in electoral systems, as the project list below makes clear. `VoteKit`
implements voting rules that stakeholders often seek to compare, with
parameters designed to be tailored by the user to the specific locality
under study. Available voting rules include:

-   **Ranking-based (ordinal).** Plurality/SNTV, STV and IRV,
    (generalized) Borda, Alaska[^6], Top-Two, Dominating sets/Smith
    method, Condo-Borda[^7], Sequential RCV.

-   **Score-based (cardinal).** Range voting, Cumulative, Limited.

-   **Approval-based (set).** Approval voting, Bloc plurality.

See generally
[@electoralhandbook; @STV; @Borda; @TopTwo; @SequentialRCV] for
references.

Reform advocates also need to describe voting mechanisms and their
likely outcomes effectively to members of their communities. The end-to-end pipeline provided by `VoteKit` allows advocates to toggle different system settings and compare expected outcomes.

![A comparison of a variety of electoral systems and their affect on minority representation in a case study of the Washington state legislature [@washington_leg]. \label{fig:WA_comparison}](./figures/WA_poc_seats_chartsystem_compare_pared.png){width=100%}

<!-- `VoteKit`
includes a variety of metrics and visualizations intended to facilitate
this.

![Four visualizations for the same synthetic four-candidate preference profile with 1000 voters. The profile was generated with the CS model, which produces incomplete ballots at a realistic rate. \label{fig:comparison}](./figures/comparison.png){width=100%}

For a given preference profile, basic `VoteKit` functions provide
statistics and visuals for first-place votes, Borda count, and mention
frequency, as well as head-to-head comparisons (see \autoref{fig:comparison}). The pairwise comparison graph
shows head-to-head margins between candidates. For instance, 282 more
voters ranked $W_2>C_1$ than $C_1>W_2$ in the preference profile used to
make the figure. Note that $W_1$, who has the most first-place votes by
far, is also preferred head-to-head over all alternatives, making them
the *Condorcet candidate* in this election. -->

## Area of need: Resources for research

Previous research works such as [@elkind2017multiwinner] have compared
properties of earlier generative models; `VoteKit` facilitates robust
comparisons across a more comprehensive and up-to-date list of
alternatives. It also offers new analytical tools that will support
research on elections. Some examples of more sophisticated functionality
are shown in Figure 2. At left is a *ballot graph*,
where nodes are ballots weighted by their frequency in the profile; a
recent research paper shows that ballot graphs can be metrized to
realize classical statistical ranking distances, like Kendall tau and
the Spearman footrule [@duchin_tapp_24]. VoteKit also implements a class
of election distances, as surveyed in [@distance-elex]. Choices for
measuring the difference between two profiles on the same set of
candidates include $L^p$ distance and Wasserstein (earth-mover)
distance. At right is a multidimensional scaling (MDS) plot of a
different set of data, showing mutual $L^1$ differences between
generated profiles across various selections of model (shown in colors)
and candidate strength parameters (shown with symbols), enabling
comparisons in the style of [@drawing-a-map].

![At left, the ballot graph for a 3-candidate election. There
is one node per possible ballot, and the weights show the number of
instances of that ballot in the profile. At right, a multidimensional
scaling (MDS) plot for 160 synthetic profiles made with various
generative models and candidate strength parameters for two slates of 3
candidates each. The MDS plot is a low-distortion planar embedding of
those 160 profiles and their pairwise differences. \label{fig:viz_plots}](./figures/viz_plots.png){width=100%}

Finally, `VoteKit` interacts seamlessly with a wide range of actual vote
data, such as thousands of political elections collected by FairVote and
a cleaned repository of over 1000 Scottish STV local government
elections [@RCV-Cruncher; @Scot-Elex]. Previously, the use of real data
in election research was often extremely limited; for instance, a recent
survey reports that the single most popular "real-life\" dataset has
been a survey of 5000 respondents' sushi preferences
[@GuideExperiments].

# Projects

A significant number of white papers and scholarly articles have used
`VoteKit`(and its predecessor codebase) in recent years. These include
the following.

-   A large number of case studies in ranked-choice modeling, such as
    studies for the city councils of Chicago, IL [@chicago_city] and
    Lowell, MA [@lowell_city]; the state legislatures of Oregon and
    Washington [@oregon_state; @washington_leg], and a range of county
    commissions and school boards across the Pacific Northwest
    [@tukwila_school; @chelan_county];

-   A study modeling the impact of proposed legislation called the Fair
    Representation Act, which would convert U.S. Congressional elections
    to the single transferable vote system [@FairVote];

-   A detailed study isolating the impacts of varying hypotheses about
    voter behavior and candidate availability on the Massachusetts
    legislature [@massachusetts_leg];

-   A peer-reviewed article for an election law audience on the impact
    of STV elections on minority representation [@Benade2021];

-   A peer-reviewed article for a CS/econ audience that probes whether
    STV delivers proportional representation
    [@benade_donnay_duchin_weighill_24]; and

-   A peer-reviewed article for an CS/operations research audience on
    optimizing to "learn\" blocs and slates in real-world elections
    [@duchin_tapp_24].

# Acknowledgements

This work was initiated in a research cluster in Summer 2023, funded by
the Democracy Fund and graciously hosted at the Faculty of Computing and
Data Sciences at Boston University and the Tisch College of Civic Life
at Tufts University. Major contributors to the initiation of the project
include Brenda Macias, Emarie De La Nuez, Greg Kehne, Jordan Phan, Rory
Erlich, James Turk, and David McCune. Earlier code contributions were
made by Chanel Richardson, Anthony Pizzimenti, Gabe Schoenbach, Dylan
Phelan, Thomas Weighill, Dara Gold, and Amy Becker. The authors also
thank Deb Otis, Peter Rock, Jeanne Clelland, and Michael Parsons for
helpful feedback. FairVote's data repository in Dataverse
(<https://dataverse.harvard.edu/dataverse/rcv_cvrs>) and RCV Cruncher
code on GitHub (<https://github.com/fairvotereform/rcv_cruncher/>) are
excellent open-source efforts that were inspirational for the current
project.

# References

[^1]: Recent ranked-choice voting reforms include the adoption of
    instant runoff voting (IRV) in Maine, Alaska, New York City, and
    single transferable vote (STV) in Portland, Oregon. Advocacy groups
    claiming various pro-democratic properties of ranked choice include
    [Campaign Legal Center](https://perma.cc/77MM-DCPH),
    [FairVote](https://perma.cc/L66Z-AB4R), and many others.

[^2]: Nobel Laureates with significant work in social choice include
    Arrow, Sen, Maskin, and Myerson.

[^3]: For example, a very active research direction in computational
    social choice theory has been the development of fairness axioms for
    approval elections, such as the definition called JR (justified
    representation) and its relatives, which have been extended to
    rankings. See [@aziz2017justified; @skowron2017proportional] and
    their references.

[^4]: See for instance the extensive array of open-source tools on the
    Computational Social Choice (COMSOC) community page [@ComSoc]
    including the widely used collection of ranked data called PrefLib
    [@ComSoc] See also the materials provided by FairVote, including
    their DataVerse and GitHub [@RCV-Cruncher]. The ArXiV preprint
    [@GuideExperiments] provides an impressively comprehensive list of
    numerical experiments on elections. The PRAGMA Project
    (<https://perma.cc/2P6V-8ZER>) echoes our statement of need, noting
    that the current literature and software falls short in practical
    applicability and that the understanding of real and synthetic data
    is "very limited.\"

[^5]: Spatial models assume voters rank by proximity in a metric space
    defined by issue positions or other attributes; the metric space may
    be latent, or unknown to voters, but it is presumed to universally
    govern the way voters rank candidates. See for instance [@Burden],
    which introduces probabilistic voting keyed to proximity. Though
    spatial models have been argued to perform adequately to model roll
    call voting in Congress, their efficacy for selecting political
    representation is debatable. In a meta-analysis of 163 papers
    [@GuideExperiments], the authors report that Impartial Culture and
    Euclidean (spatial) models make up more than $75\%$ of the election
    experiments found in 163 papers.

[^6]: Our model of the Alaska method is an SNTV/STV hybrid that uses
    single non-transferable vote to choose a set of finalists, then runs
    STV on the same preference profile to fill the seats. Alaska's
    elections run this with four finalists and one seat; the top-two
    system runs this with two finalists and one seat.

[^7]: This system orders candidates within dominating sets by Borda
    score. Note that this is distinct from Black's method [@Black],
    which uses Borda score as a backup system in case the smallest
    dominating set is not a singleton.
