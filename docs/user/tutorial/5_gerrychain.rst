Gerrychain
==========

MGGG also maintains a package called ``gerrychain``, which is used to
draw legislative districts. One common pipeline that we use in the lab
is to generate a large number of possible maps using ``gerrychain``, and
then for each district in each map, use VoteKit to understand possible
election outcomes. This section of the tutorial provides a short example
of how this might be done.

To learn how to use gerrychain, please visit
https://mggg.github.io/GerryChain/index.html

You will need to run ``pip install 'gerrychain[geo]'`` in your terminal
with your virtual environment activated.

TODO download a json for pennsylvania
=====================================

https://github.com/mggg/GerryChain/blob/main/docs/\_static/PA_VTDs.json

We will use the 2016 Presidential race in Pennsylvania to estimate the
number of Republicans and Democrats in each district in the map. To
start, we run gerrychain to generate a large number of plans with four
districts. Again, don’t worry about the implementation details of
gerrychain here.

.. code:: ipython3

    import matplotlib.pyplot as plt
    from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                            proposals, updaters, constraints, accept, Election)
    from gerrychain.proposals import recom
    from functools import partial
    import pandas
    
    NUM_DISTRICTS = 4
    graph = Graph.from_json("./PA_VTDs.json")
    
    # Population updater, for computing how close to equality the district
    # populations are. "TOTPOP" is the population column from our shapefile.
    my_updaters = {"population": updaters.Tally("TOT_POP", alias="population")}
    
    # Election updaters, for computing election results using the vote totals
    # from our shapefile.
    
    elections = [
        Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
    ]
    
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)
    
    # we use a random plan with 4 districts
    initial_partition = Partition.from_random_assignment(
        graph,
        n_parts = NUM_DISTRICTS,
        epsilon = .02,
        pop_col="TOT_POP",
        updaters=my_updaters
    )
    
    # The ReCom proposal needs to know the ideal population for the districts so that
    # we can improve speed by bailing early on unbalanced partitions.
    
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    
    # We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
    # of the recom proposal.
    proposal = partial(
        recom,
        pop_col="TOT_POP",
        pop_target=ideal_population,
        epsilon=0.02,
        node_repeats=2
    )
    
    
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)
    
    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=100
    )
    
    # This might take a minute.
    # store the democrat and republican vote totals
    data = pandas.DataFrame(
        [partition["PRES16"].counts("Democratic") + partition["PRES16"].counts("Republican")
        for partition in chain], 
        columns = [f"D_{i}_Dem" for i in range(len(initial_partition))] + \
            [f"D_{i}_Rep" for i in range(len(initial_partition))]
            )

Now for each district in each map (4 x 100), we ask VoteKit to simulate
an STV election where each district elects 4 candidates. We compute the
number of Republican candidates who won, and display this for three
different models of ballot generator using an area plot.


.. code:: ipython3

    import votekit.ballot_generator as bg
    
    
    
    for map_num, row in data.iterrows():
    
        for district in range(NUM_DISTRICTS):

