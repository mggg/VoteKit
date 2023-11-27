# Ballot Generators

In addition to being able to [read real world voting data](api.md#cvr-loaders), VoteKit also has the ability to generate ballots using different models. This is useful when you want to run experiments or just play around with some data. We make no claims that these models accurately predict real voting behavior.

## Ballot Simplex Models

Models listed below generate ballots by using the [ballot simplex](SCR_simplex.md). This means we take a draw from the Dirichlet distribution, which gives us a probability distribution on full, linear rankings. We then generate ballots according to this distribution.

### Impartial Culture

The Impartial Culture model has $\alpha = \infty$. As discussed in [ballot simplex](SCR_simplex.md), this is not actually a valid parameter for the Dirichlet distribution, so instead VoteKit sets $\alpha = 10^{20}$. This means that the point drawn from the ballot simplex has a very high probability of being in the center, which means it gives uniform probability to each linear ranking.

### Impartial Anonymous Culture

The Impartial Anonymous Culture model has $\alpha = 1$. This means that the point is uniformly drawn from the ballot simplex. This does not mean we have a uniform distribution on rankings; rather, we have a uniform chance of choosing any distribution on rankings.

## Candidate Simplex Models

### Plackett-Luce

The Plackett-Luce model (PL) samples ranked ballots as follows. Given a bloc's preference interval, it samples candidates without replacement from the interval. That means when a candidate is selected, their portion of the interval is removed, and the interval is normalized to be length 1 again. 

- The PL model generates full ballots, with the caveat that any candidates with 0 support are listed as ties at the end of the ballot.

- It can be initialized directly from a set of preference intervals (one for each bloc), or by using [from_params](api.md#ballot-generators). This method uses cohesion and Dirichlet parameters.

- The PL model can handle arbitrarily many blocs.

- The PL model also requires information about what proportion of voters belong to each bloc.

### Bradley-Terry

The Bradley-Terry model (BT) samples ranked ballots as follows. Given a preference interval, the probability of sampling the ballot $A>B>C$ is equal to the product of the probabilities $P(A>B)P(B>C)P(A>C)$. One of these probabilities can be computed as $P(A>B) = A/(A+B)$, where we let $A$ denote both the candidate and the length of its interval.


- The BT model generates full ballots, with the caveat that any candidates with 0 support are listed as ties at the end of the ballot.

- It can be initialized directly from a set of preference intervals (one for each bloc), or by using [from_params](api.md#ballot-generators). This method uses cohesion and Dirichlet parameters.

- The BT model can handle arbitrarily many blocs.

- The BT model also requires information about what proportion of voters belong to each bloc.

### Alternating-Crossover

The Alternating-Crossover model (AC) samples ranked ballots as follows. It assumes there are only two blocs. Within a bloc, voters either vote with the bloc, or they alternate. The proportion of such voters is determined by the cohesion parameter. If a voter votes with the bloc, they list all of their bloc's candidates above the other bloc's. If a voter alternates, they list an opposing candidate first, and then alternate between their bloc and the opposing until they run out of one set of candidates. In either case, the order of candidates is determined by a PL model.

- The AC model can generate incomplete ballots if there are a different number of candidates in each bloc.

- The AC model can be initialized from a set of preference intervals, along with which candidates belong to which bloc and a set of cohesion parameters.

- The AC model only works with two blocs.

- The AC model also requires information about what proportion of voters belong to each bloc.

### Cambridge-Sampler

The Cambridge-Sampler (CS) samples ranked ballots as follows. Assume there is a majority and a minority bloc. If a voter votes with their bloc, they rank a bloc candidate first. If they vote against their bloc, they rank an opposing bloc candidate first. The proportion of such voters is determined by the cohesion parameter. Once a first entry is recorded, the CS samples a ballot type from historical Cambridge, MA election data. That is, if a voter puts a majorrity candidate first, the rest of their ballot type is sampled in proportion to the number of historical ballots that started with a majority candidate. Once a ballot type is determined, the order of candidates is determined by a PL model.

Let's do an example. I am a voter in the majority bloc. I flip a coin weighted by the cohesion parameter, and it comes up tails. My ballot type will start with a minority candidate $m$. The CS samples historical ballots that also started with $m$, and tells me my ballot type is $mmM$; two minority candidates, then a majority. Finally, CS uses a PL model to determine which minority/majority candidates go in the slots.

- CS can generate incomplete ballots since it uses historical data.

- The CS model can be initialized from a set of preference intervals, along with which candidates belong to which bloc and a set of cohesion parameters.

- The CS model only works with two blocs if you use the Cambridge data.

- The CS model also requires information about what proportion of voters belong to each bloc.

- You can give the CS model other historical election data to use.

## Distance Models

#### 1-D Spatial

The 1-D Spatial model samples ranked ballots as follows. First, it assigns each candidate a position on the real number line according to a normal distribution. Then, it does the same with each voter. Finally, a voter's ranking is determined by their distance from each candidate.

- The 1-D Spatial model only generates full ballots.

- The 1-D Spatial model can be initialized from a list of candidates.
