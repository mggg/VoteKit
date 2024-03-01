# Ballot Generators

In addition to being able to [read real world voting data](api.md#cvr-loaders), VoteKit also has the ability to generate ballots using different models. This is useful when you want to run experiments or just play around with some data. We make no claims that these models accurately predict real voting behavior.

## Ballot Simplex Models

Models listed below generate ballots by using the [ballot simplex](SCR_simplex.md). This means we take a draw from the Dirichlet distribution, which gives us a probability distribution on full, linear rankings. We then generate ballots according to this distribution.

### Impartial Culture

The Impartial Culture model has $\alpha = \infty$. As discussed in [ballot simplex](SCR_simplex.md), this is not actually a valid parameter for the Dirichlet distribution, so instead VoteKit sets $\alpha = 10^{20}$. This means that the point drawn from the ballot simplex has a very high probability of being in the center, which means it gives uniform probability to each linear ranking.

### Impartial Anonymous Culture

The Impartial Anonymous Culture model has $\alpha = 1$. This means that the point is uniformly drawn from the ballot simplex. This does not mean we have a uniform distribution on rankings; rather, we have a uniform chance of choosing any distribution on rankings.

## Candidate Simplex Models

### Name-Plackett-Luce
The name-Plackett-Luce model (n-PL) samples ranked ballots as follows. Assume there are $n$ blocs of voters. Within a bloc, say bloc $A$, voters have $n$ preference intervals, one for each slate of candidates. A bloc also has a fixed $n$-tuple of cohesion parameters $\pi_A = (\pi_{AA}, \pi_{AB},\dots)$; we require that $\sum_B \pi_{AB}=1$. To generate a ballot for a voter in bloc $A$, each preference interval $I_B$ is rescaled by the corresponding cohesion parameter $\pi_{AB}$, and then concatenated to create one preference interval. 
Voters then sample without replacement from the combined preference interval.

### Name-Bradley-Terry
The name-Bradley-Terry model (n-BT) samples ranked ballots as follows. Assume there are $n$ blocs of voters. Within a bloc, say bloc $A$, voters have $n$ preference intervals, one for each slate of candidates. A bloc also has a fixed $n$-tuple of cohesion parameters $\pi_A = (\pi_{AA}, \pi_{AB},\dots)$; we require that $\sum_B \pi_{AB}=1$. To generate a ballot for a voter in bloc $A$, each preference interval $I_B$ is rescaled by the corresponding cohesion parameter $\pi_{AB}$, and then concatenated to create one preference interval. 
Voters then sample ballots proportional to pairwise probabilities of candidates. That is, the probability that the ballot $C_1>C_2>C_3$ is sampled is proprotional to $P(C_1>C_2)P(C_2>C_3)P(C_1>C_3)$, where these pairwise probabilities are given by $P(C_1>C_2) = C_1/(C_1+C_2)$.
Here $C_i$ denotes the length of $C_i$'s share of the combined preference interval.

### Name-Cumulative
The name-Cumulative model (n-C) samples ranked ballots as follows. Assume there are $n$ blocs of voters. Within a bloc, say bloc $A$, voters have $n$ preference intervals, one for each slate of candidates. A bloc also has a fixed $n$-tuple of cohesion parameters $\pi_A = (\pi_{AA}, \pi_{AB},\dots)$; we require that $\sum_B \pi_{AB}=1$. To generate a ballot for a voter in bloc $A$, each preference interval $I_B$ is rescaled by the corresponding cohesion parameter $\pi_{AB}$, and then concatenated to create one preference interval. To generate a ballot, voters sample with replacement from the combined interval as many times as determined by the length of the desired ballot.

### Slate-Plackett-Luce
The slate-Plackett-Luce model (s-PL) samples ranked ballots as follows. Assume there are $n$ blocs of voters. Within a bloc, say bloc $A$, voters have $n$ preference intervals, one for each slate of candidates. A bloc also has a fixed $n$-tuple of cohesion parameters $\pi_A = (\pi_{AA}, \pi_{AB},\dots)$; we require that $\sum_B \pi_{AB}=1$. Now the cohesion parameters play a different role than in the name models above. For s-PL, $\pi_{AB}$ gives the probability that we put a $B$ candidate in each position on the ballot. If we have already exhausted the number of $B$ candidates, we remove $\pi_{AB}$ and renormalize. Once we have a ranking of the slates on the ballot, we fill in candidate ordering by sampling without replacement from each individual preference interval (we do not concatenate them!).

### Slate-Bradley-Terry
The slate-Bradley-Terry model (s-BT) samples ranked ballots as follows. We assume there are 2 blocs of voters. Within a bloc, say bloc $A$, voters have 2 preference intervals, one for each slate of candidates. A bloc also has a fixed tuple of cohesion parameters $\pi_A = (\pi_A, 1-\pi_A)$. Now the cohesion parameters play a different role than in the name models above. For s-BT, we again start by filling out a ballot with bloc labels only. Now, the probability that we sample the ballot $A>A>B$ is proportional to $\pi_A^2$; just like name-Bradley-Terry, we are computing pairwise comparisons. In $A>A>B$, slate $A$ must beat slate $B$ twice. As another example, the probability of $A>B>A$ is proportional to $\pi_A(1-\pi_A)$. Once we have a ranking of the slates on the ballot, we fill in candidate ordering by sampling without replacement from each individual preference interval (we do not concatenate them!).

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
