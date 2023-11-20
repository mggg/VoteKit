# Distances between PreferenceProfiles

# Earthmover Distance

The Earthmover distance is a measure of how far apart two distributions are over a given metric space. In our case, the metric space is the `BallotGraph` endowed with the shortes path metric. We then consider a `PreferenceProfile` to be a distribution that assigns the number of times a ballot was cast to a node of the `BallotGraph`. Informally, the Earthmover distance is the minimum cost of moving the "dirt" piled on the nodes by the first profile to the second profile given the distance it must travel.

# $L_p$ Distance

The $L_p$ distance is a metric parameterized by $p\in (0,\infty]$. It is computed as $d(P_1,P_2) = \left(\sum |P_1(b)-P_2(b)|^p\right)^{1/p}$, where the sum is indexed over all possible ballots, and $P_i(b)$ denotes the number of times that ballot was cast.


```python

```
