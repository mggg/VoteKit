# Elections

## STV

An STV election stands for single transferable vote. Voters cast ranked choice ballots. A threshold is set; if a candidate crosses the threshold, they are elected. The threshold defaults to the Droop quota. We also enable functionality for the Hare quota.

In the first round, the first place votes for each candidate are tallied. If a candidate crosses the threshold, they are elected. Any surplus votes are distributed amongst the other candidates according to a transfer rule. If another candidate crosses the threshold, they are elected. If no candidate does, the candidate with the least first place votes is eliminated, and their ballots are redistributed according to the transfer rule. This repeats until all seats are filled.

- An STV election can use either the Droop or Hare quota.

- The current transfer methods are stored in the [elections](api.md#elections) module.

- If there is a tiebreak needed, STV defaults to a random tiebreak. Other methods of tiebreak are given in the tie_broken_ranking function of the utils module.

## Limited

## Bloc

## SNTV

## SNTV_STV_Hybrid

## TopTwo

## DominatingSets

## Condo Borda

## SequentialRCV

## Borda

## Plurality

## Quotas and Transfers

### Droop

### Hare

### Fractional Trasnfer


