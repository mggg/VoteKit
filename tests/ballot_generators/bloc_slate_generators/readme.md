# Overall testing outline 

Each ballot generator should be tested as follows:
- a "completion" test: does the generator create a profile with the correct number of ballots? Does the generator create a profile dictionary with the correct blocs?
- invalid configs: an invalid bloc slate config object should raise an error
- any errors specific to the ballot generator should be raised
- zero support slates are randomly permuted at the end of the ballot 
- a 2 bloc and 2 slate config matches the correct distribution for name and slate ballot types. should have at least 2 candidates per slate.
- a 1 bloc and 3 slate config matches the correct distribution for name and slate ballot types. should be a 3/2/2 candidate split.