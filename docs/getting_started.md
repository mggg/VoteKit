# Getting started with `votekit`

This guide will help you get started using `votekit`, by using real election data from the 2013 Minneapolis mayoral election. This election had 35 candidates running for one seat, and used a single-winner IRV method to elect the winner. Voters were allowed to rank their top three candidates. 


```python
# these are the votekit functions we'll need access to
from votekit.cvr_loaders import load_csv
from votekit.elections import STV, fractional_transfer
from votekit.cleaning import remove_noncands
```

You can find the necessary csv file `mn_2013_cast_vote_record.csv` in the `votekit/data` folder of the GitHub repo. Alternatively, you can download the offical cast vote record (CVR) [here](https://vote.minneapolismn.gov/results-data/election-results/2013/mayor/). Download a verison of the file, and then edit the path below to where you placed it. The csv file has 3 columns we care about. The first, entitled '1ST CHOICE MAYOR MINNEAPOLIS' in the official CVR, tells us a voters top choice, then the second tells us their second choice, and the third their third choice.

The first thing we will do is create a `PreferenceProfile` object from our csv. A preference profile is a term from the social choice literature that represents the rankings of some set of candidates from some voters. Put another way, a preference profile stores the votes from an election, and is a collection of `Ballot` objects and candidates. 

We give the `load_csv` function the path to the csv file. By default, each column of the csv should correspond to a ranking of a candidate, given in decreasing order (the first column is the voters top choice, the last column their bottom choice.) There are some other optional parameters which you can read about in the documentation, like how to read a csv file that has more columns than just rankings.


```python
# you'll need to edit this path!
minneapolis_profile = load_csv("../src/votekit/data/mn_2013_cast_vote_record.csv")
```

The `PreferenceProfile` object has lots of helpful methods that allow us to study our votes. Let's use some of them to explore the ballots that were submitted. This is crucial since our data was not preprocessed. There could be undervotes, overvotes, defective, or spoiled ballots.

The `get_candidates` method returns a unique list of candidates. The `head` method shows the top *n* ballots. In the first column, we see the ballot that was cast. In the second column, we see how many of that type of ballot were cast. 


```python
# returns a list of unique candidates
print(minneapolis_profile.get_candidates())

# returns the top n ballots
minneapolis_profile.head(n=5)
```

    ['JOHN LESLIE HARTWIG', 'ALICIA K. BENNETT', 'ABDUL M RAHAMAN "THE ROCK"', 'CAPTAIN JACK SPARROW', 'STEPHANIE WOODRUFF', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'DOUG MANN', 'CHRISTOPHER CLARK', 'TROY BENJEGERDES', 'JACKIE CHERRYHOMES', 'DON SAMUELS', 'KURTIS W. HANNA', 'overvote', 'MARK ANDREW', 'OLE SAVIOR', 'TONY LANE', 'JAYMIE KELLY', 'MIKE GOULD', 'CHRISTOPHER ROBIN ZIMMERMAN', 'GREGG A. IVERSON', 'DAN COHEN', 'CYD GORMAN', 'UWI', 'BILL KAHN', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'CAM WINTON', 'EDMUND BERNARD BRUYERE', 'BETSY HODGES', 'undervote', 'BOB FINE', 'JOHN CHARLES WILSON', 'JEFFREY ALAN WAGNER', 'JOSHUA REA', 'MARK V ANDERSON', 'NEAL BAXTER', 'BOB "AGAIN" CARNEY JR']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>Ballots</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(MARK ANDREW, undervote, undervote)</td>
      <td>3864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(BETSY HODGES, MARK ANDREW, DON SAMUELS)</td>
      <td>3309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(BETSY HODGES, DON SAMUELS, MARK ANDREW)</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(MARK ANDREW, BETSY HODGES, DON SAMUELS)</td>
      <td>2502</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(BETSY HODGES, undervote, undervote)</td>
      <td>2212</td>
    </tr>
  </tbody>
</table>
</div>



Woah, that's a little funky! There's a candidate called 'undervote','overvote', and 'UWI'. In this dataset, 'undervote' says that someone left a ranking blank. The 'overvote' candidate arises when someone lists two candidates in one ranking, and in our data set, we lose any knowledge of their actual preference. 'UWI' stands for unregistered write-in.

It's really important to think carefully about how you want to handle cleaning up the ballots, as this depends entirely on the context of a given election. For now, let's assume that we want to get rid of the 'undervote', 'overvote', and 'UWI' candidates. The function `remove_noncands` will do this for us. If a ballot was "A B undervote", it would now be "A B". If a ballot was "A UWI B" it would now be "A B" as well. This might not be how you want to handle such things, but for now let's go with it. 


```python
minneapolis_profile = remove_noncands(minneapolis_profile, ["undervote", "overvote", "UWI"])
print(minneapolis_profile.get_candidates())
```

    ['NEAL BAXTER', 'JAYMIE KELLY', 'MIKE GOULD', 'CHRISTOPHER ROBIN ZIMMERMAN', 'GREGG A. IVERSON', 'DAN COHEN', 'JOHN LESLIE HARTWIG', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BILL KAHN', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'CAPTAIN JACK SPARROW', 'CAM WINTON', 'STEPHANIE WOODRUFF', 'EDMUND BERNARD BRUYERE', 'JAMES EVERETT', 'BETSY HODGES', 'JAMES "JIMMY" L. STROUD, JR.', 'DOUG MANN', 'CHRISTOPHER CLARK', 'TROY BENJEGERDES', 'JACKIE CHERRYHOMES', 'BOB FINE', 'JOHN CHARLES WILSON', 'DON SAMUELS', 'JEFFREY ALAN WAGNER', 'KURTIS W. HANNA', 'JOSHUA REA', 'MARK ANDREW', 'OLE SAVIOR', 'MARK V ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'TONY LANE', 'BOB "AGAIN" CARNEY JR']


Alright, things are looking a bit cleaner. Let's examine some of the ballots.


```python
# returns the top n ballots
minneapolis_profile.head(n=5, percents = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>Ballots</th>
      <th>Weight</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(MARK ANDREW,)</td>
      <td>3864</td>
      <td>4.87%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(BETSY HODGES, MARK ANDREW, DON SAMUELS)</td>
      <td>3309</td>
      <td>4.17%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(BETSY HODGES, DON SAMUELS, MARK ANDREW)</td>
      <td>3031</td>
      <td>3.82%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(MARK ANDREW, BETSY HODGES, DON SAMUELS)</td>
      <td>2502</td>
      <td>3.15%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(BETSY HODGES,)</td>
      <td>2212</td>
      <td>2.79%</td>
    </tr>
  </tbody>
</table>
</div>



We can similarly print the bottom $n$ ballots. Here we toggle the optional `percents` and `totals` arguments, which will show us the fraction of the total vote, as well as sum up the weights.


```python
# returns the bottom n ballots
minneapolis_profile.tail(n=5, percents = False, totals = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>Ballots</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6916</th>
      <td>(STEPHANIE WOODRUFF,)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6915</th>
      <td>(DON SAMUELS, ABDUL M RAHAMAN "THE ROCK", MARK...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6914</th>
      <td>(DON SAMUELS, ABDUL M RAHAMAN "THE ROCK", MIKE...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6913</th>
      <td>(DON SAMUELS, ABDUL M RAHAMAN "THE ROCK", OLE ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6912</th>
      <td>(DON SAMUELS, ABDUL M RAHAMAN "THE ROCK", RAHN...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Totals</th>
      <td></td>
      <td>5 out of 79378</td>
    </tr>
  </tbody>
</table>
</div>



There are a few other methods you can read about in the documentation, but now let's run an election!

Just because we have a collection of ballots does not mean we have a winner. To convert a PreferenceProfile into a winner (or winners), we need to choose a method of election. The mayoral race was conducted as a single winner IRV election, which in `votekit` is equivalent to a STV election with one seat. The transfer method tells us what to do if someone has a surplus of votes over the winning quota (which by default is the Droop quota). 


```python
minn_election = STV(profile = minneapolis_profile, transfer = fractional_transfer, seats = 1)
```


```python
# the run_election method prints a dataframe showing the order in which candidates are eliminated under STV
minn_election.run_election()
```

    Current Round: 35





                       Candidate     Status  Round
                    BETSY HODGES    Elected     35
                     MARK ANDREW Eliminated     34
                     DON SAMUELS Eliminated     33
                      CAM WINTON Eliminated     32
              JACKIE CHERRYHOMES Eliminated     31
                        BOB FINE Eliminated     30
                       DAN COHEN Eliminated     29
              STEPHANIE WOODRUFF Eliminated     28
                 MARK V ANDERSON Eliminated     27
                       DOUG MANN Eliminated     26
                      OLE SAVIOR Eliminated     25
                   JAMES EVERETT Eliminated     24
               ALICIA K. BENNETT Eliminated     23
      ABDUL M RAHAMAN "THE ROCK" Eliminated     22
            CAPTAIN JACK SPARROW Eliminated     21
               CHRISTOPHER CLARK Eliminated     20
                       TONY LANE Eliminated     19
                    JAYMIE KELLY Eliminated     18
                      MIKE GOULD Eliminated     17
                 KURTIS W. HANNA Eliminated     16
     CHRISTOPHER ROBIN ZIMMERMAN Eliminated     15
             JEFFREY ALAN WAGNER Eliminated     14
                     NEAL BAXTER Eliminated     13
                TROY BENJEGERDES Eliminated     12
                GREGG A. IVERSON Eliminated     11
                MERRILL ANDERSON Eliminated     10
                      JOSHUA REA Eliminated      9
                       BILL KAHN Eliminated      8
             JOHN LESLIE HARTWIG Eliminated      7
          EDMUND BERNARD BRUYERE Eliminated      6
    JAMES "JIMMY" L. STROUD, JR. Eliminated      5
                RAHN V. WORKCUFF Eliminated      4
           BOB "AGAIN" CARNEY JR Eliminated      3
                      CYD GORMAN Eliminated      2
             JOHN CHARLES WILSON Eliminated      1



And there you go! You've created a PreferenceProfile from real election data, done some cleaning, and then conducted an STV election. You can look at the [offical results](https://vote.minneapolismn.gov/results-data/election-results/2013/mayor/) and confirm that `votekit` elected the same candidate as in the real 2013 election.
