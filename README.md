## VoteKit

`VoteKit` is a Swiss army knife for computational social choice research.

**Helpful links:** [Source Repository](https://github.com/mggg/VoteKit) | [Documentation](https://votekit.readthedocs.io/en/latest/) | [Issues & Feature Requests](https://votekit.readthedocs.io/en/latest/package_info/issues/) | [Contributing](https://votekit.readthedocs.io/en/latest/package_info/contributing/) | [MGGG.org](https://mggg.org/)


[![PyPI badge](https://badge.fury.io/py/votekit.svg)](https://badge.fury.io/py/votekit)
![Test badge](https://github.com/mggg/VoteKit/workflows/Test%20&%20Lint/badge.svg)

## Installation
Votekit can be installed through any standard package management tool:

    pip install votekit

For more detailed instructions, please see the [installation](https://votekit.readthedocs.io/en/latest/#installation) section of the VoteKit documentation.

## Example

A simple example of how to use VoteKit to load, clean, and run an election using real [data](https://vote.minneapolismn.gov/results-data/election-results/2013/mayor/) taken from the 2013 Minneapolis Mayoral election. For a more comprehensive walkthrough, see the [documentation](https://votekit.readthedocs.io/en/latest/). 

```python
from votekit.cvr_loaders import load_ranking_csv
from votekit.cleaning import remove_repeated_candidates, remove_cand, condense_profile
from votekit.elections import STV

minneapolis_profile = load_ranking_csv("mn_2013_cast_vote_record.csv", [0,1,2], header_row = 0)

# clean downloaded file to remove edited aspects of the cast vote record
minneapolis_profile = remove_cand(["undervote", "overvote", "UWI"], minneapolis_profile)
minneapolis_profile = remove_repeated_candidates(minneapolis_profile)
minneapolis_profile = condense_profile(minneapolis_profile)

minn_election = STV(profile = minneapolis_profile, m = 1)
minn_election.run_election()
```

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

## Issues and Contributing
This project is in active development in the [mggg/VoteKit](https://github.com/mggg/VoteKit) GitHub repository, where [bug reports and feature requests](https://votekit.readthedocs.io/en/latest/package_info/issues/), as well as [contributions](https://votekit.readthedocs.io/en/latest/package_info/contributing/), are welcome.

Currently VoteKit uses `poetry` to manage the development environment. If you want to make a pull request, first `pip install poetry` to your computer. Then, within the Votekit directory and with a virtual environment activated, run `poetry install` This will install all of the development packages you might need. Before making a pull request, run the following:
- `poetry run pytest tests --runslow` to check the test suite,
- `poetry run black .` to format your code,
- `poetry run ruff check src tests` to check the formatting, and then
- `poetry run mypy src` to ensure that your typesetting is correct.

Then you can create your PR! Please do not make your PR against `main`.

