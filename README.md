## VoteKit

`VoteKit` ia a Swiss army knife for computational social choice research.

**Helpful links:** [Source Repository](https://github.com/mggg/VoteKit) | [Documentation](https://mggg.github.io/VoteKit/) | [Issues](https://github.com/mggg/VoteKit/issues) | [MGGG.org](https://mggg.org/)


[![PyPI badge](https://badge.fury.io/py/votekit.svg)](https://badge.fury.io/py/votekit)
![Test badge](https://github.com/mggg/VoteKit/workflows/Test%20&%20Lint/badge.svg)

## Installation
Votekit can be installed through any standard package management tool:

    pip install votekit

or

    poetry add votekit

## Example

A simple example of how to use VoteKit to load, clean, and run an election using real [data](https://vote.minneapolismn.gov/results-data/election-results/2013/mayor/) taken from the 2013 Minneapolis Mayoral election. For a more comprehensive walkthrough, see the [documentation](https://mggg.github.io/VoteKit/). 

```python
from votekit import load_csv, remove_noncands
from votekit.elections import STV, fractional_transfer

minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")

# clean downloaded file to remove edited aspects of the cast vote record
minneapolis_profile = remove_noncands(minneapolis_profile, ["undervote", "overvote", "UWI"])

minn_election = STV(profile = minneapolis_profile, transfer = fractional_transfer, seats = 1)
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

## Development
*This project is in active development* in the [mggg/VoteKit](https://github.com/mggg/VoteKit) GitHub repository, where bug reports and feature requests, as well as contributions, are welcome.

VoteKit project requires [`poetry`](https://python-poetry.org/docs/#installation), and Python >= 3.9. (This version chosen somewhat arbitrarily.)

To get up and running, run `poetry install` from within the project directory to install all dependencies. This will create a `.venv` directory that will contain dependencies. You can interact with this virtualenv by running your commands prefixed with `poetry run`, or use `poetry shell` to activate the virtualenv.

Once you've run `poetry install`, if you run `poetry run pre-commit install` it will install code linting hooks that will run on every commit. This helps ensure code quality.

To run tests run `poetry run pytest` or `./run_tests.sh` (the latter will generate a coverage report).

To release, run `poetry publish --build`.
