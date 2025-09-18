from pandas.errors import EmptyDataError, DataError
from pathlib import Path
import pytest

from votekit.ballot import Ballot
from votekit.cvr_loaders import load_scottish
from votekit.pref_profile import RankProfile


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data/csv/"


def test_scot_csv_parse():
    pp, seats, cand_list, cand_to_party, ward = load_scottish(
        CSV_DIR / "scot_wardy_mc_ward.csv"
    )

    assert seats == 1
    assert isinstance(pp, RankProfile)
    assert set(["Paul", "George", "Ringo"]) == set(pp.candidates)
    assert len(pp.candidates) == 3
    assert cand_list == ["Paul", "George", "Ringo"]
    assert cand_to_party == {
        "Paul": "Orange (O)",
        "George": "Yellow (Y)",
        "Ringo": "Red (R)",
    }
    assert ward == "Wardy McWard Ward"
    assert int(pp.total_ballot_wt) == 146
    assert Ballot(ranking=tuple([frozenset({"Paul"})]), weight=126) in pp.ballots
    assert (
        Ballot(
            ranking=tuple(
                [frozenset({"Ringo"}), frozenset({"George"}), frozenset({"Paul"})]
            ),
            weight=1,
        )
        in pp.ballots
    )


def test_scot_csv_blank_rows():
    pp, seats, cand_list, cand_to_party, ward = load_scottish(
        CSV_DIR / "scot_blank_rows.csv"
    )

    assert seats == 1
    assert isinstance(pp, RankProfile)
    assert set(["Paul", "George", "Ringo"]) == set(pp.candidates)
    assert len(pp.candidates) == 3
    assert cand_list == ["Paul", "George", "Ringo"]
    assert cand_to_party == {
        "Paul": "Orange (O)",
        "George": "Yellow (Y)",
        "Ringo": "Red (R)",
    }
    assert ward == "Wardy McWard Ward"
    assert int(pp.total_ballot_wt) == 146
    assert Ballot(ranking=tuple([frozenset({"Paul"})]), weight=126) in pp.ballots
    assert (
        Ballot(
            ranking=tuple(
                [frozenset({"Ringo"}), frozenset({"George"}), frozenset({"Paul"})]
            ),
            weight=1,
        )
        in pp.ballots
    )


def test_bad_file_path_scot_csv():
    with pytest.raises(ValueError, match="unknown url type:"):
        load_scottish("")


def test_empty_file_scot_csv():
    with pytest.raises(EmptyDataError):
        load_scottish(CSV_DIR / "scot_empty.csv")


def test_bad_metadata_scot_csv():
    with pytest.raises(DataError):
        load_scottish(CSV_DIR / "scot_bad_metadata.csv")


def test_incorrect_metadata_scot_csv():
    with pytest.raises(DataError):
        load_scottish(CSV_DIR / "scot_candidate_overcount.csv")

    with pytest.raises(DataError):
        load_scottish(CSV_DIR / "scot_candidate_undercount.csv")


def test_scot_csv_url():
    pp, seats, cand_list, cand_to_party, ward = load_scottish(
        "https://github.com/mggg/scot-elex/raw/refs/heads/main/10_cands/aberdeen_2017_ward12.csv"
    )
    true_cand_to_party = {
        "Yvonne Allan": "Labour (Lab)",
        "Christian Guy Allard": "Scottish National Party (SNP)",
        "Alan Donnelly": "Conservative and Unionist Party (Con)",
        "David Fryer": "Independent (Ind)",
        "Catriona Mackenzie": "Scottish National Party (SNP)",
        "Gregor Mcabery": "Liberal Democrat (LD)",
        "William Allan Mcintosh": "UK Independence Party (UKIP)",
        "Ren": "Green (Gr)",
        "Piotr Teodorowski": "Labour (Lab)",
        "Billy Watson": "National Front (NF)",
    }
    assert isinstance(pp, RankProfile)
    assert seats == 4
    assert ward == "Torry/Ferryhill Ward"
    assert cand_to_party == true_cand_to_party
    assert set(cand_list) == set(true_cand_to_party.keys())
