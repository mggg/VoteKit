from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _lexicographic_symbol_tuple_iterator,
)


def test_lexicographic_symbol_tuple_iterator_empty():
    result = list(_lexicographic_symbol_tuple_iterator([]))
    assert result == []


def test_lexicographic_symbol_tuple_iterator_single():
    result = list(_lexicographic_symbol_tuple_iterator(["A"]))
    assert result == [("A",)]


def test_lexicographic_symbol_tuple_iterator_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "A", "B"]))
    expected = [("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_multiple():
    result = list(_lexicographic_symbol_tuple_iterator(["C", "A", "B"]))
    expected = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    ]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_with_more_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "B", "A", "C"]))
    expected = [
        ("A", "A", "B", "C"),
        ("A", "A", "C", "B"),
        ("A", "B", "A", "C"),
        ("A", "B", "C", "A"),
        ("A", "C", "A", "B"),
        ("A", "C", "B", "A"),
        ("B", "A", "A", "C"),
        ("B", "A", "C", "A"),
        ("B", "C", "A", "A"),
        ("C", "A", "A", "B"),
        ("C", "A", "B", "A"),
        ("C", "B", "A", "A"),
    ]
    assert result == expected
