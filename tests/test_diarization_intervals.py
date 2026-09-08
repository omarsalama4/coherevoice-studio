from coherex.diarize import IntervalTree


def test_interval_query_and_nearest():
    tree = IntervalTree([(0, 2, "A"), (5, 7, "B"), (9, 10, "C")])
    assert tree.query(1, 6) == [("A", 1), ("B", 1)]
    assert tree.find_nearest(8.8) == "C"
