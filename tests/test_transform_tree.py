"""Tests for the recursive transform-source tree helper."""

from fractions import Fraction

from cfmath import CF, Ln, PolyTransform, Sin, describe_source_tree
from cfmath.gosper_cf import GosperBi, GosperGeneric, GosperMono


class TestDescribeSourceTree:
    def test_recurses_through_single_source_transforms(self):
        base = CF.from_fraction(3, 7)
        inner = GosperMono(base, 1, 3, 0, 1)
        outer = PolyTransform(inner, [1, 1], [1])

        lines = describe_source_tree(outer).splitlines()

        assert lines[0] == "PolyTransform(num=[1, 1], den=[1])"
        assert lines[1] == "  source:"
        assert lines[2] == "    GosperMono(1, 3, 0, 1)"
        assert lines[3].startswith("      [0;")

    def test_labels_multi_source_nodes(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        bi = GosperBi(x, y, 0, 1, 1, 0, 0, 0, 0, 1)
        tree = describe_source_tree(GosperGeneric([bi, y], [0, 1, 1, 0], [1, 0, 0, 0]))

        assert "GosperGeneric(num=[0, 1, 1, 0], den=[1, 0, 0, 0])" in tree
        assert "source[0]:" in tree
        assert "source[1]:" in tree
        assert "GosperBi(0, 1, 1, 0, 0, 0, 0, 1)" in tree
        assert "x:" in tree
        assert "y:" in tree

    def test_includes_named_cf_constructors(self):
        expr = CF.from_int(2) + Ln(Sin(Fraction(1, 2)))

        tree = describe_source_tree(expr)

        assert "cf_add" in tree
        assert "Ln" in tree
        assert "Sin" in tree
