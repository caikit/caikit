# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Third Party
import utils

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestSpan(TestCaseBase):
    def setUp(self):
        self.span = dm.Span(0, 1)

    def test_fields(self):
        self.assertTrue(hasattr(self.span, "begin"))
        self.assertTrue(hasattr(self.span, "end"))

    def test_invalid_text(self):
        """Check that an assertion error is raised if text length does not match span width."""
        with self.assertRaises(ValueError):
            dm.Span(0, 2, text="hello world")

        with self.assertRaises(ValueError):
            dm.Span(0, 20, text="hello world")

        test_span = dm.Span(0, 20)
        with self.assertRaises(ValueError):
            test_span.slice_and_set_text("hello world")

    def test_empty_span(self):
        """Verify that an empty span is valid."""
        empty_span = dm.Span(0, 0)
        self.assertEqual(len(empty_span), 0)

    def test_order(self):
        a = dm.Span(0, 2)
        b = dm.Span(2, 4)
        self.assertTrue(a < b)
        self.assertTrue(b > a)

        a = dm.Span(0, 3)
        b = dm.Span(2, 4)
        self.assertTrue(a < b)
        self.assertTrue(b > a)

        a = dm.Span(0, 5)
        b = dm.Span(2, 4)
        self.assertTrue(a < b)
        self.assertTrue(b > a)

        a = dm.Span(0, 4)
        b = dm.Span(0, 5)
        self.assertTrue(a < b)
        self.assertTrue(b > a)

        a = dm.Span(0, 10)
        b = dm.Span(2, 4)
        self.assertTrue(b in a)
        self.assertFalse(a in b)

        a = dm.Span(0, 5)
        b = dm.Span(0, 5)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a <= b)
        self.assertTrue(a >= b)

        a = dm.Span(0, 5)
        b = dm.Span(1, 4)
        self.assertTrue(a != b)
        self.assertFalse(a == b)
        self.assertEqual(len(a), 5)

    def test_from_proto_and_back(self):
        new = dm.Span.from_proto(self.span.to_proto())
        self.assertEqual(new.begin, self.span.begin)
        self.assertEqual(new.end, self.span.end)

    def test_from_json_and_back(self):
        new = dm.Span.from_json(self.span.to_json())
        self.assertEqual(new.begin, self.span.begin)
        self.assertEqual(new.end, self.span.end)

    def test_from_json_negative(self):
        with self.assertRaises(ValueError):
            dm.Token.from_json(self.span.to_json())

    def test_overlaps(self):
        a = dm.Span(0, 2)
        b = dm.Span(2, 4)
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))

        a = dm.Span(10, 20)
        b = dm.Span(30, 40)
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))

        a = dm.Span(1, 4)
        b = dm.Span(3, 5)
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

        a = dm.Span(1, 30)
        b = dm.Span(4, 10)
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))


class TestNGram(TestCaseBase):
    def setUp(self):
        self.n_gram = dm.NGram(["drive", "travel", "car", "road"])
        self.n_gram_w_rel = dm.NGram(["bus", "station", "visit"], 0.8)

    def test_fields(self):
        self.assertTrue(hasattr(self.n_gram, "texts"))
        self.assertTrue(type(self.n_gram), None)
        self.assertTrue(type(self.n_gram.texts), list)
        self.assertEqual(len(self.n_gram.texts), 4)

        self.assertTrue(hasattr(self.n_gram_w_rel, "texts"))
        self.assertTrue(hasattr(self.n_gram_w_rel, "relevance"))
        self.assertTrue(type(self.n_gram_w_rel.texts), list)
        self.assertEqual(len(self.n_gram_w_rel.texts), 3)
        self.assertTrue(type(self.n_gram_w_rel), float)

    def test_from_proto_and_back(self):
        new = dm.NGram.from_proto(self.n_gram.to_proto())
        self.assertEqual(new.texts, self.n_gram.texts)

        new = dm.NGram.from_proto(self.n_gram_w_rel.to_proto())
        self.assertEqual(new.texts, self.n_gram_w_rel.texts)
        self.assertAlmostEqual(new.relevance, self.n_gram_w_rel.relevance)

    def test_from_json_and_back(self):
        new = dm.NGram.from_json(self.n_gram.to_json())
        self.assertEqual(new.texts, self.n_gram.texts)

        new = dm.NGram.from_json(self.n_gram_w_rel.to_json())
        self.assertEqual(new.texts, self.n_gram_w_rel.texts)
        self.assertAlmostEqual(new.relevance, self.n_gram_w_rel.relevance)


class TestProducerId(TestCaseBase):
    def setUp(self):
        self.producer_id = dm.ProducerId(name="TestProducer", version="1.0.0")

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.producer_id))

    def test_from_proto_and_back(self):
        new = dm.ProducerId.from_proto(self.producer_id.to_proto())
        self.assertEqual(new.name, self.producer_id.name)
        self.assertEqual(new.version, self.producer_id.version)

    def test_from_json_and_back(self):
        new = dm.ProducerId.from_json(self.producer_id.to_json())
        self.assertEqual(new.name, self.producer_id.name)
        self.assertEqual(new.version, self.producer_id.version)
