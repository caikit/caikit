# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Standard
import itertools

# Third Party
import utils

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestKeyword(TestCaseBase):
    def setUp(self):
        self.text = (
            "Barack Obama was the 44th president of the United States of America.  "
            + "Obama graduated from Columbia University in 1983."
        )

        self.keyword1 = dm.Keyword(
            text="Barack Obama",
            relevance=0.98,
            mentions=[dm.Span(70, 75), dm.Span(0, 12)],
            document=self.text,
            count=1,
        )

        self.keyword2 = dm.Keyword(
            "United States of America",
            0.72,
            [
                (43, 67),
            ],
            self.text,
            1,
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.keyword1))
        self.assertTrue(utils.validate_fields(self.keyword2))

    def test_from_proto_and_back(self):
        new = dm.Keyword.from_proto(self.keyword1.to_proto())
        self.assertEqual(new.text, self.keyword1.text)
        self.assertAlmostEqual(new.relevance, self.keyword1.relevance)
        self.assertEqual(new.count, self.keyword1.count)

        for new_mention, mention in itertools.zip_longest(
            new.mentions, self.keyword1.mentions
        ):
            self.assertIsNotNone(new_mention)
            self.assertIsNotNone(mention)
            self.assertEqual(new_mention.begin, mention.begin)
            self.assertEqual(new_mention.end, mention.end)

    def test_from_json_and_back(self):
        new = dm.Keyword.from_json(self.keyword1.to_json())
        self.assertEqual(new.text, self.keyword1.text)
        self.assertAlmostEqual(new.relevance, self.keyword1.relevance)
        self.assertEqual(new.count, self.keyword1.count)

        for new_mention, mention in itertools.zip_longest(
            new.mentions, self.keyword1.mentions
        ):
            self.assertIsNotNone(new_mention)
            self.assertIsNotNone(mention)
            self.assertEqual(new_mention.begin, mention.begin)
            self.assertEqual(new_mention.end, mention.end)

    def test_sort(self):
        self.assertLess(self.keyword1.mentions[0], self.keyword1.mentions[1])
        self.assertLess(self.keyword2, self.keyword1)

    def test_text_extraction(self):
        self.assertEqual(self.keyword1.mentions[0].text, "Barack Obama")
        self.assertEqual(self.keyword1.mentions[1].text, "Obama")
        self.assertEqual(self.keyword2.mentions[0].text, "United States of America")


class TestKeywordsPrediction(TestCaseBase):
    def setUp(self):
        self.text = (
            "Barack Obama was the 44th president of the United States of America.  "
            + "Obama graduated from Columbia University in 1983."
        )

        self.keyword1 = dm.Keyword(
            text="Barack Obama",
            relevance=0.98,
            mentions=[dm.Span(70, 75), dm.Span(0, 12)],
            document=self.text,
            count=1,
        )

        self.keyword2 = dm.Keyword(
            "United States of America",
            0.72,
            [
                (43, 67),
            ],
            self.text,
            1,
        )

        self.keywords_prediction = dm.KeywordsPrediction(
            keywords=[self.keyword2, self.keyword1],
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.keywords_prediction))

    def test_from_proto_and_back(self):
        new = dm.KeywordsPrediction.from_proto(self.keywords_prediction.to_proto())
        for new_kw, orig_kw in zip(new.keywords, self.keywords_prediction.keywords):
            self.assertEqual(new_kw.text, orig_kw.text)
            self.assertAlmostEqual(new_kw.relevance, orig_kw.relevance)
            self.assertAlmostEqual(new_kw.mentions[0].begin, orig_kw.mentions[0].begin)
            self.assertAlmostEqual(new_kw.count, orig_kw.count)

    def test_from_json_and_back(self):
        new = dm.KeywordsPrediction.from_json(self.keywords_prediction.to_json())
        for new_kw, orig_kw in zip(new.keywords, self.keywords_prediction.keywords):
            self.assertEqual(new_kw.text, orig_kw.text)
            self.assertAlmostEqual(new_kw.relevance, orig_kw.relevance)
            self.assertAlmostEqual(new_kw.mentions[0].begin, orig_kw.mentions[0].begin)
            self.assertAlmostEqual(new_kw.count, orig_kw.count)

    def test_sort(self):
        shifted_keywords = zip(
            self.keywords_prediction.keywords[1:],
            self.keywords_prediction.keywords[:-1],
        )

        for a, b in shifted_keywords:
            self.assertLess(a.relevance, b.relevance)
