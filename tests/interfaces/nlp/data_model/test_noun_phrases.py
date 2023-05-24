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


class TestNounPhrase(TestCaseBase):
    def setUp(self):
        self.noun_phrase = dm.NounPhrase(dm.Span(0, 11, text="Hello World"))
        self.noun_phrase_minimal = dm.NounPhrase((0, 20))

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.noun_phrase))
        self.assertTrue(utils.validate_fields(self.noun_phrase_minimal))

    def test_from_proto_and_back(self):
        new = dm.NounPhrase.from_proto(self.noun_phrase.to_proto())
        self.assertEqual(new.span.begin, self.noun_phrase.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase.span.end)

        new = dm.NounPhrase.from_proto(self.noun_phrase_minimal.to_proto())
        self.assertEqual(new.span.begin, self.noun_phrase_minimal.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase_minimal.span.end)

    def test_from_json_and_back(self):
        new = dm.NounPhrase.from_json(self.noun_phrase.to_json())
        self.assertEqual(new.span.begin, self.noun_phrase.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase.span.end)

        new = dm.NounPhrase.from_json(self.noun_phrase_minimal.to_json())
        self.assertEqual(new.span.begin, self.noun_phrase_minimal.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase_minimal.span.end)
