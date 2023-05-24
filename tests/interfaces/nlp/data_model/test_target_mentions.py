# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Local
from . import utils
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestTargetPhrases(TestCaseBase):
    def setUp(self):
        self.target_phrases = dm.TargetPhrases(["obama", "trump"])

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.target_phrases))

    def test_from_proto_and_back(self):
        new = dm.TargetPhrases.from_proto(self.target_phrases.to_proto())

        self.assertEqual(len(new.targets), 2)
        self.assertEqual(len(new.targets), len(self.target_phrases.targets))
        for target1, target2 in zip(self.target_phrases.targets, new.targets):
            self.assertEqual(target1, target2)

    def test_from_json_and_back(self):
        new = dm.TargetPhrases.from_json(self.target_phrases.to_json())

        self.assertEqual(len(new.targets), 2)
        self.assertEqual(len(new.targets), len(self.target_phrases.targets))
        for target1, target2 in zip(self.target_phrases.targets, new.targets):
            self.assertEqual(target1, target2)


class TestTargetMentions(TestCaseBase):
    def setUp(self):
        self.target_mentions = dm.TargetMentions([(0, 3), (3, 5)])
        self.entity_mentions = [
            dm.EntityMention((0, 5), "Person"),
            dm.EntityMention((10, 20), "Location"),
        ]

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.target_mentions))

    def test_from_proto_and_back(self):
        new = dm.TargetMentions.from_proto(self.target_mentions.to_proto())

        self.assertEqual(len(new.spans), 2)
        self.assertEqual(len(new.spans), len(self.target_mentions.spans))
        for span1, span2 in zip(self.target_mentions.spans, new.spans):
            self.assertTrue(isinstance(span2, dm.Span))
            self.assertEqual(span1.begin, span2.begin)
            self.assertEqual(span1.end, span2.end)

    def test_from_json_and_back(self):
        new = dm.TargetMentions.from_json(self.target_mentions.to_json())

        self.assertEqual(len(new.spans), 2)
        self.assertEqual(len(new.spans), len(self.target_mentions.spans))
        for span1, span2 in zip(self.target_mentions.spans, new.spans):
            self.assertTrue(isinstance(span2, dm.Span))
            self.assertEqual(span1.begin, span2.begin)
            self.assertEqual(span1.end, span2.end)

    def test_span_types(self):
        for span in self.target_mentions.spans:
            self.assertTrue(isinstance(span, dm.Span))

        target_mentions_from_mixed_tuples = dm.target_mentions = dm.TargetMentions(
            ((0, 1), dm.Span(2, 4), (8, 10), dm.Span(20, 30))
        )

        for span in target_mentions_from_mixed_tuples.spans:
            self.assertTrue(isinstance(span, dm.Span))

    def test_from_entity_mentions(self):
        target_mentions = dm.TargetMentions.from_entity_mentions(self.entity_mentions)
        self.assertIsInstance(target_mentions, dm.TargetMentions)
        self.assertEqual(len(target_mentions.spans), len(self.entity_mentions))
        for idx, mention in enumerate(self.entity_mentions):
            self.assertEqual(target_mentions.spans[idx], mention.span)


class TestTargetMentionsPrediction(TestCaseBase):
    def setUp(self):
        self.target_mentions_prediction = dm.TargetMentionsPrediction(
            [[(0, 1), (2, 4)], dm.TargetMentions([(8, 10), (20, 30)])]
        )

        text = "My name is Samuel.  Sam for short.  Sam McRam I am.  I like green eggs and ham."
        syn_doc = dm.SyntaxPrediction(text)

        mention1 = dm.EntityMention((11, 17), "person", document=syn_doc)
        mention2 = dm.EntityMention((20, 23), "person", document=syn_doc)
        mention3 = dm.EntityMention((36, 39), "person", document=syn_doc)

        self.entities_prediction = dm.EntitiesPrediction(
            entities=[
                dm.Entity(
                    mentions=(mention1, mention2, mention3),
                    type="person",
                    text="Samual McRam",
                )
            ]
            * 3,
            producer_id=dm.ProducerId(name="Test", version="1.0.0"),
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.target_mentions_prediction))

    def test_from_proto_and_back(self):
        new = dm.TargetMentionsPrediction.from_proto(
            self.target_mentions_prediction.to_proto()
        )

        self.assertEqual(len(new.targets), 2)
        self.assertEqual(len(new.targets), len(self.target_mentions_prediction.targets))
        for target1, target2 in zip(
            new.targets, self.target_mentions_prediction.targets
        ):
            self.assertEqual(len(target1.spans), len(target2.spans))
            for span1, span2 in zip(target2.spans, target2.spans):
                self.assertEqual(span1.begin, span2.begin)
                self.assertEqual(span1.end, span2.end)

    def test_from_json_and_back(self):
        new = dm.TargetMentionsPrediction.from_json(
            self.target_mentions_prediction.to_json()
        )

        self.assertEqual(len(new.targets), 2)
        self.assertEqual(len(new.targets), len(self.target_mentions_prediction.targets))
        for target1, target2 in zip(
            new.targets, self.target_mentions_prediction.targets
        ):
            self.assertEqual(len(target1.spans), len(target2.spans))
            for span1, span2 in zip(target2.spans, target2.spans):
                self.assertEqual(span1.begin, span2.begin)
                self.assertEqual(span1.end, span2.end)

    def test_from_entities_prediction(self):
        tm_pred = dm.TargetMentionsPrediction.from_entities_prediction(
            self.entities_prediction
        )
        self.assertIsInstance(tm_pred, dm.TargetMentionsPrediction)
        self.assertEqual(len(tm_pred.targets), len(self.entities_prediction.entities))
        for idx, entity in enumerate(self.entities_prediction.entities):
            self.assertEqual(len(tm_pred.targets[idx].spans), len(entity.mentions))
        self.assertEqual(tm_pred.producer_id, self.entities_prediction.producer_id)
