# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party

# Local
from caikit.core.toolkit.errors import DataValidationError
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestEntityMention(TestCaseBase):
    def setUp(self):
        self.mention = dm.EntityMention(
            (10, 20),
            "Person",
            producer_id=dm.ProducerId("Test", "1.0.0"),
            confidence=0.314159,
            mention_type=dm.EntityMentionType.MENTT_NOM.value,
            mention_class=dm.EntityMentionClass.MENTC_SPC.value,
            role="developer",
        )

        self.producer_this = dm.ProducerId(name="this", version="0.0.1")
        self.producer_other = dm.ProducerId(name="other", version="0.0.1")
        self.producer_long = dm.ProducerId(name="long", version="0.0.1")

        self.this_mention = dm.EntityMention(
            (40, 45), "number", producer_id=self.producer_this
        )
        self.other_mention = dm.EntityMention(
            (43, 48), "person", producer_id=self.producer_other
        )

        self.long_mention = dm.EntityMention(
            (1, 500), "location", producer_id=self.producer_long
        )

        self.mention_minimal = dm.EntityMention((10, 20), "person")

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.mention))

    def test_from_proto_and_back(self):
        new = dm.EntityMention.from_proto(self.mention.to_proto())
        self.assertEqual(new.span.begin, self.mention.span.begin)
        self.assertEqual(new.span.end, self.mention.span.end)
        self.assertEqual(new.type, self.mention.type)
        self.assertEqual(new.producer_id.name, self.mention.producer_id.name)
        self.assertAlmostEqual(new.confidence, self.mention.confidence)
        self.assertEqual(new.mention_type, self.mention.mention_type)
        self.assertEqual(new.mention_class, self.mention.mention_class)
        self.assertEqual(new.role, self.mention.role)

        new = dm.EntityMention.from_proto(self.mention_minimal.to_proto())
        self.assertEqual(new.span.begin, self.mention_minimal.span.begin)
        self.assertEqual(new.span.end, self.mention_minimal.span.end)
        self.assertEqual(new.type, self.mention_minimal.type)
        self.assertEqual(new.producer_id, self.mention_minimal.producer_id)
        self.assertEqual(new.producer_id, None)
        self.assertAlmostEqual(new.confidence, self.mention_minimal.confidence)
        self.assertAlmostEqual(new.confidence, 0.0)
        self.assertEqual(new.mention_type, self.mention_minimal.mention_type)
        self.assertEqual(new.mention_type, 0)
        self.assertEqual(new.mention_class, self.mention_minimal.mention_class)
        self.assertEqual(new.mention_class, 0)
        self.assertEqual(new.role, self.mention_minimal.role)
        self.assertEqual(new.role, "")

    def test_from_json_and_back(self):
        new = dm.EntityMention.from_json(self.mention.to_json())
        self.assertEqual(new.span.begin, self.mention.span.begin)
        self.assertEqual(new.span.end, self.mention.span.end)
        self.assertEqual(new.type, self.mention.type)
        self.assertEqual(new.producer_id.name, self.mention.producer_id.name)
        self.assertAlmostEqual(new.confidence, self.mention.confidence)
        self.assertEqual(new.mention_type, self.mention.mention_type)
        self.assertEqual(new.mention_class, self.mention.mention_class)
        self.assertEqual(new.role, self.mention.role)

        new = dm.EntityMention.from_json(self.mention_minimal.to_json())
        self.assertEqual(new.span.begin, self.mention_minimal.span.begin)
        self.assertEqual(new.span.end, self.mention_minimal.span.end)
        self.assertEqual(new.type, self.mention_minimal.type)
        self.assertEqual(new.producer_id, self.mention_minimal.producer_id)
        self.assertEqual(new.producer_id, None)
        self.assertAlmostEqual(new.confidence, self.mention_minimal.confidence)
        self.assertAlmostEqual(new.confidence, 0.0)
        self.assertEqual(new.mention_type, self.mention_minimal.mention_type)
        self.assertEqual(new.mention_type, 0)
        self.assertEqual(new.mention_class, self.mention_minimal.mention_class)
        self.assertEqual(new.mention_class, 0)
        self.assertEqual(new.role, self.mention_minimal.role)
        self.assertEqual(new.role, "")

    def test_extract_text(self):
        text = "Hello World!  My name is John Doe and I live in Springfield."
        raw_doc = dm.RawDocument(text)
        syn_doc = dm.SyntaxPrediction(text)

        john_doe = dm.EntityMention((25, 33), "person", document=text)
        self.assertEqual(john_doe.span.text, "John Doe")

        springfield = dm.EntityMention(
            dm.Span(48, 59, text="hello world"), type="location", document=raw_doc
        )
        self.assertEqual(springfield.span.text, "Springfield")

        world = dm.EntityMention((6, 11), type="planet", document=syn_doc)
        self.assertEqual(world.text, "World")

    def test_compare_priority(self):
        # longer mentions get higher priority, all other things equal
        self.assertTrue(self.this_mention.compare_priority(self.other_mention))
        self.assertTrue(self.other_mention.compare_priority(self.this_mention))

        self.assertTrue(self.long_mention.compare_priority(self.other_mention))
        self.assertTrue(self.long_mention.compare_priority(self.this_mention))

        self.assertFalse(self.this_mention.compare_priority(self.long_mention))
        self.assertFalse(self.other_mention.compare_priority(self.long_mention))

    def test_favor_types_compare_priority(self):
        self.assertFalse(
            self.this_mention.compare_priority(
                self.other_mention, favor_types=[self.other_mention.type]
            )
        )
        self.assertFalse(
            self.other_mention.compare_priority(
                self.this_mention, favor_types=[self.this_mention.type]
            )
        )

        self.assertTrue(
            self.this_mention.compare_priority(
                self.other_mention, favor_types=[self.this_mention.type]
            )
        )
        self.assertTrue(
            self.other_mention.compare_priority(
                self.this_mention, favor_types=[self.other_mention.type]
            )
        )

        self.assertFalse(
            self.long_mention.compare_priority(
                self.other_mention, favor_types=[self.other_mention.type]
            )
        )
        self.assertFalse(
            self.long_mention.compare_priority(
                self.this_mention, favor_types=[self.this_mention.type]
            )
        )

    def test_disfavor_types_compare_priority(self):
        self.assertTrue(
            self.this_mention.compare_priority(
                self.other_mention, disfavor_types=[self.other_mention.type]
            )
        )
        self.assertTrue(
            self.other_mention.compare_priority(
                self.this_mention, disfavor_types=[self.this_mention.type]
            )
        )

        self.assertFalse(
            self.this_mention.compare_priority(
                self.other_mention, disfavor_types=[self.this_mention.type]
            )
        )
        self.assertFalse(
            self.other_mention.compare_priority(
                self.this_mention, disfavor_types=[self.other_mention.type]
            )
        )

        self.assertTrue(
            self.this_mention.compare_priority(
                self.long_mention, disfavor_types=[self.long_mention.type]
            )
        )
        self.assertTrue(
            self.other_mention.compare_priority(
                self.long_mention, disfavor_types=[self.long_mention.type]
            )
        )

    def test_model_priorities_compare_priority(self):
        model_priorities = [self.producer_this, self.producer_other, self.producer_long]
        self.assertTrue(
            self.this_mention.compare_priority(
                self.other_mention, model_priorities=model_priorities
            )
        )
        self.assertFalse(
            self.other_mention.compare_priority(
                self.this_mention, model_priorities=model_priorities
            )
        )

        model_priorities = [self.producer_other, self.producer_this, self.producer_long]
        self.assertFalse(
            self.this_mention.compare_priority(
                self.other_mention, model_priorities=model_priorities
            )
        )
        self.assertTrue(
            self.other_mention.compare_priority(
                self.this_mention, model_priorities=model_priorities
            )
        )

        # should break the span length assumptions / execution order
        self.assertFalse(
            self.long_mention.compare_priority(
                self.other_mention, model_priorities=model_priorities
            )
        )
        self.assertFalse(
            self.long_mention.compare_priority(
                self.this_mention, model_priorities=model_priorities
            )
        )

        self.assertTrue(
            self.this_mention.compare_priority(
                self.long_mention, model_priorities=model_priorities
            )
        )
        self.assertTrue(
            self.other_mention.compare_priority(
                self.long_mention, model_priorities=model_priorities
            )
        )


class TestEntityMentionsPrediction(TestCaseBase):
    def setUp(self):
        text = "My name is Samual.  Sam for short.  Sam McRam I am.  I like green eggs and ham."
        syn_doc = dm.SyntaxPrediction(text)

        mention1 = dm.EntityMention((11, 17), "person", document=syn_doc)
        mention2 = dm.EntityMention((20, 23), "person", document=syn_doc)
        mention3 = dm.EntityMention((36, 39), "person", document=syn_doc)

        self.mentions_prediction = dm.EntityMentionsPrediction(
            mentions=[mention1, mention2, mention3],
            producer_id=dm.ProducerId(name="Test", version="1.0.0"),
        )

        self.producer_4a = dm.ProducerId(name="4a", version="0.0.1")
        self.producer_4b = dm.ProducerId(name="4b", version="0.0.1")
        self.mention4a = dm.EntityMention(
            (40, 45), "number", producer_id=self.producer_4a
        )
        self.mention4b = dm.EntityMention(
            (43, 48), "person", producer_id=self.producer_4b
        )

        self.ments_pred_conflict = dm.EntityMentionsPrediction(
            mentions=[mention1, mention2, mention3, self.mention4a, self.mention4b],
            producer_id=dm.ProducerId(name="Test2", version="1.0.0"),
        )

    def test___add__(self):
        new_mention_prediction = self.mentions_prediction + self.ments_pred_conflict
        self.assertTrue(new_mention_prediction is not self.mentions_prediction)
        self.assertTrue(new_mention_prediction is not self.ments_pred_conflict)
        self.assertTrue(
            all(
                ment in new_mention_prediction.mentions
                for ment in self.mentions_prediction.mentions
            )
        )
        self.assertTrue(
            all(
                ment in new_mention_prediction.mentions
                for ment in self.ments_pred_conflict.mentions
            )
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.mentions_prediction))

    def test_from_proto_and_back(self):
        new = dm.EntityMentionsPrediction.from_proto(
            self.mentions_prediction.to_proto()
        )
        self.assertEqual(len(new.mentions), len(self.mentions_prediction.mentions))
        self.assertEqual(
            new.mentions[0].span.begin, self.mentions_prediction.mentions[0].span.begin
        )
        self.assertEqual(
            new.producer_id.name, self.mentions_prediction.producer_id.name
        )

    def test_from_json_and_back(self):
        new = dm.EntityMentionsPrediction.from_json(self.mentions_prediction.to_json())
        self.assertEqual(len(new.mentions), len(self.mentions_prediction.mentions))
        self.assertEqual(
            new.mentions[0].span.begin, self.mentions_prediction.mentions[0].span.begin
        )
        self.assertEqual(
            new.producer_id.name, self.mentions_prediction.producer_id.name
        )

    def test_non_conflicting_resolve(self):
        # test non-conflicing mentions
        resolved_non_conflicting = self.mentions_prediction.resolve_conflicts()
        self.assertEqual(
            len(self.mentions_prediction.mentions),
            len(resolved_non_conflicting.mentions),
        )
        # should be new instance
        self.assertFalse(resolved_non_conflicting is self.mentions_prediction)

    def test_conflicting_resolve(self):
        # test conflicting mentions get removed
        resolved_mentions_pred = self.ments_pred_conflict.resolve_conflicts()

        self.assertFalse(resolved_mentions_pred is self.ments_pred_conflict)
        self.assertNotIn(self.mention4a, resolved_mentions_pred.mentions)
        self.assertIn(self.mention4b, resolved_mentions_pred.mentions)
        self.assertEqual(4, len(resolved_mentions_pred.mentions))

    def test_favor_types_conflict_resolve(self):
        favor_types = [self.mention4b.type]
        # test conflicting mentions get removed
        resolved_mentions_pred = self.ments_pred_conflict.resolve_conflicts(
            favor_types=favor_types
        )

        self.assertFalse(resolved_mentions_pred is self.ments_pred_conflict)
        self.assertNotIn(self.mention4a, resolved_mentions_pred.mentions)
        self.assertIn(self.mention4b, resolved_mentions_pred.mentions)
        self.assertEqual(4, len(resolved_mentions_pred.mentions))

    def test_disfavor_types_conflict_resolve(self):
        disfavor_types = [self.mention4a.type]
        # test conflicting mentions get removed
        resolved_mentions_pred = self.ments_pred_conflict.resolve_conflicts(
            disfavor_types=disfavor_types
        )

        self.assertFalse(resolved_mentions_pred is self.ments_pred_conflict)
        self.assertNotIn(self.mention4a, resolved_mentions_pred.mentions)
        self.assertIn(self.mention4b, resolved_mentions_pred.mentions)
        self.assertEqual(4, len(resolved_mentions_pred.mentions))

    def test_model_priorities_conflict_resolve(self):
        model_priorities = [self.producer_4b, self.producer_4a]
        # test conflicting mentions get removed
        resolved_mentions_pred = self.ments_pred_conflict.resolve_conflicts(
            model_priorities=model_priorities
        )

        self.assertFalse(resolved_mentions_pred is self.ments_pred_conflict)
        self.assertNotIn(self.mention4a, resolved_mentions_pred.mentions)
        self.assertIn(self.mention4b, resolved_mentions_pred.mentions)
        self.assertEqual(4, len(resolved_mentions_pred.mentions))


class TestEntityDisambiguation(TestCaseBase):
    def setUp(self):
        self.disambig = dm.EntityDisambiguation(
            name="Barack Obama",
            subtypes=["senator", "president"],
            dbpedia_resource="http://dbpedia.org/resource/Barack_Obama",
        )

        self.disambig_minimal = dm.EntityDisambiguation("Barack Obama")

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.disambig))
        self.assertTrue(self.validate_fields(self.disambig_minimal))

    def test_from_proto_and_back(self):
        new = dm.EntityDisambiguation.from_proto(self.disambig.to_proto())
        self.assertEqual(new.name, self.disambig.name)
        self.assertEqual(new.subtypes, self.disambig.subtypes)
        self.assertEqual(new.dbpedia_resource, self.disambig.dbpedia_resource)

        new = dm.EntityDisambiguation.from_proto(self.disambig_minimal.to_proto())
        self.assertEqual(new.name, self.disambig_minimal.name)
        self.assertEqual(new.subtypes, self.disambig_minimal.subtypes)
        self.assertEqual(new.dbpedia_resource, self.disambig_minimal.dbpedia_resource)


class TestEntity(TestCaseBase):
    def setUp(self):
        text = "My name is Samual.  Sam for short.  Sam McRam I am.  I like green eggs and ham."
        syn_doc = dm.SyntaxPrediction(text)

        mention1 = dm.EntityMention((11, 17), "person", document=syn_doc)
        mention2 = dm.EntityMention((20, 23), "person", document=syn_doc)
        mention3 = dm.EntityMention((36, 39), "person", document=syn_doc)

        self.entity = dm.Entity(
            mentions=(mention1, mention2, mention3),
            text="Samual McRam",
            type="person",
            confidence=0.98,
            disambiguation=dm.EntityDisambiguation(
                name="Samual McRam",
                subtypes=("celebrity", "goofball", "green egg eater"),
                dbpedia_resource="http://dbpedia.org/resource/Sam_McRam",
            ),
        )

        self.entity_minimal = dm.Entity(
            mentions=(mention1, mention2, mention3), type="person", text="Samual McRam"
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.entity))
        self.assertTrue(self.validate_fields(self.entity_minimal))

    def test_from_proto_and_back(self):
        new = dm.Entity.from_proto(self.entity.to_proto())
        self.assertEqual(new.text, self.entity.text)

        new = dm.Entity.from_proto(self.entity.to_proto())
        self.assertEqual(new.text, self.entity_minimal.text)


class TestEntitiesPrediction(TestCaseBase):
    def setUp(self):
        text = "My name is Samual.  Sam for short.  Sam McRam I am.  I like green eggs and ham."
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
        self.assertTrue(self.validate_fields(self.entities_prediction))

    def test_from_proto_and_back(self):
        new = dm.EntitiesPrediction.from_proto(self.entities_prediction.to_proto())
        self.assertEqual(len(new.entities), len(self.entities_prediction.entities))
        self.assertEqual(
            new.entities[0].mentions[0].span.begin,
            self.entities_prediction.entities[0].mentions[0].span.begin,
        )
        self.assertEqual(
            new.producer_id.name, self.entities_prediction.producer_id.name
        )


class TestEntityMentionAnnotation(TestCaseBase):
    def setUp(self):
        self.entity_mentions_annotation = dm.EntityMentionAnnotation(
            text="20 degrees",
            type="temperature",
            location=(6, 16),
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.entity_mentions_annotation))

    def test_from_proto_and_back(self):
        new = dm.EntityMentionAnnotation.from_proto(
            self.entity_mentions_annotation.to_proto()
        )
        self.assertEqual(new.text, self.entity_mentions_annotation.text)
        self.assertEqual(new.type, self.entity_mentions_annotation.type)
        self.assertEqual(
            new.location.begin, self.entity_mentions_annotation.location.begin
        )
        self.assertEqual(new.location.end, self.entity_mentions_annotation.location.end)


class TestEntityMentionsTrainRecord(TestCaseBase):
    def setUp(self):
        self.entity_mentions_train_record = dm.EntityMentionsTrainRecord(
            text="It is 20 degrees today",
            mentions=[
                dm.EntityMentionAnnotation(
                    text="20 degrees",
                    type="temperature",
                    location=(6, 16),
                )
            ],
            language_code="en",
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.entity_mentions_train_record))

    def test_from_proto_and_back(self):
        new = dm.EntityMentionsTrainRecord.from_proto(
            self.entity_mentions_train_record.to_proto()
        )
        self.assertEqual(new.text, self.entity_mentions_train_record.text)
        self.assertEqual(
            new.language_code, self.entity_mentions_train_record.language_code
        )
        self.assertEqual(
            new.mentions[0].text, self.entity_mentions_train_record.mentions[0].text
        )
        self.assertEqual(
            new.mentions[0].type, self.entity_mentions_train_record.mentions[0].type
        )
        self.assertEqual(
            new.mentions[0].location.begin,
            self.entity_mentions_train_record.mentions[0].location.begin,
        )
        self.assertEqual(
            new.mentions[0].location.end,
            self.entity_mentions_train_record.mentions[0].location.end,
        )

    def test_from_json_and_back(self):
        new = dm.EntityMentionsTrainRecord.from_json(
            self.entity_mentions_train_record.to_json()
        )
        self.assertEqual(new.text, self.entity_mentions_train_record.text)
        self.assertEqual(
            new.language_code, self.entity_mentions_train_record.language_code
        )
        self.assertEqual(
            new.mentions[0].text, self.entity_mentions_train_record.mentions[0].text
        )
        self.assertEqual(
            new.mentions[0].type, self.entity_mentions_train_record.mentions[0].type
        )
        self.assertEqual(
            new.mentions[0].location.begin,
            self.entity_mentions_train_record.mentions[0].location.begin,
        )
        self.assertEqual(
            new.mentions[0].location.end,
            self.entity_mentions_train_record.mentions[0].location.end,
        )

    def test_from_data_obj_valid_dict(self):
        dict_obj = {
            "text": "It is 20 degrees today",
            "mentions": [
                {
                    "location": {"begin": 6, "end": 16},
                    "text": "20 degrees",
                    "type": "temperature",
                }
            ],
        }
        new = dm.EntityMentionsTrainRecord.from_data_obj(dict_obj)
        self.assertEqual(new.text, dict_obj["text"])
        self.assertEqual(len(new.mentions), len(dict_obj["mentions"]))

        dict_obj_lang = {
            "text": "It is 20 degrees today",
            "mentions": [
                {
                    "location": {"begin": 6, "end": 16},
                    "text": "20 degrees",
                    "type": "temperature",
                },
                {"location": {"begin": 17, "end": 22}, "text": "today", "type": "time"},
            ],
            "language_code": "en",
        }
        new = dm.EntityMentionsTrainRecord.from_data_obj(dict_obj_lang)
        self.assertEqual(new.text, dict_obj_lang["text"])
        self.assertEqual(new.language_code, dict_obj_lang["language_code"])
        self.assertEqual(len(new.mentions), len(dict_obj_lang["mentions"]))
        self.assertEqual(new.mentions[0].text, dict_obj_lang["mentions"][0]["text"])
        self.assertEqual(new.mentions[0].type, dict_obj_lang["mentions"][0]["type"])
        self.assertEqual(
            new.mentions[0].location.begin,
            dict_obj_lang["mentions"][0]["location"]["begin"],
        )
        self.assertEqual(
            new.mentions[0].location.end,
            dict_obj_lang["mentions"][0]["location"]["end"],
        )
        self.assertEqual(new.mentions[1].text, dict_obj_lang["mentions"][1]["text"])
        self.assertEqual(new.mentions[1].type, dict_obj_lang["mentions"][1]["type"])
        self.assertEqual(
            new.mentions[1].location.begin,
            dict_obj_lang["mentions"][1]["location"]["begin"],
        )
        self.assertEqual(
            new.mentions[1].location.end,
            dict_obj_lang["mentions"][1]["location"]["end"],
        )

    def test_from_data_obj_valid_TrainRecord(self):
        new = dm.EntityMentionsTrainRecord.from_data_obj(
            self.entity_mentions_train_record
        )
        self.assertEqual(new.text, self.entity_mentions_train_record.text)
        self.assertEqual(new.mentions, self.entity_mentions_train_record.mentions)
        self.assertEqual(
            new.language_code, self.entity_mentions_train_record.language_code
        )

    def test_from_data_obj_invalid_dict(self):
        # Dictionary with missing mentions
        dict_obj = {"text": "It is 20 degrees today"}
        with self.assertRaises(DataValidationError):
            dm.EntityMentionsTrainRecord.from_data_obj(dict_obj)

        # Dictionary with wrong type of elements
        dict_obj = {"text": "It is 20 degrees today", "mentions": "temperature"}
        with self.assertRaises(DataValidationError):
            dm.EntityMentionsTrainRecord.from_data_obj(dict_obj)

        # Dictionary with missing span in mentions annotation
        dict_obj = {
            "text": "It is 20 degrees today",
            "mentions": [{"text": "20 degrees", "type": "temperature"}],
        }
        with self.assertRaises(DataValidationError):
            dm.EntityMentionsTrainRecord.from_data_obj(dict_obj)
