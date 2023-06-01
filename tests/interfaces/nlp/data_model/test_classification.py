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
import numpy as np

# Local
from caikit.core.toolkit.errors import DataValidationError
from caikit.interfaces.common.data_model import ProducerId
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestClassInfo(TestCaseBase):
    def setUp(self):
        self.class1 = dm.ClassInfo("temperature", 0.71)
        self.class2 = dm.ClassInfo("conditions", 0.98)

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.class1))
        self.assertTrue(self.validate_fields(self.class2))

    def test_from_proto_and_back(self):
        new = dm.ClassInfo.from_proto(self.class1.to_proto())
        self.assertEqual(new.class_name, self.class1.class_name)
        self.assertAlmostEqual(new.confidence, self.class1.confidence)

    def test_from_json_and_back(self):
        new = dm.ClassInfo.from_json(self.class1.to_json())
        self.assertEqual(new.class_name, self.class1.class_name)
        self.assertAlmostEqual(new.confidence, self.class1.confidence)

    def test_sort(self):
        self.assertLess(self.class1, self.class2)

    def test_float_32_serialization(self):
        """Ensure that if we create a classification object with a Float32, which is not JSON
        serializable, it's cast to a JSON serializable float so that we can serialize, e.g., for
        printing."""
        float32_classinfo = dm.ClassInfo("temperature", np.float32(0.5))
        float32_classinfo.to_json()


class TestClassificationPrediction(TestCaseBase):
    def setUp(self):
        self.classification_prediction = dm.ClassificationPrediction(
            classes=[
                dm.ClassInfo("temperature", 0.71),
                dm.ClassInfo("conditions", 0.98),
            ],
            producer_id=ProducerId("Test", "1.2.3"),
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.classification_prediction))

    def test_from_proto_and_back(self):
        new = dm.ClassificationPrediction.from_proto(
            self.classification_prediction.to_proto()
        )
        for new_class, orig_class in zip(
            new.classes, self.classification_prediction.classes
        ):
            self.assertEqual(new_class.class_name, orig_class.class_name)
            self.assertAlmostEqual(new_class.confidence, orig_class.confidence)

    def test_from_json_and_back(self):
        new = dm.ClassificationPrediction.from_json(
            self.classification_prediction.to_json()
        )
        for new_class, orig_class in zip(
            new.classes, self.classification_prediction.classes
        ):
            self.assertEqual(new_class.class_name, orig_class.class_name)
            self.assertAlmostEqual(new_class.confidence, orig_class.confidence)

    def test_sort(self):
        shifted_classifications = zip(
            self.classification_prediction.classes[1:],
            self.classification_prediction.classes[:-1],
        )

        for a, b in shifted_classifications:
            self.assertLess(a.confidence, b.confidence)


class ClassificationTrainRecord(TestCaseBase):
    def setUp(self):
        self.classification_train_record = dm.ClassificationTrainRecord(
            text="It is 20 degrees today", labels=["temperature"]
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.classification_train_record))

    def test_from_proto_and_back(self):
        new = dm.ClassificationTrainRecord.from_proto(
            self.classification_train_record.to_proto()
        )
        self.assertEqual(new.text, self.classification_train_record.text)
        self.assertEqual(new.labels, self.classification_train_record.labels)

    def test_from_json_and_back(self):
        new = dm.ClassificationTrainRecord.from_json(
            self.classification_train_record.to_json()
        )
        self.assertEqual(new.text, self.classification_train_record.text)
        self.assertEqual(new.labels, self.classification_train_record.labels)

    def test_from_data_obj_valid_list(self):
        list_obj = ["It is 20 degrees today", "temperature"]
        new = dm.ClassificationTrainRecord.from_data_obj(list_obj)
        self.assertEqual(new.text, list_obj[0])
        self.assertEqual(new.labels, [list_obj[1]])

        list_obj_2 = ["It is 20 degrees today", "temperature", "present"]
        new = dm.ClassificationTrainRecord.from_data_obj(list_obj_2)
        self.assertEqual(new.text, list_obj_2[0])
        self.assertEqual(new.labels, [list_obj_2[1], list_obj_2[2]])

        list_obj_3 = ["It is 20 degrees today", ["temperature", "present"]]
        new = dm.ClassificationTrainRecord.from_data_obj(list_obj_3)
        self.assertEqual(new.text, list_obj_3[0])
        self.assertEqual(new.labels, list_obj_3[1])

    def test_from_data_obj_valid_tuples(self):
        tuple_obj = ("It is 20 degrees today", "temperature")
        new = dm.ClassificationTrainRecord.from_data_obj(tuple_obj)
        self.assertEqual(new.text, tuple_obj[0])
        self.assertEqual(new.labels, [tuple_obj[1]])

        tuple_obj_2 = ("It is 20 degrees today", "temperature", "present")
        new = dm.ClassificationTrainRecord.from_data_obj(tuple_obj_2)
        self.assertEqual(new.text, tuple_obj_2[0])
        self.assertEqual(new.labels, [tuple_obj_2[1], tuple_obj_2[2]])

        tuple_obj_3 = ("It is 20 degrees today", ["temperature", "present"])
        new = dm.ClassificationTrainRecord.from_data_obj(tuple_obj_3)
        self.assertEqual(new.text, tuple_obj_3[0])
        self.assertEqual(new.labels, tuple_obj_3[1])

    def test_from_data_obj_valid_dict(self):
        dict_obj = {
            "text": "It is 20 degrees today",
            "labels": ["temperature", "present"],
        }
        new = dm.ClassificationTrainRecord.from_data_obj(dict_obj)
        self.assertEqual(new.text, dict_obj["text"])
        self.assertEqual(new.labels, dict_obj["labels"])

    def test_from_data_obj_valid_TrainRecord(self):
        new = dm.ClassificationTrainRecord.from_data_obj(
            self.classification_train_record
        )
        self.assertEqual(new.text, self.classification_train_record.text)
        self.assertEqual(new.labels, self.classification_train_record.labels)

    def test_from_data_obj_invalid_iterables(self):
        # Iterables with only one element
        list_obj = ["It is 20 degrees today"]
        with self.assertRaises(DataValidationError):
            dm.ClassificationTrainRecord.from_data_obj(list_obj)
        tuple_obj = ("It is 20 degrees today",)
        with self.assertRaises(DataValidationError):
            dm.ClassificationTrainRecord.from_data_obj(tuple_obj)

        # Iterables with wrong type of elements
        list_obj = ["It is 20 degrees today", 1]
        with self.assertRaises(DataValidationError):
            dm.ClassificationTrainRecord.from_data_obj(list_obj)

    def test_from_data_obj_invalid_dict(self):
        # Dictionary with missing labels
        dict_obj = {
            "document": "It is 20 degrees today",
            "labels": ["temperature", "present"],
        }
        with self.assertRaises(DataValidationError):
            dm.ClassificationTrainRecord.from_data_obj(dict_obj)

        # Dictionary with wrong type of elements
        dict_obj = {"text": "It is 20 degrees today", "labels": "temperature"}
        with self.assertRaises(DataValidationError):
            dm.ClassificationTrainRecord.from_data_obj(dict_obj)
