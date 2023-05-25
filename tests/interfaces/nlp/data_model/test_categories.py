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
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestRelevantText(TestCaseBase):
    def setUp(self):
        self.expl = dm.RelevantText("ibm")

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.expl))

    def test_from_proto_and_back(self):
        new = dm.RelevantText.from_proto(self.expl.to_proto())
        self.assertEqual(new.text, self.expl.text)


class TestCategory(TestCaseBase):
    def setUp(self):
        self.category1 = dm.Category(
            ["sports", "footbag"], 0.71, [dm.RelevantText("basketball")]
        )
        self.category2 = dm.Category(["sports", "baseketball"], 0.98, None)

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.category1))
        self.assertTrue(self.validate_fields(self.category2))

    def test_from_proto_and_back(self):
        new = dm.Category.from_proto(self.category1.to_proto())
        self.assertEqual(new.labels, self.category1.labels)
        self.assertEqual(new.explanation[0].text, self.category1.explanation[0].text)
        self.assertAlmostEqual(new.score, self.category1.score)

    def test_from_json_and_back(self):
        new = dm.Category.from_json(self.category1.to_json())
        self.assertEqual(new.labels, self.category1.labels)
        self.assertEqual(new.explanation[0].text, self.category1.explanation[0].text)
        self.assertAlmostEqual(new.score, self.category1.score)

    def test_sort(self):
        self.assertLess(self.category1, self.category2)

    def test_extract_subhierarchy_from_str_happy_values(self):
        """Ensure that subhierachy extraction helper works as advertised on happy cases."""
        cases = [
            [{"str_labels": "/foo", "level": None}, "/foo"],
            [{"str_labels": "/foo/bar/baz", "level": None}, "/foo/bar/baz"],
            [{"str_labels": "/foo/bar/baz", "level": 1}, "/foo"],
            [{"str_labels": "/foo/bar/baz", "level": 2}, "/foo/bar"],
            [{"str_labels": "/foo/bar/baz", "level": 5}, "/foo/bar/baz"],
        ]
        for (cat_inputs, expected_output) in cases:
            actual_output = dm.Category.extract_subhierarchy_from_str(**cat_inputs)
            self.assertEqual(actual_output, expected_output)

    def test_extract_subhierarchy_from_str_sad_values(self):
        """Ensure that bad values are handled correctly in the subhierachy extraction helper."""
        with self.assertRaises(ValueError):
            dm.Category.extract_subhierarchy_from_str("/foo/bar", -1)

    def test_extract_subhierarchy_requires_str_cats(self):
        """Ensure that the subhierarchy extraction helper raises if no str cats are given."""
        with self.assertRaises(TypeError):
            dm.Category.extract_subhierarchy_from_str(["foo", "bar"], 1)

    def test_extract_subhierarchy_requires_numeric_levels(self):
        """Ensure that the subhierarchy extraction helper requires numeric levels."""
        with self.assertRaises(TypeError):
            dm.Category.extract_subhierarchy_from_str("/foo/bar", "1")

    def test_get_label_hierarchy_as_str(self):
        """Ensure that the helper for pulling labels out as strings works as expected."""
        self.assertEqual("/", dm.Category([], 0.0).get_label_hierarchy_as_str())
        self.assertEqual("/foo", dm.Category(["foo"], 0.0).get_label_hierarchy_as_str())
        self.assertEqual(
            "/foo/bar", dm.Category(["foo", "bar"], 0.0).get_label_hierarchy_as_str()
        )


class TestCategoriesPrediction(TestCaseBase):
    def setUp(self):
        self.categories_prediction = dm.CategoriesPrediction(
            categories=[
                dm.Category(["sports", "footbag"], 0.71),
                dm.Category(["sports", "baseketball"], 0.98),
                dm.Category(["computer", "operating system", "Linux"], 0.23),
                dm.Category(["computer", "operating system", "AIX"], 0.11),
            ],
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )

        self.categories_prediction_minimal = dm.CategoriesPrediction(
            categories=[
                dm.Category(["computer operating system", "Linux"], 0.23),
                dm.Category(["computer operating system", "AIX"], 0.11),
            ]
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.categories_prediction))
        self.assertTrue(self.validate_fields(self.categories_prediction_minimal))

    def test_from_proto_and_back(self):
        new = dm.CategoriesPrediction.from_proto(self.categories_prediction.to_proto())
        for new_cat, orig_cat in zip(
            new.categories, self.categories_prediction.categories
        ):
            self.assertEqual(new_cat.labels, orig_cat.labels)
            self.assertAlmostEqual(new_cat.score, orig_cat.score)

        new = dm.CategoriesPrediction.from_proto(
            self.categories_prediction_minimal.to_proto()
        )
        for new_cat, orig_cat in zip(
            new.categories, self.categories_prediction_minimal.categories
        ):
            self.assertEqual(new_cat.labels, orig_cat.labels)
            self.assertAlmostEqual(new_cat.score, orig_cat.score)

    def test_from_json_and_back(self):
        new = dm.CategoriesPrediction.from_json(self.categories_prediction.to_json())
        for new_cat, orig_cat in zip(
            new.categories, self.categories_prediction.categories
        ):
            self.assertEqual(new_cat.labels, orig_cat.labels)
            self.assertAlmostEqual(new_cat.score, orig_cat.score)

        new = dm.CategoriesPrediction.from_json(
            self.categories_prediction_minimal.to_json()
        )
        for new_cat, orig_cat in zip(
            new.categories, self.categories_prediction_minimal.categories
        ):
            self.assertEqual(new_cat.labels, orig_cat.labels)
            self.assertAlmostEqual(new_cat.score, orig_cat.score)

    def test_sort(self):
        shifted_categories = zip(
            self.categories_prediction.categories[1:],
            self.categories_prediction.categories[:-1],
        )

        for a, b in shifted_categories:
            self.assertLess(a.score, b.score)
