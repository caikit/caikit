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

# Standard
import json

# First Party
import alog

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase

log = alog.use_channel("TEST")

## View Tests ##################################################################


class TestRulesPrediction(TestCaseBase):
    """Tests for RulesPrediction"""

    def test_construct_multiple_views(self):
        """Test that the constructor can be called with a list of views"""
        msg = dm.RulesPrediction(
            [
                dm.View("TestView1", [{"k1": 1}, {"k1": 2}]),
                dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}]),
            ]
        )
        self.assertEqual(len(msg.views), 2)

    def test_construct_empty(self):
        """Test that the constructor can be called with an empty list of views"""
        msg = dm.RulesPrediction([])
        self.assertEqual(len(msg.views), 0)

    def test_construct_empty_view(self):
        """Test that the constructor can be called with an empty view"""
        msg = dm.RulesPrediction([dm.View("EmptyView", [])])
        self.assertEqual(len(msg.views), 1)

    def test_view_names(self):
        """Test that view_names creates the correct set of names"""
        msg = dm.RulesPrediction(
            [
                dm.View("TestView1", [{"k1": 1}, {"k1": 2}]),
                dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}]),
            ]
        )
        self.assertListEqual(msg.view_names(), ["TestView1", "TestView2"])

    def test_view(self):
        """Test that the view() function correctly extracts a view by name"""
        v1 = dm.View("TestView1", [{"k1": 1}, {"k1": 2}])
        v2 = dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}])
        msg = dm.RulesPrediction([v1, v2])
        self.assertEqual(msg.view("TestView1"), v1)
        self.assertEqual(msg.view("TestView2"), v2)
        with self.assertRaises(ValueError):
            msg.view("Not There")

    def test_filter_view(self):
        """Test that filter_view filters accurately on an expression"""
        v1 = dm.View("TestView1", [])
        v2 = dm.View("TestView2", [])
        msg = dm.RulesPrediction([v1, v2])
        self.assertListEqual(
            [v.name for v in msg.filter_view("TestView1")], ["TestView1"]
        )
        self.assertListEqual(
            [v.name for v in msg.filter_view("TestView*")], ["TestView1", "TestView2"]
        )
        self.assertListEqual(
            [v.name for v in msg.filter_view("TestView[234]")], ["TestView2"]
        )
        self.assertListEqual([v.name for v in msg.filter_view("Foobar")], [])

    def test_to_dict(self):
        """Test that to_dict correctly serializes all the way down to the view
        property values.
        """
        v1 = dm.View(
            "TestView1",
            [{"k1": 1, "k2": [dm.Span(1, 2)]}, {"k1": 2, "k2": [dm.Span(3, 4)]}],
        )
        v2 = dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}])
        msg = dm.RulesPrediction([v1, v2])
        self.assertDictEqual(
            msg.to_dict(),
            {
                "producer_id": None,
                "views": [
                    {
                        "name": "TestView1",
                        "properties": [
                            {
                                "aql_property": {
                                    "k1": 1,
                                    "k2": [{"begin": 1, "end": 2, "text": ""}],
                                }
                            },
                            {
                                "aql_property": {
                                    "k1": 2,
                                    "k2": [{"begin": 3, "end": 4, "text": ""}],
                                }
                            },
                        ],
                    },
                    {
                        "name": "TestView2",
                        "properties": [
                            {"aql_property": {"a": "asdf", "b": 2}},
                            {"aql_property": {"a": "qwer", "b": 99}},
                        ],
                    },
                ],
            },
        )

    def test_to_json(self):
        """Test that to_json correctly serializes all the way down to the view
        property values.
        """
        v1 = dm.View(
            "TestView1",
            [{"k1": 1, "k2": [dm.Span(1, 2)]}, {"k1": 2, "k2": [dm.Span(3, 4)]}],
        )
        v2 = dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}])
        msg = dm.RulesPrediction([v1, v2])
        got_js = msg.to_json()
        exp_js = json.dumps(
            {
                "producer_id": None,
                "views": [
                    {
                        "name": "TestView1",
                        "properties": [
                            {
                                "aql_property": {
                                    "k1": 1,
                                    "k2": [{"begin": 1, "end": 2, "text": ""}],
                                }
                            },
                            {
                                "aql_property": {
                                    "k1": 2,
                                    "k2": [{"begin": 3, "end": 4, "text": ""}],
                                }
                            },
                        ],
                    },
                    {
                        "name": "TestView2",
                        "properties": [
                            {"aql_property": {"a": "asdf", "b": 2}},
                            {"aql_property": {"a": "qwer", "b": 99}},
                        ],
                    },
                ],
            }
        )
        log.debug("Got JSON: %s", got_js)
        log.debug("Expected JSON: %s", exp_js)
        self.assertDictEqual(json.loads(got_js), json.loads(exp_js))

    def test_proto_round_trip(self):
        v1 = dm.View(
            "TestView1",
            [{"k1": 1, "k2": [dm.Span(1, 2)]}, {"k1": 2, "k2": [dm.Span(3, 4)]}],
        )
        v2 = dm.View("TestView2", [{"a": "asdf", "b": 2}, {"a": "qwer", "b": 99}])
        msg = dm.RulesPrediction([v1, v2])
        proto = msg.to_proto()
        msg_round_trip = dm.RulesPrediction.from_proto(proto)
        self.assertDictEqual(msg.to_dict(), msg_round_trip.to_dict())

    def test_proto_round_trip_span(self):
        v1 = dm.View(
            "TestView1", [{"a": "asdf", "b": ["ty"]}, {"a": "qwer", "b": ["gh"]}]
        )

        v2 = dm.View(
            "TestView2",
            [{"k1": 1, "k2": [1, 2]}, {"k1": 2, "k2": [3, 4]}],
        )

        v3 = dm.View(
            "TestView3",
            [
                {"v1": 2.2, "v2": [3.14, 9.2, 132.1]},
                {"v1": 3.3, "v2": [4.1, 0.99, 1.87]},
            ],
        )

        v4 = dm.View(
            "TestView4",
            [{"k1": True, "k2": [False, True]}, {"k1": False, "k2": [True, False]}],
        )
        v5 = dm.View(
            "TestView5",
            [
                {"k1": dm.Span(1, 2), "k2": [dm.Span(3, 4), dm.Span(5, 6)]},
                {"k1": dm.Span(7, 8), "k2": [dm.Span(9, 10), dm.Span(11, 12)]},
            ],
        )
        msg = dm.RulesPrediction([v1, v2])
        proto = msg.to_proto()
        msg_round_trip = dm.RulesPrediction.from_proto(proto)
        self.assertDictEqual(msg.to_dict(), msg_round_trip.to_dict())

        msg2 = dm.RulesPrediction([v3])
        proto2 = msg2.to_proto()
        msg_round_trip2 = dm.RulesPrediction.from_proto(proto2)
        aql_property = (
            msg2.to_dict().get("views")[0].get("properties")[0].get("aql_property")
        )
        aql_property_round_trip = (
            msg_round_trip2.to_dict()
            .get("views")[0]
            .get("properties")[0]
            .get("aql_property")
        )
        self.assertAlmostEqual(
            aql_property.get("v1"), aql_property_round_trip.get("v1"), places=4
        )
        self.assertAlmostEqual(
            aql_property.get("v2")[0], aql_property_round_trip.get("v2")[0], places=4
        )
        self.assertEqual(aql_property.keys(), aql_property_round_trip.keys())

        msg3 = dm.RulesPrediction([v4, v5])
        proto3 = msg3.to_proto()
        msg_round_trip3 = dm.RulesPrediction.from_proto(proto3)
        self.assertDictEqual(msg3.to_dict(), msg_round_trip3.to_dict())


## View Tests ##################################################################


class TestView(TestCaseBase):
    """Tests for View"""

    ## Happy Path Tests ##

    def test_construct_list_dicts(self):
        """Test that the constructor works with a list of dicts"""
        d1 = {"k1": 1, "k2": "asdf"}
        d2 = {"k1": 2, "k2": "qwer"}
        msg = dm.View("TestView", [d1, d2])
        self.assertEqual(len(msg.properties), 2)
        self.assertTrue(all(isinstance(v, dm.ViewProperty) for v in msg.properties))
        self.assertDictEqual(msg.properties[0].to_dict(), {"aql_property": d1})
        self.assertDictEqual(msg.properties[1].to_dict(), {"aql_property": d2})

    def test_construct_list_views(self):
        """Test that the constructor works with a list of dm.ViewProperty msgs"""
        d1 = {"k1": 1, "k2": "asdf"}
        d2 = {"k1": 2, "k2": "qwer"}
        msg = dm.View("TestView", [dm.ViewProperty(d1), dm.ViewProperty(d2)])
        self.assertEqual(len(msg.properties), 2)
        self.assertTrue(all(isinstance(v, dm.ViewProperty) for v in msg.properties))
        self.assertDictEqual(msg.properties[0].to_dict(), {"aql_property": d1})
        self.assertDictEqual(msg.properties[1].to_dict(), {"aql_property": d2})

    def test_construct_empty(self):
        """Test that the constructor works with an empty list of properties"""
        msg = dm.View("TestView", [])
        self.assertEqual(len(msg.properties), 0)

    def test_construct_empty_property(self):
        """Test that the constructor works with an empty dict property"""
        msg = dm.View("TestView", [{}])
        self.assertEqual(len(msg.properties), 1)

    def test_to_dict(self):
        """Test that to_dict serializes to a list of dicts"""
        d1 = {"k1": 1, "k2": "asdf"}
        d2 = {"k1": 2, "k2": "qwer"}
        msg = dm.View("TestView", [d1, d2])
        self.assertDictEqual(
            msg.to_dict(),
            {
                "name": "TestView",
                "properties": [
                    {"aql_property": d1},
                    {"aql_property": d2},
                ],
            },
        )

    def test_to_json(self):
        """Test that to_json serializes to the json-representation of a list of
        dicts
        """
        d1 = {"k1": 1, "k2": "asdf"}
        d2 = {"k1": 2, "k2": "qwer"}
        msg = dm.View("TestView", [d1, d2])
        self.assertEqual(
            msg.to_json(),
            json.dumps(
                {
                    "name": "TestView",
                    "properties": [
                        {"aql_property": d1},
                        {"aql_property": d2},
                    ],
                }
            ),
        )

    def test_proto_round_trip(self):
        """Test that to_proto / from_proto successfully round-trip"""
        d1 = {"k1": 1, "k2": "asdf"}
        d2 = {"k1": 2, "k2": "qwer"}
        msg = dm.View("TestView", [d1, d2])
        proto = msg.to_proto()
        msg_round_trip = dm.View.from_proto(proto)
        self.assertDictEqual(msg.to_dict(), msg_round_trip.to_dict())

    def test_property_names(self):
        """Test that the list of property names is produced correctly"""
        d1 = {"k1": 1, "k2": "asdf", "k3": [dm.Span(1, 2)]}
        d2 = {"k1": 2, "k2": "qwer", "k3": [dm.Span(3, 4)]}
        msg = dm.View("TestView", [d1, d2])
        self.assertListEqual(msg.property_names(), ["k1", "k2", "k3"])

    def test_property_names_empty(self):
        """Test that the list of property names doesn't fail for an empty view"""
        msg = dm.View("TestView", [])
        self.assertListEqual(msg.property_names(), [])

    def test_property(self):
        """Test that the list of property values is produced correctly"""
        d1 = {"k1": 1, "k2": "asdf", "k3": [dm.Span(1, 2)]}
        d2 = {"k1": 2, "k2": "qwer", "k3": [dm.Span(3, 4)]}
        msg = dm.View("TestView", [d1, d2])
        self.assertListEqual(msg.property("k1"), [1, 2])
        self.assertListEqual(msg.property("k2"), ["asdf", "qwer"])
        self.assertListEqual(msg.property("k3"), [[dm.Span(1, 2)], [dm.Span(3, 4)]])
        with self.assertRaises(ValueError):
            msg.property("k4")

    ## Error Caser Tests ##

    def construct_non_str_name(self):
        """Test that a non-str name is rejected"""
        with self.assertRaises(ValueError):
            dm.View(1, [])

    def construct_non_list_properties(self):
        """Test that a non-list properties argument is rejected"""
        with self.assertRaises(ValueError):
            dm.View("TestView", {"k1": 1, "k2": 2})

    def construct_bad_property_type(self):
        """Test that constructing with a property that is not a dict or a
        ViewProperty is rejected
        """
        with self.assertRaises(ValueError):
            dm.View("TestView", [1])

    def construct_mixed_bad_property_types(self):
        """Test that constructing with a property that is not a dict and one
        that is gets rejected
        """
        with self.assertRaises(ValueError):
            dm.View("TestView", [1, {"k1": 1}])


## ViewProperty Tests ##########################################################


class TestViewProperty(TestCaseBase):
    """Tests for ViewProperty"""

    ## Happy Path Tests ##

    def test_construct(self):
        """Test that the constructor works with a dict. Note that we leave all
        of the supported value-type testing to the ViewPropertyValue tests.
        """
        dict_val = {"k1": 1, "k2": 2}
        msg = dm.ViewProperty(dict_val)
        self.assertTrue(isinstance(msg.aql_property["k1"], dm.ViewPropertyValue))
        self.assertEqual(msg.aql_property["k1"].int_val, 1)
        self.assertTrue(isinstance(msg.aql_property["k2"], dm.ViewPropertyValue))
        self.assertEqual(msg.aql_property["k2"].int_val, 2)

    def test_construct_mixed_value_types(self):
        """Test that constructing with a dict that has different types for its
        values is ok
        """
        dm.ViewProperty({"k1": 1, "k2": "two"})

    def test_construct_empty(self):
        """Test that the constructor works with a dict"""
        msg = dm.ViewProperty({})
        self.assertEqual(len(msg.aql_property), 0)

    def test_to_dict(self):
        """Test that to_dict serializes to a list of dicts"""
        dict_val = {"k1": 1, "k2": 2}
        msg = dm.ViewProperty(dict_val)
        dict_round_trip = msg.to_dict()
        self.assertTrue("aql_property" in dict_round_trip)
        self.assertDictEqual(dict_round_trip["aql_property"], dict_val)

    def test_to_json(self):
        """Test that to_json serializes to the json-representation of a list of
        dicts
        """
        dict_val = {"k1": 1, "k2": 2}
        msg = dm.ViewProperty(dict_val)
        js_val = msg.to_json()
        self.assertEqual(json.dumps({"aql_property": dict_val}), js_val)

    def test_proto_round_trip(self):
        """Test that to_proto / from_proto successfully round-trip"""
        msg = dm.ViewProperty({"k1": 1, "k2": 2})
        proto = msg.to_proto()
        msg_round_trip = dm.ViewProperty.from_proto(proto)
        self.assertDictEqual(msg.to_dict(), msg_round_trip.to_dict())

    def test_proto_round_trip_list_spans(self):
        """Test that to_proto / from_proto successfully round-trip for a list of
        Spans (the hardest one!)
        """
        log.debug("1. Creating message")
        msg = dm.ViewProperty({"k1": [dm.Span(1, 2)]})
        self.assertTrue("k1" in msg.aql_property)
        self.assertIsNotNone(msg.aql_property["k1"].list_span_val)
        self.assertEqual(len(msg.aql_property["k1"].list_span_val.val), 1)
        log.debug("2. Calling to_proto")
        proto = msg.to_proto()
        self.assertTrue("k1" in proto.aql_property)
        self.assertIsNotNone(proto.aql_property["k1"].list_span_val)
        self.assertEqual(len(proto.aql_property["k1"].list_span_val.val), 1)
        log.debug("3. Calling from_proto")
        msg_round_trip = dm.ViewProperty.from_proto(proto)
        self.assertTrue("k1" in msg_round_trip.aql_property)
        self.assertIsNotNone(msg_round_trip.aql_property["k1"].list_span_val)
        self.assertEqual(len(msg_round_trip.aql_property["k1"].list_span_val.val), 1)
        self.assertDictEqual(msg.to_dict(), msg_round_trip.to_dict())

    def test_schema_matches(self):
        """Make sure matching schemas are correctly identified and different
        schemas are correctly identified.
        """
        msg1 = dm.ViewProperty({"k1": [dm.Span(1, 2)], "k2": 1})
        msg2 = dm.ViewProperty({"k1": [dm.Span(3, 4)], "k2": 2})
        msg3 = dm.ViewProperty({"key1": [dm.Span(3, 4)], "key2": 2})
        msg4 = dm.ViewProperty({"k1": dm.Span(3, 4), "k2": 2})
        msg5 = dm.ViewProperty({"k1": [1], "k2": 1})

        log.debug("Comparing msg1 <-> msg2")
        self.assertTrue(msg1.schema_matches(msg2))
        log.debug("Comparing msg1 <-> msg3")
        self.assertFalse(msg1.schema_matches(msg3))
        log.debug("Comparing msg1 <-> msg4")
        self.assertFalse(msg1.schema_matches(msg4))
        log.debug("Comparing msg1 <-> msg5")
        self.assertFalse(msg1.schema_matches(msg5))

    ## Error Caser Tests ##

    def test_construct_non_string_key_throws(self):
        """Test that constructing with a dict that has non-string keys throws"""
        with self.assertRaises(TypeError):
            dm.ViewProperty({1: "one"})

    def test_construct_non_aql_type_values_throws(self):
        """Test that constructing with a dict that has values that aren't valid AQL types throws"""
        with self.assertRaises(TypeError):
            dm.ViewProperty({"k": {"nested": 1}})


## Oneof ViewPropertyValue Tests ###############################################


class TestViewPropertyValue(TestCaseBase):
    """Tests for ViewPropertyValue"""

    # All of the possible oneof values
    _ONEOF_FIELDS = [
        "str_val",
        "float_val",
        "int_val",
        "bool_val",
        "span_val",
        "list_str_val",
        "list_float_val",
        "list_int_val",
        "list_bool_val",
        "list_span_val",
    ]

    def assertListAlmostEqual(self, A, B):
        self.assertEqual(len(A), len(B))
        for a, b in zip(A, B):
            self.assertAlmostEqual(a, b, places=6)

    def _test_type(self, val, val_field, assert_equal_fn=None):
        """Shared test for all types"""
        assert_equal_fn = assert_equal_fn or self.assertEqual

        # Construct the message
        with self.subTest("Message Constructor"):
            msg = dm.ViewPropertyValue(val)

        # Make sure the expected value property is present and not None
        with self.subTest("Stored Value Properties"):
            if val is not None:
                stored_val = getattr(msg, val_field)
                self.assertIsNotNone(stored_val)

            # Make sure the stored _val is equivalent to val
            self.assertEqual(val, msg.value)

            # Make sure all other possible fields are None
            for field in self._ONEOF_FIELDS:
                if field != val_field:
                    self.assertIsNone(getattr(msg, field))

        # Make sure to_dict serialization works as expected:
        #   * Sub-message (Span): Serializes to the to_dict of the message type
        #   * Primitive: Serialize to the primitive itself
        #   * List: Serialize to a list of the corresponding primitive's
        #       serialized version
        with self.subTest("Serialize with to_dict"):
            if hasattr(val, "to_dict"):
                assert_equal_fn(msg.to_dict(), val.to_dict())
            elif isinstance(val, list):
                self.assertListEqual(
                    msg.to_dict(),
                    [v.to_dict() if hasattr(v, "to_dict") else v for v in val],
                )
            else:
                assert_equal_fn(msg.to_dict(), val)

        # Make sure to_json serializes to the json representation of val:
        #   * Sub-message (Span): The json-serialized message
        #   * Primitive: The json representation of the primitive
        #   * List: The json representation of a list of the serialized element
        #       type
        with self.subTest("Serialize with to_json"):
            if hasattr(val, "to_json"):  # Raw Span
                js_val = val.to_json()
            elif isinstance(val, list):  # Lists of valid values
                js_val = json.dumps(
                    [v.to_dict() if hasattr(v, "to_dict") else v for v in val]
                )
            else:  # Primitives
                js_val = json.dumps(val)
            self.assertEqual(msg.to_json(), js_val)

        # Make sure to_proto/from_proto round trips
        with self.subTest("Round trip with to_proto / from_proto"):
            proto = msg.to_proto()
            round_trip_msg = dm.ViewPropertyValue.from_proto(proto)
            assert_equal_fn(val, round_trip_msg.value)

    ## Happy Path Tests ##

    def test_str(self):
        self._test_type("asdf", "str_val")

    def test_float(self):
        self._test_type(1.234, "float_val", self.assertAlmostEqual)

    def test_int(self):
        self._test_type(42, "int_val")

    def test_bool(self):
        self._test_type(True, "bool_val")

    def test_span(self):
        self._test_type(dm.Span(1, 2), "span_val")

    def test_list_str(self):
        self._test_type(["asdf", "qwer"], "list_str_val")

    def test_list_float(self):
        self._test_type([1.234, 3.14], "list_float_val", self.assertListAlmostEqual)

    def test_list_int(self):
        self._test_type([42, 123], "list_int_val")

    def test_list_bool(self):
        self._test_type([True, False], "list_bool_val")

    def test_list_span(self):
        self._test_type([dm.Span(1, 2), dm.Span(3, 4)], "list_span_val")

    def test_None(self):
        self._test_type(None, "")

    ## Error Caser Tests ##

    def test_invalid_type_dict(self):
        with self.assertRaises(TypeError):
            msg = dm.ViewPropertyValue({"key": 1})

    def test_invalid_type_list_invalid(self):
        with self.assertRaises(TypeError):
            msg = dm.ViewPropertyValue([{"key": 1}])

    def test_invalid_type_list_mixed(self):
        with self.assertRaises(TypeError):
            msg = dm.ViewPropertyValue([1, {"key": 1}])

    def test_from_proto_invalid_type(self):
        with self.assertRaises(TypeError):
            dm.ViewPropertyValue.from_proto(1)


## Repeated Value Type Wrappers ################################################
#
# The most complex part of this schema to test is the value-types that are lists
# of homogonous AQL primitive types. To avoid the need for massive
# copy-and-paste tests, we use a base class that implements a standard set of
# test functions. These formulaic tests work for *most* of the list types, but
# individual list types (notably [float] and [dm.Span]) require overrides for
# certain parts to allow for more complex comparisons. The result of this
# (below) is a set of TestPropertyListValue<TYPE> classes that inherit from
# PropertyListValueTestBase and each contain a for each of the individual test
# functions in the base class. Most are just one-line calls to the parent's
# implementation with a few overrides where needed.
################################################################################


class PropertyListValueTestBase(TestCaseBase):
    """Shared baseclass for all tests of the repeated value types to avoid
    copy-and-paste for most of these tests.
    """

    def do_test_construct_good(self):
        """Test that it constructs with a list of the correct type"""
        self._MSG_TYPE([self._TYPE_SAMPLE])

    def do_test_construct_empty(self):
        """Test that it constructs with an empty list"""
        self._MSG_TYPE([])

    def do_test_construct_bad_type(self):
        """Test that it throws with a non-list"""
        with self.assertRaises(TypeError):
            self._MSG_TYPE(self._TYPE_SAMPLE)

    def do_test_construct_bad_entry_type(self):
        """Test that it throws with a list of the wrong type. Since lists of
        dicts are not supported, a dict is an invalid type for any of the child
        classes.
        """
        with self.assertRaises(TypeError):
            self._MSG_TYPE([{"k": self._TYPE_SAMPLE}])

    def do_test_construct_mixed_entry_type(self):
        """Test that it throws with a list of some elements of the correct type
        and some of the wrong type. Since lists of dicts are not supported, a
        dict is an invalid type for any of the child classes.
        """
        with self.assertRaises(TypeError):
            self._MSG_TYPE([{"k": self._TYPE_SAMPLE}, self._TYPE_SAMPLE])

    def do_test_to_dict(self):
        """Test that to_dict correctly serializes the list"""
        msg = self._MSG_TYPE([self._TYPE_SAMPLE])
        raw_val = msg.val
        dict_val = msg.to_dict()
        self.assertTrue("val" in dict_val)
        self.assertTrue(
            all(a == b for a, b in zip(raw_val, dict_val["val"])),
            "{} != {}".format(str(raw_val), str(dict_val)),
        )

    def do_test_to_json(self):
        """Test that to_json correctly serializes the list"""
        msg = self._MSG_TYPE([self._TYPE_SAMPLE])
        raw_val = msg.val
        json_val = msg.to_json()
        dict_val = json.loads(json_val)
        self.assertIn("val", dict_val)
        self.assertTrue(
            all(a == b for a, b in zip(raw_val, dict_val["val"])),
            "{} != {}".format(str(raw_val), str(dict_val)),
        )

    def do_test_proto_round_trip(self):
        """Test that to_proto/from_proto correctly serializes and deserializes
        the list
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE, self._TYPE_SAMPLE])
        self.assertDictEqual(
            self._MSG_TYPE.from_proto(msg.to_proto()).to_dict(), msg.to_dict()
        )

    def do_test_json_round_trip(self):
        """Test that to_proto/from_proto correctly serializes and deserializes
        the list
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE, self._TYPE_SAMPLE])
        self.assertDictEqual(
            self._MSG_TYPE.from_json(msg.to_json()).to_dict(), msg.to_dict()
        )


class TestPropertyListValueStr(PropertyListValueTestBase):
    """Tests for PropertyListValueStr"""

    _MSG_TYPE = dm.PropertyListValueStr
    _TYPE_SAMPLE = "asdf"

    def test_construct_good(self):
        self.do_test_construct_good()

    def test_construct_empty(self):
        self.do_test_construct_empty()

    def test_construct_bad_type(self):
        self.do_test_construct_bad_type()

    def test_construct_bad_entry_type(self):
        self.do_test_construct_bad_entry_type()

    def test_construct_mixed_entry_type(self):
        self.do_test_construct_mixed_entry_type()

    def test_to_dict(self):
        self.do_test_to_dict()

    def test_to_json(self):
        self.do_test_to_json()

    def test_proto_round_trip(self):
        self.do_test_proto_round_trip()


class TestPropertyListValueFloat(PropertyListValueTestBase):
    """Tests for PropertyListValueFloat"""

    _MSG_TYPE = dm.PropertyListValueFloat
    _TYPE_SAMPLE = 1.234

    def test_construct_good(self):
        self.do_test_construct_good()

    def test_construct_empty(self):
        self.do_test_construct_empty()

    def test_construct_bad_type(self):
        self.do_test_construct_bad_type()

    def test_construct_bad_entry_type(self):
        self.do_test_construct_bad_entry_type()

    def test_construct_mixed_entry_type(self):
        self.do_test_construct_mixed_entry_type()

    def test_to_dict(self):
        self.do_test_to_dict()

    def test_to_json(self):
        self.do_test_to_json()

    def test_proto_round_trip(self):
        """Test that to_proto/from_proto correctly serializes and deserializes
        the list (float serialization/deserialization is not floating-point
        equivalent, so no base class)
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE, self._TYPE_SAMPLE])
        orig_dict = msg.to_dict()
        round_trip_dict = self._MSG_TYPE.from_proto(msg.to_proto()).to_dict()
        self.assertListEqual(list(orig_dict.keys()), list(round_trip_dict.keys()))
        for a, b in zip(orig_dict["val"], round_trip_dict["val"]):
            self.assertAlmostEqual(a, b)

    def test_json_round_trip(self):
        """Test that to_proto/from_proto correctly serializes and deserializes
        the list (float serialization/deserialization is not floating-point
        equivalent, so no base class)
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE, self._TYPE_SAMPLE])
        orig_dict = msg.to_dict()
        round_trip_dict = self._MSG_TYPE.from_json(msg.to_json()).to_dict()
        self.assertListEqual(list(orig_dict.keys()), list(round_trip_dict.keys()))
        for a, b in zip(orig_dict["val"], round_trip_dict["val"]):
            self.assertAlmostEqual(a, b)


class TestPropertyListValueInt(PropertyListValueTestBase):
    """Tests for PropertyListValueInt"""

    _MSG_TYPE = dm.PropertyListValueInt
    _TYPE_SAMPLE = 1

    def test_construct_good(self):
        self.do_test_construct_good()

    def test_construct_empty(self):
        self.do_test_construct_empty()

    def test_construct_bad_type(self):
        self.do_test_construct_bad_type()

    def test_construct_bad_entry_type(self):
        self.do_test_construct_bad_entry_type()

    def test_construct_mixed_entry_type(self):
        self.do_test_construct_mixed_entry_type()

    def test_to_dict(self):
        self.do_test_to_dict()

    def test_to_json(self):
        self.do_test_to_json()

    def test_proto_round_trip(self):
        self.do_test_proto_round_trip()

    def test_json_round_trip(self):
        self.do_test_json_round_trip()


class TestPropertyListValueBool(PropertyListValueTestBase):
    """Tests for PropertyListValueBool"""

    _MSG_TYPE = dm.PropertyListValueBool
    _TYPE_SAMPLE = True

    def test_construct_good(self):
        self.do_test_construct_good()

    def test_construct_empty(self):
        self.do_test_construct_empty()

    def test_construct_bad_type(self):
        self.do_test_construct_bad_type()

    def test_construct_bad_entry_type(self):
        self.do_test_construct_bad_entry_type()

    def test_construct_mixed_entry_type(self):
        self.do_test_construct_mixed_entry_type()

    def test_to_dict(self):
        self.do_test_to_dict()

    def test_to_json(self):
        self.do_test_to_json()

    def test_proto_round_trip(self):
        self.do_test_proto_round_trip()

    def test_json_round_trip(self):
        self.do_test_json_round_trip()


class TestPropertyListValueSpan(PropertyListValueTestBase):
    """Tests for PropertyListValueSpan"""

    _MSG_TYPE = dm.PropertyListValueSpan
    _TYPE_SAMPLE = dm.Span(1, 2)

    def test_construct_good(self):
        self.do_test_construct_good()

    def test_construct_empty(self):
        self.do_test_construct_empty()

    def test_construct_bad_type(self):
        self.do_test_construct_bad_type()

    def test_construct_bad_entry_type(self):
        self.do_test_construct_bad_entry_type()

    def test_construct_mixed_entry_type(self):
        self.do_test_construct_mixed_entry_type()

    def test_to_dict(self):
        self.do_test_to_dict()

    def test_to_dict(self):
        """Test that to_dict correctly serializes the list (needs to_dict on
        nested values, so no base-class)
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE])
        raw_val = [v.to_dict() for v in msg.val]
        dict_val = msg.to_dict()
        self.assertTrue("val" in dict_val)
        self.assertTrue(
            all(a == b for a, b in zip(raw_val, dict_val["val"])),
            "{} != {}".format(str(raw_val), str(dict_val)),
        )

    def test_to_json(self):
        """Test that to_dict correctly serializes the list (needs to_dict on
        nested values, so no base-class)
        """
        msg = self._MSG_TYPE([self._TYPE_SAMPLE])
        raw_val = [v.to_dict() for v in msg.val]
        json_val = msg.to_json()
        dict_val = json.loads(json_val)
        self.assertTrue("val" in dict_val)
        self.assertTrue(
            all(a == b for a, b in zip(raw_val, dict_val["val"])),
            "{} != {}".format(str(raw_val), str(dict_val)),
        )
