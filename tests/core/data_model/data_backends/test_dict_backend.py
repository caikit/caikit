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

"""Tests for DictBackend"""

# Standard
from typing import Iterable

# Third Party
import pytest

# Local
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.data_backends.dict_backend import DictBackend
from tests.data_model_helpers import make_proto_def, temp_data_model


def test_dict_backend_basic_message():
    """Test that the dict backend can be used for messages containing primitives"""
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": str,
                    "bar": int,
                }
            }
        )
    ) as dm:
        assert hasattr(dm, "ThingOne")
        assert issubclass(dm.ThingOne, DataBase)

        data_dict = {
            "foo": "foo-val",
            "bar": 1234,
        }

        backend = DictBackend(data_dict)
        msg = dm.ThingOne.from_backend(backend)

        assert msg.foo == data_dict["foo"]
        assert msg.bar == data_dict["bar"]


def test_dict_backend_sub_message():
    """Make sure that a nested message is handled correctly"""
    with temp_data_model(
        make_proto_def(
            {
                "SomeThing": {
                    "foo": str,
                    "bar": int,
                },
                "ThingWrapper": {
                    "the_thing": "SomeThing",
                },
            }
        )
    ) as dm:
        assert hasattr(dm, "SomeThing")
        assert hasattr(dm, "ThingWrapper")
        assert issubclass(dm.SomeThing, DataBase)
        assert issubclass(dm.ThingWrapper, DataBase)

        data_dict = {
            "the_thing": {
                "foo": "foo-val",
                "bar": 1234,
            },
        }

        backend = DictBackend(data_dict)
        msg = dm.ThingWrapper.from_backend(backend)

        sub_msg = msg.the_thing
        assert isinstance(sub_msg, DataBase)
        assert sub_msg.foo == data_dict["the_thing"]["foo"]
        assert sub_msg.bar == data_dict["the_thing"]["bar"]


def test_dict_backend_repeated_sub_message():
    """Make sure that a nested repeated message is handled correctly"""
    with temp_data_model(
        make_proto_def(
            {
                "Foo": {
                    "foo": str,
                    "bar": int,
                },
                "FooWrapper": {
                    "some_foos": ["Foo"],
                },
            }
        )
    ) as dm:
        assert hasattr(dm, "Foo")
        assert hasattr(dm, "FooWrapper")
        assert issubclass(dm.Foo, DataBase)
        assert issubclass(dm.FooWrapper, DataBase)

        data_dict = {
            "some_foos": [
                {"foo": "foo-val", "bar": 1234},
                {"foo": "baz-val", "bar": 4321},
            ],
        }

        backend = DictBackend(data_dict)
        msg = dm.FooWrapper.from_backend(backend)
        sub_msgs = msg.some_foos
        assert isinstance(sub_msgs, Iterable)
        assert len(sub_msgs) == 2
        sub_msg1, sub_msg2 = sub_msgs
        assert sub_msg1.foo == data_dict["some_foos"][0]["foo"]
        assert sub_msg1.bar == data_dict["some_foos"][0]["bar"]
        assert sub_msg2.foo == data_dict["some_foos"][1]["foo"]
        assert sub_msg2.bar == data_dict["some_foos"][1]["bar"]


def test_dict_backend_oneof():
    """Make sure that a oneof can be correctly accessed from a backend"""
    with temp_data_model(make_proto_def({"Foo": {"foo": "Union[str, int]"}})) as dm:
        assert hasattr(dm, "Foo")
        assert issubclass(dm.Foo, DataBase)

        data_dict = {"foo": "asdf"}
        backend = DictBackend(data_dict)
        msg = dm.Foo.from_backend(backend)
        assert msg.foo == "asdf"
        assert msg.which_oneof("foo") == "foo_str"


def test_dict_backend_invalid_field_error():
    """Make sure that an AttributeError is raised if an invalid field is
    requested
    """
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": str,
                    "bar": int,
                }
            }
        )
    ) as dm:
        data_dict = {
            "foo": "foo-val",
            "bar": 1234,
        }

        backend = DictBackend(data_dict)
        msg = dm.ThingOne.from_backend(backend)

        with pytest.raises(AttributeError):
            backend.get_attribute(dm.ThingOne, "baz")

        with pytest.raises(AttributeError):
            msg.baz
