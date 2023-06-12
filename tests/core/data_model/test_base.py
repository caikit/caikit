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

"""Tests for the functionality in the base class for data model objects"""

# Standard
import importlib
import os

# Third Party
import pytest

# Local
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.data_backends.base import DataModelBackendBase
from caikit.core.data_model.dataobject import DataObjectBase
from tests.data_model_helpers import (
    justify_script_string,
    make_proto_def,
    temp_data_model,
    temp_module,
)

## Helpers #####################################################################


class AccessCounterBackend(DataModelBackendBase):
    """Simple backend that counts how many times a given attribute is accessed
    and can be configured to cache or not.
    """

    def __init__(self, data, do_cache):
        self.data = data
        self.do_cache = do_cache
        self.access_counter = {}

    def cache_attribute(self, name, value):
        return self.do_cache

    # pylint: disable=unused-argument
    def get_attribute(self, data_model_class, name):
        self.access_counter[name] = self.access_counter.get(name, 0) + 1
        return self.data[name]

    def access_count(self, name):
        return self.access_counter.get(name, 0)


## Tests #######################################################################

##########################
## Proto Class Matching ##
##########################


def test_derived_class_no_proto_class():
    """Test that an appropriate error is raised when the protobufs module for a
    derived class does not have a corresponding protobufs class
    """
    with temp_module() as (mod_name, mod_dir):
        # Add a "protobufs.py" file to the library that has nothing in it
        with open(
            os.path.join(mod_dir, "protobufs.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write(justify_script_string("""# Nothing here"""))

        # Add a derived data model class file
        with open(os.path.join(mod_dir, "object.py"), "w", encoding="utf-8") as handle:
            handle.write(
                justify_script_string(
                    """
                    from caikit.core.data_model import base
                    class Object(base.DataBase):
                        def __init__(self, foo):
                            self.foo = foo
                    """
                )
            )

        # Make sure an ValueError is raised when the import is tried
        with pytest.raises(ValueError):
            importlib.import_module(".".join([mod_name, "object"]))


def test_derived_class_no_proto_mod():
    """Test that an appropriate error is raised when the derived library does
    not have a protobufs directory
    """
    with temp_module() as (mod_name, mod_dir):
        # Add a derived data model class file
        with open(os.path.join(mod_dir, "object.py"), "w", encoding="utf-8") as handle:
            handle.write(
                justify_script_string(
                    """
                    from caikit.core.data_model import base
                    class Object(base.DataBase):
                        def __init__(self, foo):
                            self.foo = foo
                    """
                )
            )

        # Make sure an ValueError is raised when the import is tried
        with pytest.raises(ValueError):
            importlib.import_module(".".join([mod_name, "object"]))


def test_derived_class_no_import_side_effects():
    """Test that a library which does not use the import side effects can still
    safely create a derived data model class
    """
    with temp_module() as (mod_name, mod_dir):
        # Add a "protobufs.py" file to the library that has nothing in it
        with open(
            os.path.join(mod_dir, "protobufs.py"), "w", encoding="utf-8"
        ) as handle:
            handle.write(
                justify_script_string(
                    """
                    class FakeDescriptor:
                        fields = []
                        fields_by_name = {}
                        oneofs_by_name = {}
                        oneofs = []
                        name = "Baz"
                        full_name = "foo.bar.Baz"

                    class Object:
                        DESCRIPTOR = FakeDescriptor
                    """
                )
            )

        # Add a derived data model class file
        with open(os.path.join(mod_dir, "object.py"), "w", encoding="utf-8") as handle:
            handle.write(
                justify_script_string(
                    """
                    from . import protobufs
                    from caikit.core.data_model import base
                    class Object(base.DataBase):
                        _proto_class = protobufs.Object
                        def __init__(self, foo):
                            self.foo = foo
                    """
                )
            )

        # Make sure the import succeeds and that the data model class has the
        # right inheritance
        lib = importlib.import_module(".".join([mod_name, "object"]))
        assert hasattr(lib, "Object")
        assert issubclass(lib.Object, DataBase)


def test_compiled_proto_init():
    """Make sure that support for 'compiled' protos works cleanly without using
    the dataobject wrapper
    """
    # pylint: disable=duplicate-code
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": str,
                    "bar": int,
                }
            },
            mock_compiled=True,
        )
    ) as dm:
        assert isinstance(dm.ThingOne, type)
        assert issubclass(dm.ThingOne, DataBase)
        assert not issubclass(dm.ThingOne, DataObjectBase)
        assert dm.ThingOne.fields == ("foo", "bar")

        # Test construction with positional args
        inst = dm.ThingOne("foo", 1)
        assert inst.foo == "foo"
        assert inst.bar == 1

        # Test construction with keyword args
        inst = dm.ThingOne(foo="foo", bar=1)
        assert inst.foo == "foo"
        assert inst.bar == 1

        # Test defaulting to None
        inst = dm.ThingOne()
        assert inst.foo is None
        assert inst.bar is None

        # Test error cases for construction
        with pytest.raises(TypeError):
            dm.ThingOne("foo", foo="bar")
        with pytest.raises(TypeError):
            dm.ThingOne(widget="qewr")
        with pytest.raises(TypeError):
            dm.ThingOne("foo", 1, 2)


def test_compiled_proto_oneof():
    """Make sure that support for 'compiled' protos works cleanly without using
    the dataobject wrapper
    """
    # pylint: disable=duplicate-code
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": "Union[str, int]",
                }
            },
            mock_compiled=True,
        )
    ) as dm:
        assert isinstance(dm.ThingOne, type)
        assert issubclass(dm.ThingOne, DataBase)
        assert not issubclass(dm.ThingOne, DataObjectBase)
        assert set(dm.ThingOne.fields) == {"foo_str", "foo_int"}

        # Construct with the oneof name
        inst = dm.ThingOne(foo=1)
        assert inst.foo == 1
        assert inst.which_oneof("foo") == "foo_int"

        # Construct with field name
        inst = dm.ThingOne(foo_str="asdf")
        assert inst.foo == "asdf"
        assert inst.which_oneof("foo") == "foo_str"

        # Conflicting args
        with pytest.raises(TypeError):
            dm.ThingOne(foo=1, fooster="asdf")


##################
## Data Backend ##
##################


def test_cached_backend():
    """Make sure that a data model object will cache the result of get_attribute
    if configured to do so by the backend
    """
    # pylint: disable=duplicate-code
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
        data = {"foo": "one", "bar": 2}
        backend = AccessCounterBackend(data, True)
        msg = dm.ThingOne.from_backend(backend)

        # Access the attrs to increment the counters
        msg.foo
        msg.foo
        msg.bar
        msg.bar
        msg.bar
        assert backend.access_count("foo") == 1
        assert backend.access_count("bar") == 1


def test_uncached_backend():
    """Make sure that a data backend can skip the step of caching the data on
    the data model object
    """
    # pylint: disable=duplicate-code
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
        data = {"foo": "one", "bar": 2}
        backend = AccessCounterBackend(data, False)
        msg = dm.ThingOne.from_backend(backend)

        # Access the attrs to increment the counters
        msg.foo
        msg.foo
        msg.bar
        msg.bar
        msg.bar
        assert backend.access_count("foo") == 2
        assert backend.access_count("bar") == 3


def test_invalid_attribute_no_backend():
    """Make sure that when created without a backend and without proper
    initialization, an AttributeError is raised
    """
    # pylint: disable=duplicate-code
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
        msg = dm.ThingOne.__new__(dm.ThingOne)
        with pytest.raises(AttributeError):
            msg.foo


##################
## Default Init ##
##################


def test_default_init_valid_args():
    """Make sure that valid ways of using the default __init__ all work as
    expected
    """
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": int,
                    "bar": int,
                }
            }
        )
    ) as dm:
        # Both positional
        msg = dm.ThingOne(1, 2)
        assert msg.foo == 1
        assert msg.bar == 2

        # Both kwarg
        msg = dm.ThingOne(foo=1, bar=2)
        assert msg.foo == 1
        assert msg.bar == 2

        # Mix args and kwargs
        msg = dm.ThingOne(1, bar=2)
        assert msg.foo == 1
        assert msg.bar == 2

        # Single arg, other unset
        msg = dm.ThingOne(1)
        assert msg.foo == 1
        assert msg.bar is None

        # Single kwarg, other unset
        msg = dm.ThingOne(foo=1)
        assert msg.foo == 1
        assert msg.bar is None


def test_default_init_invalid_args():
    """Make sure that a TypeError is raised if the default __init__ is given
    various combinations of incorrect arguments
    """
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": int,
                    "bar": int,
                }
            }
        )
    ) as dm:
        # Too many arguments
        with pytest.raises(TypeError):
            dm.ThingOne(1, 2, 3, 4)

        # Too many args + kwargs
        with pytest.raises(TypeError):
            dm.ThingOne(1, 2, baz=3)

        # Bad kwarg name
        with pytest.raises(TypeError):
            dm.ThingOne(1, baz=3)

        # Multiple values
        with pytest.raises(TypeError):
            dm.ThingOne(1, foo=3)


############################
## get_field_message_type ##
############################


def test_get_field_message_type_valid_fields():
    """Make sure that for valid fields, get_field_message_type returns the sub-
    field type on sub-messages and None for other fields
    """
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": int,
                },
                "WrapperThing": {
                    "bar": "ThingOne",
                },
                "RepeatedWrapperThing": {
                    "bar": ["ThingOne"],
                },
            }
        )
    ) as dm:
        # Non-message field
        thing_one = dm.ThingOne(1)
        assert thing_one.get_field_message_type("foo") is None

        # Non-repeated sub-message
        wrapper_msg = dm.WrapperThing(thing_one)
        assert wrapper_msg.get_field_message_type("bar") == dm.ThingOne

        # Repeated sub-message
        dm.RepeatedWrapperThing([thing_one])
        assert wrapper_msg.get_field_message_type("bar") == dm.ThingOne


def test_get_field_message_type_invalid_field():
    """Make sure that an appropriate error is raised for an invalid field"""
    with temp_data_model(
        make_proto_def(
            {
                "ThingOne": {
                    "foo": int,
                },
            }
        )
    ) as dm:
        msg = dm.ThingOne(1)
        with pytest.raises(AttributeError):
            msg.get_field_message_type("bar")


#############################
## Serialization Edge Cases ##
#############################
def test_bytes_are_json_serializable():
    """Ensure that bytes are json serializable."""
    with temp_data_model(
        make_proto_def(
            {
                "ByteParty": {
                    "bobject": "bytes",
                },
            }
        )
    ) as dm:
        bytestr = b"json may not like me!"
        msg = dm.ByteParty(bytestr)
        # By default, json serialization doesn't handle bytes; make sure we handle it by default.
        json_msg = msg.to_json()
        assert isinstance(json_msg, str)
        # When we reload back from json, we should still have the same bytestring
        reloaded_msg = dm.ByteParty.from_json(json_msg)
        assert isinstance(reloaded_msg.bobject, bytes)
        assert reloaded_msg.bobject == bytestr


def test_primitive_maps_are_serializable():
    """Ensure that we correctly handle primitive values for de/serialization."""
    with temp_data_model(
        make_proto_def(
            {
                "MapParty": {
                    "mobject": "Dict[str, str]",
                },
            }
        )
    ) as dm:
        msg = dm.MapParty({"foo": "bar"})
        # Make sure we can proto and back
        recon_msg = dm.MapParty.from_proto(msg.to_proto())
        assert isinstance(recon_msg, dm.MapParty)
        assert recon_msg.mobject["foo"] == "bar"
        # Make sure we can json and back
        recon_msg = dm.MapParty.from_json(msg.to_json())
        assert isinstance(recon_msg, dm.MapParty)
        assert recon_msg.mobject["foo"] == "bar"


def test_nonprimitive_maps_are_serializable():
    """Ensure that we correctly handle primitive values for de/serialization."""
    with temp_data_model(
        make_proto_def(
            {
                "ComplexType": {
                    "foo": int,
                },
                "MapParty": {
                    "mobject": "Dict[str, ComplexType]",
                },
            }
        )
    ) as dm:
        nested_msg = dm.ComplexType(100)
        msg = dm.MapParty({"foo": nested_msg})
        # Make sure we can proto and back
        recon_msg = dm.MapParty.from_proto(msg.to_proto())
        assert isinstance(recon_msg, dm.MapParty)
        assert recon_msg.mobject["foo"].foo == 100
        # Make sure we can json and back
        recon_msg = dm.MapParty.from_json(msg.to_json())
        assert isinstance(recon_msg, dm.MapParty)
        assert recon_msg.mobject["foo"].foo == 100


###############
## From Data ##
###############
