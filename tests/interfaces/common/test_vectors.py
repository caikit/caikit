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
"""Test for embedding vectors
"""
# Standard
from collections import namedtuple

# Third Party
import numpy as np
import pytest

# Local
from caikit.interfaces.common import data_model as dm

## Setup #########################################################################

DUMMY_VECTOR_SHAPE = (5,)
RANDOM_SEED = 77
np.random.seed(RANDOM_SEED)
random_number_generator = np.random.default_rng()

# To tests the limits of our type-checking, this can replace our legit data objects
TRICK_SEQUENCE = namedtuple("Trick", "values")


@pytest.fixture
def simple_array_of_floats():
    return [1.1, 2.2]


@pytest.fixture
def simple_array_of_ints():
    return ["foo", 1, 2, 3, 4]


@pytest.fixture
def random_numpy_vector1d_float32():
    return random_number_generator.random(DUMMY_VECTOR_SHAPE, dtype=np.float32)


@pytest.fixture
def random_numpy_vector1d_float64():
    return random_number_generator.random(DUMMY_VECTOR_SHAPE, dtype=np.float64)


@pytest.fixture
def random_python_vector1d_float(random_numpy_vector1d_float32):
    return random_numpy_vector1d_float32.tolist()


## Tests ########################################################################


@pytest.mark.parametrize(
    "sequence",
    [
        dm.PyFloatSequence(),
        dm.NpFloat32Sequence(),
        dm.NpFloat64Sequence(),
        TRICK_SEQUENCE(values=None),
    ],
    ids=type,
)
def test_empty_sequences(sequence):
    """No type check error with empty sequences"""
    new_dm_from_init = dm.Vector1D(sequence)
    assert isinstance(new_dm_from_init.data, type(sequence))
    assert new_dm_from_init.data.values is None

    # Test proto
    proto_from_dm = new_dm_from_init.to_proto()
    new_dm_from_proto = dm.Vector1D.from_proto(proto_from_dm)
    assert isinstance(new_dm_from_proto, dm.Vector1D)
    assert new_dm_from_proto.data.values is None

    # Test json
    json_from_dm = new_dm_from_init.to_json()
    new_dm_from_json = dm.Vector1D.from_json(json_from_dm)
    assert isinstance(new_dm_from_json, dm.Vector1D)
    assert new_dm_from_json.data.values == []


def test_vector1d_iterator_error():
    """Cannot just shove in an iterator and expect it to work"""
    with pytest.raises(ValueError):
        dm.Vector1D(data=[1.1, 2.2, 3.3])


def _assert_array_check(new_array, data_values, float_type):
    for value in new_array.data.values:
        assert isinstance(value, float_type)
    np.testing.assert_array_equal(new_array.data.values, data_values)


@pytest.mark.parametrize(
    "float_seq_class, random_values, float_type",
    [
        (dm.PyFloatSequence, "random_python_vector1d_float", float),
        (dm.NpFloat32Sequence, "random_numpy_vector1d_float32", np.float32),
        (dm.NpFloat64Sequence, "random_numpy_vector1d_float64", np.float64),
        (
            TRICK_SEQUENCE,
            "simple_array_of_floats",
            float,
        ),  # Sneaky but tests corner cases for now
    ],
)
def test_vector1d_dm(float_seq_class, random_values, float_type, request):

    # Test init
    fixture_values = request.getfixturevalue(random_values)
    dm_init = dm.Vector1D(data=float_seq_class(fixture_values))
    _assert_array_check(dm_init, fixture_values, float_type)

    # Test proto
    dm_to_proto = dm_init.to_proto()
    dm_from_proto = dm.Vector1D.from_proto(dm_to_proto)
    _assert_array_check(dm_from_proto, fixture_values, float_type)

    # Test json
    dm_to_json = dm_init.to_json()
    dm_from_json = dm.Vector1D.from_json(dm_to_json)
    _assert_array_check(
        dm_from_json, fixture_values, float
    )  # NOTE: always float after json


@pytest.mark.parametrize(
    "float_seq_class, random_values, float_type",
    [
        (dm.PyFloatSequence, "random_python_vector1d_float", float),
        (dm.NpFloat32Sequence, "random_numpy_vector1d_float32", np.float32),
        (dm.NpFloat64Sequence, "random_numpy_vector1d_float64", np.float64),
    ],
)
def test_vector1d_dm_from_vector(float_seq_class, random_values, float_type, request):
    fixture_values = request.getfixturevalue(random_values)
    v = dm.Vector1D.from_vector(fixture_values)
    assert isinstance(v.data, float_seq_class)
    assert isinstance(v.data.values[0], float_type)
    _assert_array_check(v, fixture_values, float_type)
