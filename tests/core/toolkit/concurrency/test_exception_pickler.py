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
import pickle
import traceback

# Third Party
import pytest

# Local
from caikit.core.toolkit.concurrency.pickling_exception import (
    ExceptionPickler,
    PickleFailureFallbackException,
)


#### Custom exception types for testing ##################################
class WellBehavedException(Exception):
    def __init__(self, *args):
        self.message = args[0]
        self.thing = args[1]


class PoorlyBehavedException(Exception):
    def __init__(self, message, private_arg):
        self.message = message
        self._private_arg = private_arg

    def __str__(self):
        return f"{self.message} ({self._private_arg})"


class ReallyPoorlyBehavedException(Exception):
    def __init__(self, message):
        self._something_entirely_different = message

    def __str__(self):
        return self._something_entirely_different


def raise_from_all(*exceptions):
    """Given many exceptions, raise each one from the last one to make a chain of exceptions"""
    last_exception = None
    chain = []

    for e in exceptions:
        try:
            if last_exception:
                raise e from last_exception
            raise e
        except Exception as e:
            last_exception = e
            chain.insert(0, last_exception)

    return chain


def get_traceback(exc):
    """Provides <py310 compatibility"""
    return traceback.format_exception(type(exc), value=exc, tb=exc.__traceback__)


def test_exceptions_keep_cause_and_context():
    chain = raise_from_all(ValueError("the_little_cause"), ValueError("the_big_effect"))

    cause = chain[1]
    effect = chain[0]

    pickler = ExceptionPickler(effect)
    pickler: ExceptionPickler = pickle.loads(pickle.dumps(pickler))

    unpickled_effect = pickler.get()
    assert unpickled_effect.__cause__ is not None

    assert str(unpickled_effect) == str(effect)
    assert str(unpickled_effect.__cause__) == str(cause)

    tb = "".join(get_traceback(unpickled_effect))
    assert str(cause) in tb


def test_non_conforming_exception_types_can_be_pickled():

    exception = PoorlyBehavedException(message="foo", private_arg="bar")

    # Regular old pickling fails
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(exception))

    # But the pickler handles it
    pickler = ExceptionPickler(exception)
    unpickled = pickle.loads(pickle.dumps(pickler)).get()

    assert isinstance(unpickled, PoorlyBehavedException)
    assert str(unpickled) == str(exception)


def test_really_non_conforming_exception_types_can_be_pickled():
    exception = ReallyPoorlyBehavedException(message="foo")

    # Regular old pickling fails
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(exception))

    # But the pickler handles it
    pickler = ExceptionPickler(exception)
    unpickled = pickle.loads(pickle.dumps(pickler)).get()

    # We can't get the real type back, but we can get the string representation at least
    assert isinstance(unpickled, PickleFailureFallbackException)
    assert str(unpickled) == str(exception)


def test_big_chain_of_interesting_exceptions_can_be_pickled():
    chain = raise_from_all(
        ValueError("buffer size must be > 0!"),
        ReallyPoorlyBehavedException(message="input error!"),
        PoorlyBehavedException(
            message="Failed to validate input data", private_arg="code 7260"
        ),
        WellBehavedException("Training failed due to invalid input", "retryable=False"),
    )

    pickler = ExceptionPickler(chain[0])
    pickler = pickle.loads(pickle.dumps(pickler))

    exception = pickler.get()

    assert isinstance(exception.__cause__.__cause__.__cause__, ValueError)

    tb = "".join(get_traceback(exception))
    for exc in chain:
        assert str(exc) in tb


def test_pickler_works_with_mix_of_arg_and_kwarg():
    exception = PoorlyBehavedException(
        "this is my arg", private_arg="and this is my kwarg"
    )

    unpickled = pickle.loads(pickle.dumps(ExceptionPickler(exception))).get()

    assert isinstance(unpickled, PoorlyBehavedException)
    assert str(exception) == str(unpickled)
