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
import multiprocessing
import time

# Third Party
import pytest

# Local
from caikit.core.toolkit.concurrency.destroyable_process import DestroyableProcess
from tests.core.toolkit.concurrency.test_exception_pickler import (
    ReallyPoorlyBehavedException,
    get_traceback,
)

## Helpers #####################################################################


EXPECTED_THROW = ValueError("test-any-error")
EXPECTED_SUCCESS = "test-any-result"
UNPICKLABLE_ERROR = ReallyPoorlyBehavedException(message="This will not pickle")


def infinite_wait():
    while True:
        time.sleep(0.1)


def long_sleep():
    time.sleep(1000)


def thrower():
    raise EXPECTED_THROW


def nested_thrower():
    try:
        raise EXPECTED_THROW
    except Exception as e:
        raise ValueError("some other error!") from e


def bad_thrower():
    raise UNPICKLABLE_ERROR


def succeeder():
    return EXPECTED_SUCCESS


@pytest.fixture(
    params=["fork", "forkserver", "spawn"],
)
def process_type(request):
    yield request.param


## Tests #######################################################################


def test_processes_can_be_interrupted(process_type):
    proc = DestroyableProcess(process_type, infinite_wait)
    proc.start()
    assert not proc.destroyed
    assert not proc.canceled
    assert not proc.ran
    assert not proc.threw
    proc.destroy()
    proc.join(60)
    assert not proc.is_alive()
    assert proc.destroyed
    assert proc.canceled
    assert proc.ran
    assert not proc.threw


def test_processes_can_return_results(process_type):
    proc = DestroyableProcess(process_type, succeeder, return_result=True)
    proc.start()
    proc.join()
    assert EXPECTED_SUCCESS == proc.get_or_throw()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_process_not_canceled_after_success(process_type):
    proc = DestroyableProcess(process_type, succeeder)
    proc.start()
    proc.join()
    assert not proc.canceled
    proc.destroy()
    assert not proc.canceled


def test_processes_can_be_set_to_not_return_results(process_type):
    proc = DestroyableProcess(process_type, succeeder, return_result=False)
    proc.start()
    proc.join()
    assert proc.get_or_throw() is None
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_processes_can_throw(process_type):
    proc = DestroyableProcess(process_type, thrower)
    proc.start()
    proc.join()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert proc.threw

    with pytest.raises(ValueError) as ctx:
        proc.get_or_throw()

    assert str(EXPECTED_THROW) == str(ctx.value)


def test_processes_will_not_execute_if_destroyed_before_starting(process_type):
    proc = DestroyableProcess(process_type, long_sleep)
    proc.destroy()
    proc.start()
    assert not proc.is_alive()
    proc.join()
    with pytest.raises(RuntimeError):
        proc.get_or_throw()
    assert proc.destroyed
    assert proc.canceled
    assert not proc.ran
    assert proc.threw


def test_event_is_set_on_completion(process_type):
    event = multiprocessing.get_context(process_type).Event()
    proc = DestroyableProcess(process_type, succeeder, completion_event=event)
    assert not event.is_set()
    proc.start()
    proc.join()
    assert event.is_set()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_event_is_set_on_exception(process_type):
    event = multiprocessing.get_context(process_type).Event()
    proc = DestroyableProcess(process_type, thrower, completion_event=event)
    assert not event.is_set()
    proc.start()
    proc.join()
    assert event.is_set()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert proc.threw


def test_default_event_is_set_on_completion(process_type):
    proc = DestroyableProcess(process_type, succeeder)
    assert not proc.completion_event.is_set()
    proc.start()
    proc.join()
    assert proc.completion_event.is_set()


def test_process_can_raise_unpicklable_exception(process_type):
    proc = DestroyableProcess(process_type, bad_thrower)
    proc.start()
    proc.join()

    assert proc.threw
    exception = proc.error

    assert str(UNPICKLABLE_ERROR) in str(exception)


def test_process_can_raise_nested_exception(process_type):
    proc = DestroyableProcess(process_type, nested_thrower)
    proc.start()
    proc.join()

    assert proc.threw

    assert proc.error.__cause__ is not None
    tb = get_traceback(proc.error)
    assert len(tb) > 2
    assert (
        "The above exception was the direct cause of the following exception:"
        in "".join(tb)
    )
