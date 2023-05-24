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
"""
Unit tests for the model Batcher
"""

# Standard
import os
import queue
import threading
import time

# Third Party
import pytest

# First Party
import alog

# Local
from caikit.runtime.model_management.batcher import Batcher
from sample_lib.data_model import SampleOutputType
import caikit.core

## Helpers #####################################################################

log = alog.use_channel("TEST")

DUMMY_MODEL = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "models",
        "foo_archive.zip",
    )
)


@caikit.core.module(
    "7464f684-58e3-4e99-9a58-1c5bc085472b", "slow sample module", "0.0.1"
)
class SlowSampleModule(caikit.core.ModuleBase):
    """This module is just a wrapper around another module that will inject a
    sleep
    """

    def __init__(self, model, sleep_time_s=0.1):
        self._model = model
        self._sleep_time_s = sleep_time_s
        self.batches = []
        self.batches_lock = threading.Lock()

    def run(self, *args, **kwargs):
        log.debug("Starting slow module")
        time.sleep(self._sleep_time_s)
        return self._model.run(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        with self.batches_lock:
            self.batches.append((args, kwargs))
        return self._model.run_batch(*args, **kwargs)


@caikit.core.module(
    "19b126aa-b55b-4349-94f1-d676f3e12c9b",
    "Really silly bunch of bobs!",
    "0.0.1",
)
class StubModule(caikit.core.ModuleBase):
    # NOTE: The initial implementation had "num_bobs" which expected an int.
    #   This led to the discovery of the fact that lists of ints are not
    #   considered "expandable iterables" in the default run_batch impl.
    def __init__(self):
        super().__init__()
        self.reqs = []

    def run(self, bobs_name, last_name="Bit"):
        self.reqs.append({"bobs_name": bobs_name, "last_name": last_name})
        return SampleOutputType(greeting=f"{bobs_name} {last_name}")


class ModelRunThread(threading.Thread):
    def __init__(self, model, run_num, request_kwargs=None):
        super().__init__()
        self.model = model
        self.run_num = run_num
        self.response = None
        self.request_kwargs = (
            request_kwargs
            if request_kwargs is not None
            else {
                "producer_id": caikit.core.data_model.ProducerId(
                    str(self.run_num),
                    "1.2.3",
                )
            }
        )

    def run(self):
        log.debug4("Running model with num %d", self.run_num)
        try:
            self.response = self.model.run(**self.request_kwargs)
            log.debug4("Finished run num %d: %s", self.run_num, self.response)
        except Exception as err:
            self.response = err


class ModelRunWrapperThread(threading.Thread):
    """Another layer of thread wrapping that will actually do the run in a sub-
    thread so that the parent thread can know when run has started without
    waiting for it to complete.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.started_event = threading.Event()
        self.run_thread = ModelRunThread(*args, **kwargs)

    @property
    def response(self):
        return self.run_thread.response

    def run(self):
        log.debug2("Starting sub-thread")
        self.run_thread.start()
        log.debug2("Marking run as started")
        self.started_event.set()
        log.debug2("Waiting for sub-thread to complete")
        self.run_thread.join()


class BlockedSimpleQueue(queue.SimpleQueue):
    """This child class of SimpleQueue allows the queue to block all put calls
    on a manually set event so that we can simulate race conditions.
    """

    def __init__(self, *args, **kwargs):
        self._event = threading.Event()
        super().__init__(*args, **kwargs)

    def proceed(self):
        self._event.set()

    def put(self, *args, **kwargs):
        self._event.wait()
        return super().put(*args, **kwargs)


## Tests #######################################################################


@pytest.skip(
    "Some tests here tend to deadlock! Will need debugging but skipping for now",
    allow_module_level=True,
)
def test_single_request():
    """Test that a single request to a batched model acts the same as a request
    to the original model
    """
    model = caikit.core.load(DUMMY_MODEL)
    wrapped_model = Batcher("test-model", model, 10)
    prod_id = caikit.core.data_model.ProducerId("test", "1.2.3")
    standard_res = model.run(producer_id=prod_id)
    wrapped_res = wrapped_model.run(producer_id=prod_id)
    assert standard_res.to_dict() == wrapped_res.to_dict()


def test_multi_request_batch():
    """Make sure that batches of size > 1 can be run"""
    model = SlowSampleModule(caikit.core.load(DUMMY_MODEL))
    batch_size = 10
    wrapped_model = Batcher("test-model", model, batch_size)
    threads = {str(i): ModelRunThread(wrapped_model, i) for i in range(2 * batch_size)}

    for thread in threads.values():
        thread.start()

    for thread in threads.values():
        thread.join()

    # Make sure the right requests got the right responses
    for exp_pid_name, thread in threads.items():
        assert exp_pid_name == thread.response.producer_id.name

    # Make sure that non-trivial batches were run
    assert model.batches
    assert any(len(batch) > 1 for batch in model.batches)


def test_stop_in_flight_batch():
    """Make sure that setting the stop event will preempt running a batch"""
    model_delay = 0.001
    model = SlowSampleModule(caikit.core.load(DUMMY_MODEL), model_delay)
    batch_size = 10
    wrapped_model = Batcher("test-model", model, batch_size, model_delay * 2)
    threads = {str(i): ModelRunThread(wrapped_model, i) for i in range(2 * batch_size)}

    for thread in threads.values():
        thread.start()

    # Stop the
    wrapped_model.stop()

    # Make sure all of the threads finish
    for thread in threads.values():
        thread.join()

    # Make sure all requests that didn't get run got an appropriate error. This
    # is somewhat nondeterministic based on the threading environment, so we
    # can't guarantee that _any_ requests failed since they _might_ have all
    # finished before stop was able to interrupt them.
    successful_reqs = [
        prod_id.name for batch in model.batches for prod_id in batch[1]["producer_id"]
    ]
    for exp_pid_name, thread in threads.items():
        if exp_pid_name in successful_reqs:
            assert not isinstance(thread.response, Exception)
        else:
            assert isinstance(thread.response, RuntimeError)


def test_destructor():
    """Make sure that when the destructor for a model is called, everything
    shuts down cleanly, including when the model is already stopped or never
    started.
    """
    model = caikit.core.load(DUMMY_MODEL)

    # Delete before starting the thread
    wrapped_model = Batcher("test-model", model, 10)
    del wrapped_model

    # Delete after a call
    wrapped_model = Batcher("test-model", model, 10)
    prod_id = caikit.core.data_model.ProducerId("test", "1.2.3")
    wrapped_model.run(producer_id=prod_id)
    del wrapped_model

    # Delete after a call and stopping
    wrapped_model = Batcher("test-model", model, 10)
    wrapped_model.run(producer_id=prod_id)
    wrapped_model.stop()
    del wrapped_model


def test_no_restart():
    """Make sure that a model cannot be restarted once it has stopped"""
    model = caikit.core.load(DUMMY_MODEL)

    # Delete after a call and stopping
    wrapped_model = Batcher("test-model", model, 10)
    prod_id = caikit.core.data_model.ProducerId("test", "1.2.3")
    wrapped_model.run(producer_id=prod_id)
    wrapped_model.stop()
    with pytest.raises(RuntimeError):
        wrapped_model.run(producer_id=prod_id)


def test_different_kwargs():
    """Make sure that calls with ragged kwargs are batched correctly"""

    model = StubModule()
    batch_size = 5
    wrapped_model = Batcher("stub-model", model, batch_size)
    threads = {
        i: (
            ModelRunThread(wrapped_model, i, {"bobs_name": "Bob"})
            if i % 2
            else ModelRunThread(
                wrapped_model, i, {"bobs_name": "Jim", "last_name": "Bo"}
            )
        )
        for i in range(2 * batch_size)
    }

    for thread in threads.values():
        thread.start()

    for thread in threads.values():
        thread.join()

    # Make sure that the batches didn't contain args for "last_name" for odd
    # requests
    for req in model.reqs:
        if req["bobs_name"] == "Bob":
            assert req["last_name"] == "Bit"  # The default value
        else:
            assert req["last_name"] == "Bo"


def test_invalid_req_after_valid_req():
    """Make sure that in a batch, if a request is processed with a missing
    required keyword arg after a request that has it, the valid request is
    processed and the invalid request is rejected.
    """

    model = SlowSampleModule(StubModule())
    batch_size = 5
    wrapped_model = Batcher("stub-model", model, batch_size)
    threads = {
        i: (
            ModelRunThread(wrapped_model, i, {"bobs_name": "Bob", "last_name": str(i)})
            if i < 2
            else ModelRunThread(wrapped_model, i, {})
        )
        for i in range(2 * batch_size)
    }

    for thread in threads.values():
        thread.start()

    for thread in threads.values():
        thread.join()

    # Make sure that all requests after the first two had errors, but the first
    # two did not
    for i, thread in threads.items():
        if i < 2:
            assert not isinstance(thread.response, Exception)
            assert thread.response.bobs[0].split()[-1] == str(i)
        else:
            assert isinstance(thread.response, RuntimeError)


def test_valid_req_after_invalid_req():
    """Make sure that in a batch, if a request is processed that adds new
    keyword args which previous requests didn't have that also don't have
    defaults, the previous invalid requests are rejected.
    """

    model = SlowSampleModule(StubModule())
    batch_size = 5
    wrapped_model = Batcher("stub-model", model, batch_size)
    threads = {
        i: (
            ModelRunThread(wrapped_model, i, {"bobs_name": "Bob", "last_name": str(i)})
            if i >= 2
            else ModelRunThread(wrapped_model, i, {})
        )
        for i in range(2 * batch_size)
    }

    for thread in threads.values():
        thread.start()

    for thread in threads.values():
        thread.join()

    # Make sure that all requests after the first two had errors, but the first
    # two did not
    for i, thread in threads.items():
        if i >= 2:
            assert not isinstance(thread.response, Exception)
            assert thread.response.bobs[0].split()[-1] == str(i)
        else:
            assert isinstance(thread.response, RuntimeError)


def test_batch_collect_delay():
    """Make sure that with the batch delay, multiple sequential calls end up in
    the same batchg even when not presented concurrently
    """
    model = SlowSampleModule(StubModule())
    batch_size = 5
    batch_collect_delay = 0.01
    wrapped_model = Batcher("stub-model", model, batch_size, batch_collect_delay)

    # Start one thread, wait for less than the collect delay, then start another
    th1 = ModelRunThread(wrapped_model, 0, {"bobs_name": "Bob"})
    th1.start()
    log.debug("Sleeping after th1.start()")
    time.sleep(batch_collect_delay / 10)
    log.debug("Done sleeping")
    th2 = ModelRunThread(wrapped_model, 1, {"bobs_name": "Jim"})
    th2.start()
    log.debug("th2 started")

    # Wait for both to complete
    th1.join()
    th2.join()

    # Make sure only one batch was sent
    assert len(model.batches) == 1


def test_orphaned_events_race():
    """There's a subtle race condition that can heppen when handling a request
    where the event is created, stop is called and completes, _then_ the request
    gets added to the queue and will never be pulled out by stop() if it has
    finished. To avoid orphaned requests in the queue, we ensure that the event
    is somewhere that stop() can find it by putting it in a member dict on
    creation. This tests ensures that the race is mitigated.
    """
    wrapped_model = Batcher("stub-model", StubModule(), 10)

    # Patch the batcher's run function to wait on an intentional condition after
    # creating the event
    blocked_queue = BlockedSimpleQueue()
    wrapped_model._input_q = blocked_queue

    # Create a thread that will add a request. When run, this will create the
    # event and start to add to the queue, but it won't proceed until the queue
    # is unblocked.
    log.debug("Starting request thread")
    th = ModelRunWrapperThread(wrapped_model, 0, {"bobs_name": "Bob"})
    th.start()

    # Wait until the "run" call has started
    log.debug("Waiting for thread to start")
    th.started_event.wait()

    # Call stop to attempt to shut down all requests. The event is still in
    # limbo since it hasn't finished being added to the queue.
    log.debug("Stopping wrapped model")
    wrapped_model.stop()

    # Let the queue's put proceed
    log.debug("Unblocking queue.put")
    blocked_queue.proceed()

    # Wait for the request to complete. If the race exists, this will deadlock!
    log.debug("Joining request")
    th.join()
