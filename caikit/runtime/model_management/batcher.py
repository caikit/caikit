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
The Batcher transparently aggregates individual inference calls into unified
batches to call the run_batch implementation of the wrapped model.
"""

# Standard
from typing import Optional
import inspect
import queue
import threading

# First Party
import alog

# Local
from caikit.core import ModuleBase
from caikit.core.data_model.base import DataBase
from caikit.core.toolkit.wip_decorator import Action, WipCategory, work_in_progress

log = alog.use_channel("BATCHER")


# pylint: disable=too-many-instance-attributes
@work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
class Batcher:
    __doc__ = __doc__

    def __init__(
        self,
        model_name: str,
        model: ModuleBase,
        batch_size: int,
        batch_collect_delay_s: Optional[float] = None,
    ):
        """Configure the batcher to wrap the given model and run batches of at
        most the given size with the given delay to collect a batch.

        Args:
            model_name (str): The name of this model for error reporting
            model (ModuleBase): The model instance that will have inputs batched
            batch_size (int): The maximum size of a batch that will be sent to
                run_batch. Batches of smaller sizes will be sent if no
                additional requests are available.
            batch_collect_delay_s (Optional[float]): Number of seconds to wait
                for more requests to show up when filling the batch.
        """
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._batch_collect_delay_s = batch_collect_delay_s

        # The input queue and the output tasks dict are how the internal batch
        # thread manages receiving work and sharing results
        self._input_q = queue.SimpleQueue()
        self._finished_tasks = {}

        # The request number is used to generate an id for each request, but it
        # needs to be locked. We could use a uuid instead and not lock it, but
        # that would likely be slower (TODO: validate this assumption!)
        self._req_num = 0
        self._id_lock = threading.Lock()

        # These manage the lifecycle of the run thread. In order to avoid a busy
        # thread that never shuts down, the thread will self-terminate when
        # there is no more work ready
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._batch_thread_start_lock = threading.Lock()
        self._batch_thread = None

        # Get a list of any/all default argument values for this model's run
        # function. This is needed to correctly fill in missing values when an
        # individual run() invocation is missing them, but others in the batch
        # have them.
        self._model_run_defaults = {
            param_name: param.default
            for param_name, param in inspect.signature(
                self._model.run
            ).parameters.items()
            if param.default
        }

    def __del__(self):
        """Shut down the internal thread"""
        self.stop()

    ## Public ##

    def run(self, **kwargs) -> DataBase:
        """This run function gives a facade to the underlying model's run
        function that is implemented by running batches of individual requests
        through the model's run_batch method.

        NOTE: Only kwargs accepted to simplify batching across inconsistent sets
            of kwargs (and only kwargs are used in the predict servicer)
        """
        # Make sure the batch thread is active
        self._ensure_batch_thread()

        # Get the ID for this request and make an event
        req_id = self._next_req_id()
        event = threading.Event()
        log.debug2("Starting request %d", req_id)

        # Put the args and kwargs onto the queue with the id and make sure that
        # the batch thread knows it's ready
        log.debug3("Queuing request %s", req_id)
        self._input_q.put((req_id, event, kwargs))
        self._ready_event.set()

        # Wait until notified
        log.debug3("Waiting for request %s", req_id)
        if not self._stop_event.is_set():
            event.wait()
        else:
            log.debug("Not waiting for event %s to complete", req_id)
            self._finished_tasks[req_id] = InterruptedError("Model stopped")
        log.debug3("Request %s finished", req_id)

        # Pull the results from _finished_tasks and handle errors
        res = self._finished_tasks.pop(req_id)
        if isinstance(res, Exception):
            raise RuntimeError(
                f"Exception caught for request {req_id} on model {self._model_name}"
            ) from res
        return res

    def stop(self):
        """Stop this batcher's run thread (cannot be undone)"""
        self._stop_event.set()
        with self._batch_thread_start_lock:
            if self._batch_thread and self._batch_thread.is_alive():
                # There is a race where the batch thread is waiting on the
                # ready_event to indicate that it has a first request so that it
                # doesn't preemptively shut down, so we need to unblock that
                # wait by setting the _ready_event.
                self._ready_event.set()
                if threading.current_thread() is not self._batch_thread:
                    self._batch_thread.join()
                log.debug2("Batch thread fully stopped")

                # Pull any remaining wait conditions off the input queue and
                # terminate them
                interrupt_err = InterruptedError("Model batcher stopped")
                while True:
                    try:
                        req_id, event, _ = self._input_q.get_nowait()
                        self._finished_tasks[req_id] = interrupt_err
                        event.set()
                        log.debug3("Stopped in-flight request [%s]", req_id)
                    except queue.Empty:
                        break
                log.debug2("Done stopping in-flight requests")

    ## Implementaiton Details ##

    def _ensure_batch_thread(self):
        """The run thread will stop itself if there's no work to do, so this
        function is called to ensure that it's up and running
        """
        # Once a model is stopped, it can't be restarted
        if self._stop_event.is_set():
            raise RuntimeError("Cannot restart a stopped model")

        if not self._batch_thread or not self._batch_thread.is_alive():
            with self._batch_thread_start_lock:
                # We only want to grab the lock if we're pretty sure we'll need
                # it, but there's a _chance_ that between the initial check and
                # getting the lock, someone else may have started the thread, so
                # we check again once we've got it.
                if not self._batch_thread or not self._batch_thread.is_alive():
                    log.debug3("Starting up batch thread")
                    self._batch_thread = threading.Thread(target=self._batch_thread_run)
                    self._ready_event.clear()
                    self._batch_thread.start()
                    log.debug3("Thread is started!")

    def _next_req_id(self):
        """Make a unique ID for this request"""
        with self._id_lock:
            self._req_num += 1
            return self._req_num

    def _batch_thread_run(self):
        """This function runs in an independent thread and manages pulling
        requests from the input queue, running the batch, and returning the
        completed results into _finished_tasks.
        """
        # Make sure the first input is available
        log.debug3("Batch thread started")
        self._ready_event.wait()
        log.debug3("Batch thread ready to work")

        # Start a fresh batch and keep the thread alive while there is anything
        # in the queue
        current_batch = []
        while True:
            # If the stop event is set, break out
            if self._stop_event.is_set():
                log.debug("Terminating batch thread for [%s]", self._model_name)
                err = InterruptedError("Model batcher stopped")
                for req_id, event, _ in current_batch:
                    log.debug2("Aborting batched request %s", req_id)
                    self._finished_tasks[req_id] = err
                    event.set()
                break

            # Look for more work and if found, add it to the batch
            no_more_work = False
            try:
                log.debug3(
                    "Getting next request. Waiting for %s", self._batch_collect_delay_s
                )
                next_req = self._input_q.get(
                    block=self._batch_collect_delay_s is not None,
                    timeout=self._batch_collect_delay_s,
                )
                log.debug2(
                    "Adding request %s to batch of size %d",
                    next_req[0],
                    len(current_batch),
                )
                current_batch.append(next_req)
            except queue.Empty:
                if not current_batch:
                    log.debug2(
                        "Shutting down active batch thread for %s", self._model_name
                    )
                    break
                no_more_work = True

            # If the batch is full, or the queue is empty, send the batch and
            # start a new one
            if len(current_batch) == self._batch_size or no_more_work:
                log.debug2("Running batch of size %d", len(current_batch))

                # Collect iterables for each kwarg, filling in gaps with their
                # defaults if possible
                batch_kwargs = {}
                invalid_reqs = {}
                for i, (_, _, req_kwargs) in enumerate(current_batch):
                    # Figure out kwarg names that are new to this batch and
                    # known and new kwargs that this req introduces to the batch
                    # pylint: disable=consider-iterating-dictionary
                    new_kwargs = [
                        kwarg_name
                        for kwarg_name in req_kwargs.keys()
                        if kwarg_name not in batch_kwargs
                    ]
                    missing_kwargs = [
                        kwarg_name
                        for kwarg_name in batch_kwargs.keys()
                        if kwarg_name not in req_kwargs
                    ]

                    # If there are new kwargs and the batch already has entries,
                    # we need to fill those entries in with the default value.
                    # This could cause us to realize that some of those previous
                    # requests were invalid because they were missing required
                    # kwargs!
                    if i > 0:
                        invalid_new_kwargs = [
                            kwarg
                            for kwarg in new_kwargs
                            if kwarg not in self._model_run_defaults
                        ]
                        if invalid_new_kwargs:
                            for j in range(i):
                                log.info(
                                    "<RUN71000210I>"
                                    "Rejecting invalid request to [%s] with "
                                    "missing required kwargs %s",
                                    self._model_name,
                                    invalid_new_kwargs,
                                )
                                invalid_reqs[j] = ValueError(
                                    f"Missing required arguments: {invalid_new_kwargs}"
                                )
                        for new_kwarg in new_kwargs:
                            batch_kwargs[new_kwarg] = [
                                self._model_run_defaults.get(new_kwarg)
                            ] * i

                    # If there are missing kwargs, fill them in with defaults.
                    # This could cause us to realize that this request is
                    # invalid!
                    missing_required_kwargs = [
                        kwarg
                        for kwarg in missing_kwargs
                        if kwarg not in self._model_run_defaults
                    ]
                    if missing_required_kwargs:
                        log.info(
                            "<RUN71000211I>",
                            "Rejecting invalid request to [%s] with missing required kwargs %s",
                            self._model_name,
                            missing_required_kwargs,
                        )
                        invalid_reqs[i] = ValueError(
                            f"Missing required arguments: {missing_required_kwargs}"
                        )
                    for kwarg in missing_kwargs:
                        # NOTE: Invalid requests will be removed later, but we
                        #   fill in the defaults for all requests to keep the
                        #   indices lined up.
                        dflt = self._model_run_defaults.get(kwarg)
                        log.debug3(
                            "Filling in default value for kwarg %s=%s", kwarg, dflt
                        )
                        req_kwargs[kwarg] = dflt

                    # Add to the batch kwargs, regardless of if it is valid so
                    # that the indices of the batch iterables can line up with
                    # the indices in the list of invalid requests
                    for kwarg_name, kwarg_val in req_kwargs.items():
                        batch_kwargs.setdefault(kwarg_name, []).append(kwarg_val)

                # Remove any entries from the batch that have been marked
                # invalid. Work backwards so that the earlier indices are still
                # valid once the later ones have been removed.
                for invalid_idx, invalid_err in sorted(
                    invalid_reqs.items(), reverse=True, key=lambda x: x[0]
                ):
                    # Slice the arg vals out of the kwarg lists
                    for kwarg_name, kwarg_vals in list(batch_kwargs.items()):
                        batch_kwargs[kwarg_name] = (
                            kwarg_vals[:invalid_idx] + kwarg_vals[invalid_idx + 1 :]
                        )

                    # Notify the waiting task and remove the entry from the
                    # batch so that it isn't processed later
                    req_id, event, _ = current_batch[invalid_idx]
                    current_batch = (
                        current_batch[:invalid_idx] + current_batch[invalid_idx + 1 :]
                    )
                    self._finished_tasks[req_id] = invalid_err
                    event.set()

                # Call the batch_run
                if current_batch:
                    try:
                        log.debug(
                            "Running with concrete batch size %d", len(current_batch)
                        )
                        log.debug4(batch_kwargs)
                        batch_res = self._model.run_batch(**batch_kwargs)
                        # pylint: disable=line-too-long
                        assert len(batch_res) == len(
                            current_batch
                        ), f"Got result of size [{len(batch_res)}] for batch of size [{len(current_batch)}]"
                        for i, (req_id, event, _) in enumerate(current_batch):
                            self._finished_tasks[req_id] = batch_res[i]
                            event.set()

                    # If an error occurred, push the single error as the result of
                    # all the requests in the batch
                    # pylint: disable=broad-exception-caught
                    except Exception as err:
                        log.error(
                            "Caught exception in batch thread for model %s: %s",
                            self._model_name,
                            err,
                        )
                        for req_id, event, _ in current_batch:
                            self._finished_tasks[req_id] = err
                            event.set()

                # Reset the next batch
                current_batch = []
