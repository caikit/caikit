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
from typing import Type
import abc
import multiprocessing
import os
import threading
import traceback

# Third Party
from grpc import StatusCode

# First Party
import alog

# Local
from caikit.core.modules import ModuleBase
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.work_management.destroyable_thread import DestroyableThread
import caikit.core

log = alog.use_channel("TRN_EXCTRS")
error = caikit.core.toolkit.errors.error_handler.get(log)


OOM_EXIT_CODE = 137

# NOTE: Following would get replaced with training backends potentially
# in future.
# NOTE: Instead of using executors like Local / Subprocess
# it might make sense to instead use concurrent.Future based
# executors, so ThreadPoolExecutors and ProcessPoolExecutors, but
# ProcessPoolExecutor doesn't support `fork` start method
class TrainSaveExecutorBase(abc.ABC):
    @abc.abstractmethod
    def cancel(self):
        """Function to abort train and save operation on the executor"""

    @abc.abstractmethod
    def train_and_save_model(self, *args, **kwargs):
        """Method to run train and save function(s) on the executor"""

    @staticmethod
    def _train_and_save(
        module_class: Type[ModuleBase], model_path: str, *args, **kwargs
    ):
        """Default implementation of train and save method using module_class"""

        error.type_check("<RUN41176398E>", str, model_path=model_path)

        with alog.ContextTimer(
            log.debug, "Done training %s in: ", module_class.__name__
        ):
            model = module_class.train(*args, **kwargs)

        # Save it
        with alog.ContextTimer(
            log.debug,
            "Done saving %s to %s in: ",
            module_class.__name__,
            model_path,
        ):
            model.save(model_path)

        log.info("model saved successfully at {}".format(model_path))


class LocalTrainSaveExecutor(TrainSaveExecutorBase):
    def __init__(self, cancel_event) -> None:
        self.complete_event = threading.Event()

        # collect all events
        self.events = [cancel_event, self.complete_event]

        # NOTE: worker is assigned at a later stage for Local
        self._worker = None

        self.__cancel_event = cancel_event

    def __del__(self):
        """
        NOTE: This is NOT how execution should be cancelled.
        This function is designed to make sure cleanup happens.
        """
        self.cancel()

    # pylint: disable=arguments-differ
    def train_and_save_model(
        self,
        module_class: Type[ModuleBase],
        model_path: str,
        *args,
        **kwargs,
    ):
        """This function performs a single training and can be run inside a
        subprocess if needed
        """

        args += (module_class, model_path)
        try:
            # Train it
            with alog.ContextTimer(
                log.debug, "Done training %s in: ", module_class.__name__
            ):
                self._worker = DestroyableThread(
                    self.complete_event,
                    TrainSaveExecutorBase._train_and_save,
                    *args,
                    **kwargs,
                )
                self._worker.start()
                self.complete_event.wait()

                if self.__cancel_event.is_set():
                    self.cancel()

                self._worker.join()

                # Fetch the results or throw error if the
                # task threw exception
                return self._worker.get_or_throw()

        # Handle errors as CaikitRuntime errors with appropriate error codes
        except CaikitRuntimeException as e:
            log.warning(
                {
                    "log_code": "<RUN555430380W>",
                    "message": e.message,
                    "error_id": e.id,
                    **e.metadata,
                }
            )
            raise e
        except (TypeError, ValueError) as e:
            log.warning(
                {
                    "log_code": "<RUN868639039W>",
                    "message": repr(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT,
                f"Exception raised during training. This may be a problem with your input: {e}",
            ) from e
        except Exception as e:
            log.warning(
                {
                    "log_code": "<RUN490967039W>",
                    "message": repr(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            raise CaikitRuntimeException(
                StatusCode.INTERNAL,
                f"Exception raised during training: {e}",
            ) from e

    def cancel(self) -> None:
        """Function to abort train and save operation on the executor"""
        self._worker.destroy()

        log.error("<RUN50125604E>", "Training cancelled.")

        raise CaikitRuntimeException(
            StatusCode.CANCELLED,
            "Training request terminated!",
        )


class SubProcessTrainSaveExecutor(TrainSaveExecutorBase):
    class _ErrorCaptureProcess(multiprocessing.get_context("fork").Process):
        """This class wraps a Process and keeps track of any errors that occur
        during execution

        NOTE: We explicitly use "fork" here for two reasons:
            1. It's faster
            2. Due to the auto-generated classes with stream sources, "spawn"
               can result in missing classes since it performs a full re-import,
               but does not regenerate the service APIs
        """

        def __init__(self, *args, event=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.error = None
            self._completion_event = event

        def run(self, *args, **kwargs):
            try:
                return super().run(*args, **kwargs)

            # Catch any errors thrown within a subprocess so that they can be
            # forwarded to the parent
            # pylint: disable=broad-exception-caught
            except Exception as err:
                self.error = err

            finally:
                self._completion_event.set()

        def join(self, timeout=None):
            """
            Wait until child process terminates
            """
            super().join()
            if not self._completion_event.is_set():
                self._completion_event.set()

    def __init__(self, cancel_event) -> None:

        self.complete_event = multiprocessing.Event()

        self.events = [cancel_event, self.complete_event]
        self._worker = None

        self.__cancel_event = cancel_event

    def __del__(self):
        """
        NOTE: This is NOT how execution should be cancelled.
        This function is designed to make sure cleanup happens.
        """
        self._cleanup()

    def _cleanup(self):
        """Function to clearup running workers"""
        if self._worker.is_alive():
            self._worker.terminate()
        self._worker.close()

    # pylint: disable=arguments-differ
    def train_and_save_model(
        self, module_class: Type[ModuleBase], model_path: str, *args, **kwargs
    ):

        self._worker = self._ErrorCaptureProcess(
            event=self.complete_event,
            target=TrainSaveExecutorBase._train_and_save,
            args=(module_class, model_path),
            kwargs=kwargs,
        )

        self._worker.start()

        # FIXME: Bad version of accomplishing conditional check between
        # processing kill and process completion which is required
        # for cases when process gets killed by OS
        while True:
            if self._worker.exitcode:
                self.complete_event.set()
                break
            elif self.complete_event.is_set():
                break

        if self.__cancel_event.is_set():
            # Since we are using process here, we cannot rely on
            # checking is_complete flag to be available to check if
            # the training was completed or cancelled. Therefore,
            # we will check if worker is alive and event is set.
            # This does create an edge case, were if the thread is done
            # naturally and at the exact same time, the request is cancelled
            # but in that case, the training is anyways already finished
            # so that shouldn't create huge problems
            self.cancel()

        self._worker.join()

        # If an error occurred, reraise it here
        # TODO: Make sure the stack trace is preserved
        if self._worker.error is not None:
            if isinstance(self._worker.error, CaikitRuntimeException):
                raise self._worker.error
            raise CaikitRuntimeException(
                StatusCode.INTERNAL,
                "Error caught in training subprocess",
            ) from self._worker.error

        # If process exited with a non-zero exit code
        if self._worker.exitcode and self._worker.exitcode != os.EX_OK:
            if self._worker.exitcode == OOM_EXIT_CODE:
                exception = CaikitRuntimeException(
                    StatusCode.RESOURCE_EXHAUSTED,
                    "Training process died with OOM error!",
                )
            else:
                exception = CaikitRuntimeException(
                    StatusCode.UNKNOWN,
                    f"Training process died with exit code {self._worker.exitcode}",
                )

            raise exception

        self._cleanup()

    def cancel(self):

        if self._worker.is_alive():
            self._worker.terminate()
        self._worker.close()

        log.error("<RUN57624710E>", "Training cancelled.")

        raise CaikitRuntimeException(
            StatusCode.CANCELLED,
            "Training request terminated!",
        )
