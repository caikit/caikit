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
"""Base class with common functionality across all caikit servers"""
# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import abc
import signal

# Third Party
from prometheus_client import start_http_server

# First Party
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.core.exceptions import error_handler
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.work_management.abortable_context import ThreadInterrupter
import caikit

log = alog.use_channel("SERVR-BASE")
error = error_handler.get(log)


class ServerThreadPool:
    """Simple wrapper for all servers to share a single thread pool"""

    @staticmethod
    def _build_pool() -> ThreadPoolExecutor:
        config = caikit.get_config()
        # Leave in backwards compatibility for the old runtime.grpc.server_thread_pool_size
        # parameter, which many users may have deployed with.
        if pool_size := config.runtime.grpc.server_thread_pool_size:
            log.info("Using legacy runtime.grpc.server_thread_pool_size configuration")
        else:
            pool_size = config.runtime.server_thread_pool_size

        error.type_check("<RUN92632238E>", int, pool_size=pool_size)

        pool = ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="caikit_runtime"
        )

        return pool

    # py3.9 compatibility: Can't call @staticmethod on class attribute initialization
    pool = _build_pool.__get__(object, None)()


class RuntimeServerBase(abc.ABC):  # pylint: disable=too-many-instance-attributes
    __doc__ = __doc__
    _metrics_server_started = False

    def __init__(self, base_port: int, tls_config_override: Optional[aconfig.Config]):
        self.config = get_config()
        self.port = base_port
        self.tls_config = (
            tls_config_override if tls_config_override else self.config.runtime.tls
        )
        log.debug4("Full caikit config: %s", self.config)

        # Configure using the log level and formatter type specified in config.
        caikit.core.toolkit.logging.configure()

        # We should always be able to stand up an inference service
        self.enable_inference = self.config.runtime.service_generation.enable_inference
        self.enable_training = self.config.runtime.service_generation.enable_training
        self.inference_service: Optional[ServicePackage] = (
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.INFERENCE,
            )
            if self.enable_inference
            else None
        )

        # But maybe not always a training service
        try:
            training_service: Optional[ServicePackage] = (
                ServicePackageFactory.get_service_package(
                    ServicePackageFactory.ServiceType.TRAINING,
                )
                if self.enable_training
                else None
            )
        except CaikitRuntimeException as e:
            log.warning("Cannot stand up training service, disabling training: %s", e)
            training_service = None

        self.training_service = training_service

        # create runtime info service
        self.runtime_info_service: Optional[
            ServicePackage
        ] = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFO,
        )

        self.thread_pool: ThreadPoolExecutor = ServerThreadPool.pool
        # Create an interrupter that can be used to handle request cancellations or timeouts.
        # A separate instance is held per-server so each server can handle the lifetime of their
        # own interrupter.
        self.interrupter: Optional[ThreadInterrupter] = (
            ThreadInterrupter() if self.config.runtime.use_abortable_threads else None
        )

        # Handle interrupts
        # NB: This means that stop() methods will be called even if the process is interrupted
        # before the start() method is called
        self._intercept_interrupt_signal()

    @classmethod
    def _start_metrics_server(cls) -> None:
        """Start a single instance of the metrics server based on configuration"""
        if not cls._metrics_server_started and get_config().runtime.metrics.enabled:
            log.info(
                "Serving prometheus metrics on port %s",
                get_config().runtime.metrics.port,
            )
            with alog.ContextTimer(log.info, "Booted metrics server in "):
                start_http_server(get_config().runtime.metrics.port)
            cls._metrics_server_started = True

    def interrupt(self, signal_, _stack_frame):
        log.info(
            "<RUN87630120I>",
            "Caikit Runtime received interrupt signal %s, shutting down",
            signal_,
        )
        self.stop()

    def _intercept_interrupt_signal(self) -> None:
        """Intercept signal handlers to allow the server to stop on interrupt.
        Calling this on a non-main thread has no effect.
        This does not override any existing non-default signal handlers,
        it will call them all in the reverse order they are registered.
        """
        self._add_signal_handler(signal.SIGINT, self.interrupt)
        self._add_signal_handler(signal.SIGTERM, self.interrupt)

    @staticmethod
    def _add_signal_handler(sig, handler):
        def nested_interrupt_builder(*handlers):
            """Build and return an interrupt handler that calls all of *handlers"""

            log.debug("Building interrupt handler: %s", handlers)

            def interrupt(signal_, _stack_frame):
                for handler in handlers:
                    # Only call the handler if it is a callable fn that is _not_ a default handler
                    log.debug("Running interrupt handler: %s", handler)
                    if (
                        handler
                        and callable(handler)
                        and handler != signal.SIG_DFL
                        and handler is not signal.default_int_handler
                    ):
                        handler(signal_, _stack_frame)

            return interrupt

        try:
            signal.signal(sig, nested_interrupt_builder(handler, signal.getsignal(sig)))
        except ValueError:
            log.info(
                "Unable to register signal handler. Server was started from a non-main thread."
            )

    def _shut_down_model_manager(self):
        """Shared utility for shutting down the model manager"""
        ModelManager.get_instance().shut_down()

    @abc.abstractmethod
    def start(self, blocking: bool = True):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    # Context manager impl
    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, type_, value, traceback):
        self.stop()
