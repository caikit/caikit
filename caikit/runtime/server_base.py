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
from typing import Optional
import abc
import signal

# Third Party
from opentelemetry import (
    metrics,
    trace
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from prometheus_client import start_http_server

# First Party
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit

log = alog.use_channel("SERVR-BASE")


class RuntimeServerBase(abc.ABC):
    __doc__ = __doc__
    _metrics_server_started = False
    _meter_provider = False
    _trace_provider = False

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

    @classmethod
    def _create_meter_provider(cls) -> None:
        """Create a single instance of the OpenTelemetry meter provider based on configuration"""
        if not cls._meter_provider:
            log.info(
                "Serving OpenTelemetry metrics on port %s",
                get_config().runtime.metrics.port,
            )
            console_metric_exporter = PeriodicExportingMetricReader(ConsoleMetricExporter(),
                                                export_interval_millis=5000)
            otlp_metric_exporter = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=5000)
            cls.meter_provider = MeterProvider(metric_readers=[console_metric_exporter, otlp_metric_exporter],
                                    resource=Resource.create({
                        "service.name": "caikit-runtime",
                    }),)

            # Sets the global default meter provider
            metrics.set_meter_provider(cls.meter_provider)
            cls._meter_provider = True

    @classmethod
    def _create_trace_provider(cls) -> None:
        """Create a single instance of the OpenTelemetry trace provider based on configuration"""
        if not cls._trace_provider:
            log.info(
                "Serving OpenTelemetry trace on port %s",
                get_config().runtime.metrics.port,
            )
            cls.trace_provider = TracerProvider(resource=Resource.create({
                        "service.name": "caikit-runtime",
                    }),)
            # Sets the global default trace provider
            trace.set_tracer_provider(cls.trace_provider)
            cls.trace_provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
            cls.trace_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter())
            )
            cls._trace_provider = True

    def _intercept_interrupt_signal(self) -> None:
        """intercept signal handler"""
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)

    def interrupt(self, signal_, _stack_frame):
        log.info(
            "<RUN87630120I>",
            "Caikit Runtime received interrupt signal %s, shutting down",
            signal_,
        )
        self.stop()

    def _shut_down_model_manager(self):
        """Shared utility for shutting down the model manager"""
        ModelManager.get_instance().shut_down()

    @classmethod
    def _shutdown_meter_provider(cls) -> None:
        """Shutdown the meter provider and flush the metrics"""
        if cls._meter_provider:
            cls.meter_provider.force_flush()
            cls.meter_provider.shutdown()

    @classmethod
    def _shutdown_trace_provider(cls) -> None:
        """Shutdown the meter provider and flush the metrics"""
        if cls._trace_provider:
            cls.trace_provider.force_flush()
            cls.trace_provider.shutdown()

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
